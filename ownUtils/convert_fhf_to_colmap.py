import argparse
import json
import csv
import os

import laspy

def parse_args():
    parser = argparse.ArgumentParser(description="Convert FHF dataset to COLMAP format.")
    parser.add_argument('--meta', required=True, help='Path to meta.json')
    parser.add_argument('--calib', required=True, help='Path to calibration.csv')
    parser.add_argument('--las', required=True, help='Path to annotated_ftth.las')
    parser.add_argument('--outdir', required=True, help='Output directory for COLMAP files')
    return parser.parse_args()

def read_calibration(calib_path):
    sensor_to_camid = {}
    cam_params = {}
    with open(calib_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            sensor_id = row['sensor_id']
            cam_id = int(row['id']) + 1  # COLMAP camera IDs start at 1
            fx = float(row['focal_length_px_x'])
            fy = float(row['focal_length_px_y'])
            cx = float(row['principal_point_px_x'])
            cy = float(row['principal_point_px_y'])
            # You may want to parse width/height elsewhere if available
            cam_params[cam_id] = {
                'sensor_id': sensor_id,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'model': 'PINHOLE',
                'width': 2452, 
                'height': 1840 
            }
            sensor_to_camid[sensor_id] = cam_id
    return sensor_to_camid, cam_params


def read_las_laspy(las_path):
    las = laspy.read(las_path)
    xs = las.x
    ys = las.y
    zs = las.z
    try:
        rs = las.red
        gs = las.green
        bs = las.blue
    except AttributeError:
        rs = gs = bs = [128] * len(xs)
    min_x, min_y, min_z = xs.min(), ys.min(), zs.min()
    points = [[float(x), float(y), float(z), int(r), int(g), int(b)] for x, y, z, r, g, b in zip(xs, ys, zs, rs, gs, bs)]
    return points, min_x, min_y, min_z

def read_las_text(las_path):
    # Always use text-based LAS/XYZRGB parser
    points = []
    min_x = min_y = min_z = None
    with open(las_path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 6:
                continue
            try:
                x, y, z = map(float, vals[:3])
                r, g, b = map(int, vals[3:6])
            except Exception:
                continue
            points.append([x, y, z, r, g, b])
            if min_x is None or x < min_x:
                min_x = x
            if min_y is None or y < min_y:
                min_y = y
            if min_z is None or z < min_z:
                min_z = z
    if min_x is None or min_y is None or min_z is None:
        min_x = min_y = min_z = 0.0
    return points, min_x, min_y, min_z

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # Calibration
    sensor_to_camid, cam_params = read_calibration(args.calib)
    # LAS (use laspy for binary LAS)
    points, min_x, min_y, min_z = read_las_laspy(args.las)
    # Display initial LAS value ranges
    las_xs = [p[0] for p in points]
    las_ys = [p[1] for p in points]
    las_zs = [p[2] for p in points]
    print(f"LAS file initial X range: min={min(las_xs):.6f}, max={max(las_xs):.6f}")
    print(f"LAS file initial Y range: min={min(las_ys):.6f}, max={max(las_ys):.6f}")
    print(f"LAS file initial Z range: min={min(las_zs):.6f}, max={max(las_zs):.6f}")
    print(f"LAS normalization min values: min_x={min_x:.6f}, min_y={min_y:.6f}, min_z={min_z:.6f}")

    # Meta
    with open(args.meta) as f:
        meta = json.load(f)
    # Display initial meta.json translation ranges
    meta_tx = []
    meta_ty = []
    meta_tz = []
    for img in meta['images']:
        if 'pose' in img and 'translation' in img['pose']:
            t = img['pose']['translation']
            if len(t) == 3:
                meta_tx.append(float(t[0]))
                meta_ty.append(float(t[1]))
                meta_tz.append(float(t[2]))
    if meta_tx:
        print(f"meta.json translation X range: min={min(meta_tx):.6f}, max={max(meta_tx):.6f}")
        print(f"meta.json translation Y range: min={min(meta_ty):.6f}, max={max(meta_ty):.6f}")
        print(f"meta.json translation Z range: min={min(meta_tz):.6f}, max={max(meta_tz):.6f}")

    # First, collect normalized translations for all images
    norm_tx, norm_ty, norm_tz = [], [], []
    all_t_norm = []
    for img in meta['images']:
        if 'pose' not in img or 'translation' not in img['pose']:
            continue
        t = img['pose']['translation']
        t_norm = [float(t[0]) - min_x, float(t[1]) - min_y, float(t[2]) - min_z]
        norm_tx.append(t_norm[0])
        norm_ty.append(t_norm[1])
        norm_tz.append(t_norm[2])
        all_t_norm.append(t_norm)
    # Find offsets to make all translations non-negative
    offset_x = -min(norm_tx) if norm_tx and min(norm_tx) < 0 else 0.0
    offset_y = -min(norm_ty) if norm_ty and min(norm_ty) < 0 else 0.0
    offset_z = -min(norm_tz) if norm_tz and min(norm_tz) < 0 else 0.0
    print(f"Image translation offset to ensure non-negative: x={offset_x}, y={offset_y}, z={offset_z}")
    # Write points3D.txt and collect normalized ranges
    norm_xs, norm_ys, norm_zs = [], [], []
    with open(os.path.join(args.outdir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        for i, pt in enumerate(points):
            x, y, z, r, g, b = pt
            x = x - min_x + offset_x
            y = y - min_y + offset_y
            z = z - min_z + offset_z
            norm_xs.append(x)
            norm_ys.append(y)
            norm_zs.append(z)
            # Scale and clamp RGB to 0-255
            r = min(max(int(r) // 256, 0), 255)
            g = min(max(int(g) // 256, 0), 255)
            b = min(max(int(b) // 256, 0), 255)
            f.write(f'{i+1} {x} {y} {z} {r} {g} {b} 1.0\n')
    print(f"points3D.txt normalized X range: min={min(norm_xs):.6f}, max={max(norm_xs):.6f}")
    print(f"points3D.txt normalized Y range: min={min(norm_ys):.6f}, max={max(norm_ys):.6f}")
    print(f"points3D.txt normalized Z range: min={min(norm_zs):.6f}, max={max(norm_zs):.6f}")

    # Write cameras.txt
    with open(os.path.join(args.outdir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for cam_id, params in cam_params.items():
            f.write(f'{cam_id} {params["model"]} {params["width"]} {params["height"]} {params["fx"]} {params["fy"]} {params["cx"]} {params["cy"]}\n')

    # Write images.txt and collect normalized translation ranges (with offset)
    norm_tx2, norm_ty2, norm_tz2 = [], [], []
    with open(os.path.join(args.outdir, 'images.txt'), 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        image_id = 1
        for img in meta['images']:
            if 'pose' not in img or 'translation' not in img['pose'] or 'orientation_xyzw' not in img['pose']:
                continue
            sensor_id = img['sensor_id']
            cam_id = sensor_to_camid[sensor_id]
            t = img['pose']['translation']
            t_norm = [float(t[0]) - min_x + offset_x, float(t[1]) - min_y + offset_y, float(t[2]) - min_z + offset_z]
            norm_tx2.append(t_norm[0])
            norm_ty2.append(t_norm[1])
            norm_tz2.append(t_norm[2])
            q = img['pose']['orientation_xyzw']
            if len(q) != 4:
                continue
            # COLMAP expects qw, qx, qy, qz (Hamilton convention)
            qx, qy, qz, qw = q
            f.write(f'{image_id} {qw} {qx} {qy} {qz} {t_norm[0]} {t_norm[1]} {t_norm[2]} {cam_id} {img["path"]}\n\n')
            image_id += 1
    if norm_tx2:
        print(f"images.txt normalized TX range: min={min(norm_tx2):.6f}, max={max(norm_tx2):.6f}")
        print(f"images.txt normalized TY range: min={min(norm_ty2):.6f}, max={max(norm_ty2):.6f}")
        print(f"images.txt normalized TZ range: min={min(norm_tz2):.6f}, max={max(norm_tz2):.6f}")

if __name__ == '__main__':
    main()
