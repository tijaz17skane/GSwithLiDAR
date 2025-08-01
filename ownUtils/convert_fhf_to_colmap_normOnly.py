import argparse
import json
import csv
import os
import laspy

def parse_args():
    parser = argparse.ArgumentParser(description="Convert FHF dataset to COLMAP format with normalization (translations and LAS points only).")
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
            cam_params[cam_id] = {
                'sensor_id': sensor_id,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'model': 'PINHOLE',
                'width': int(row.get('width', 2452)),
                'height': int(row.get('height', 1840))
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

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # Calibration
    sensor_to_camid, cam_params = read_calibration(args.calib)
    # LAS (use laspy for binary LAS)
    points, min_x, min_y, min_z = read_las_laspy(args.las)
    # Normalize LAS points
    norm_points = []
    for pt in points:
        x, y, z, r, g, b = pt
        norm_points.append([x - min_x, y - min_y, z - min_z, r, g, b])
    # Write points3D.txt
    with open(os.path.join(args.outdir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        for i, pt in enumerate(norm_points):
            x, y, z, r, g, b = pt
            r = min(max(int(r) // 256, 0), 255)
            g = min(max(int(g) // 256, 0), 255)
            b = min(max(int(b) // 256, 0), 255)
            f.write(f'{i+1} {x} {y} {z} {r} {g} {b} 1.0\n')
    # Write cameras.txt (intrinsics and resolution are NOT normalized)
    with open(os.path.join(args.outdir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for cam_id, params in cam_params.items():
            f.write(f'{cam_id} {params["model"]} {params["width"]} {params["height"]} {params["fx"]} {params["fy"]} {params["cx"]} {params["cy"]}\n')
    # Write images.txt (normalize translation only)
    with open(args.meta) as f:
        meta = json.load(f)
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
            t_norm = [float(t[0]) - min_x, float(t[1]) - min_y, float(t[2]) - min_z]
            q = img['pose']['orientation_xyzw']
            if len(q) != 4:
                continue
            qx, qy, qz, qw = q
            f.write(f'{image_id} {qw} {qx} {qy} {qz} {t_norm[0]} {t_norm[1]} {t_norm[2]} {cam_id} {img["path"]}\n\n')
            image_id += 1
if __name__ == '__main__':
    main()
