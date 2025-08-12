import argparse
import json
import csv
import os
import laspy
from scipy.spatial.transform import Rotation as R
from geo_tabulator_vis_chunk import save_basis_vectors_ply

def parse_args():
    parser = argparse.ArgumentParser(description="Convert FHF dataset to COLMAP format with normalization (translations and LAS points only).")
    parser.add_argument('--meta', required=True, help='Path to meta.json')
    parser.add_argument('--calib', required=True, help='Path to calibration.csv')
    parser.add_argument('--las', required=True, help='Path to annotated_ftth.las')
    parser.add_argument('--outdir', required=True, help='Output directory for COLMAP files')
    parser.add_argument('--extrinsics-type', choices=['cam_to_world', 'world_to_cam'], default='cam_to_world', help='Type of extrinsics in meta.json: cam_to_world (default) or world_to_cam')
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
    points, min_x_las, min_y_las, min_z_las = read_las_laspy(args.las)

    # Find min values for camera translations
    import numpy as np
    with open(args.meta) as f:
        meta = json.load(f)
    cam_translations = []
    for img in meta['images']:
        if 'pose' in img and 'translation' in img['pose']:
            cam_translations.append(img['pose']['translation'])
    cam_translations = np.array(cam_translations, dtype=np.float64)
    min_x_cam = np.min(cam_translations[:, 0])
    min_y_cam = np.min(cam_translations[:, 1])
    min_z_cam = np.min(cam_translations[:, 2])

    # Use the minimums from both LAS and camera translations
    min_x = min(min_x_las, min_x_cam)
    min_y = min(min_y_las, min_y_cam)
    min_z = min(min_z_las, min_z_cam)

    # Normalize LAS points
    norm_points = []
    for pt in points:
        x, y, z, r, g, b = pt
        norm_points.append([x - min_x, y - min_y, z - min_z, r, g, b])

    # Write points3D.txt
    points3d_xyz = []
    with open(os.path.join(args.outdir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        for i, pt in enumerate(norm_points):
            x, y, z, r, g, b = pt
            points3d_xyz.append([x, y, z])
            r = min(max(int(r) // 256, 0), 255)
            g = min(max(int(g) // 256, 0), 255)
            b = min(max(int(b) // 256, 0), 255)
            f.write(f'{i+1} {x} {y} {z} {r} {g} {b} 1.0\n')
    points3d_xyz = np.array(points3d_xyz)
    print(f"points3D.txt ranges: x=[{points3d_xyz[:,0].min():.3f}, {points3d_xyz[:,0].max():.3f}], y=[{points3d_xyz[:,1].min():.3f}, {points3d_xyz[:,1].max():.3f}], z=[{points3d_xyz[:,2].min():.3f}, {points3d_xyz[:,2].max():.3f}]")

    # Write cameras.txt (intrinsics and resolution are NOT normalized)
    with open(os.path.join(args.outdir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for cam_id, params in cam_params.items():
            f.write(f'{cam_id} {params["model"]} {params["width"]} {params["height"]} {params["fx"]} {params["fy"]} {params["cx"]} {params["cy"]}\n')

    # Write images.txt (normalize translation only)
    
    images_txt_tx = []
    images_txt_ty = []
    images_txt_tz = []
    images_txt_qx = []
    images_txt_qy = []
    images_txt_qz = []
    images_txt_qw = []
    with open(os.path.join(args.outdir, 'images.txt'), 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        image_id = 1
        #poonts3d_xyz.tolist() = 

        trajectory= {"positions": [], "quaternions": []}  # Initialize trajectory for visualization

        for img in meta['images']:
            if 'pose' not in img or 'translation' not in img['pose'] or 'orientation_xyzw' not in img['pose']:
                continue
            sensor_id = img['sensor_id']
            cam_id = sensor_to_camid[sensor_id]
            t = np.array(img['pose']['translation'], dtype=np.float64)
            q = img['pose']['orientation_xyzw']
            if len(q) != 4:
                continue
            qx, qy, qz, qw = q
            # Normalize translation using global min values
            t_norm = t - np.array([min_x, min_y, min_z], dtype=np.float64)
            # Convert extrinsics to COLMAP convention
            if args.extrinsics_type == 'cam_to_world':
                # t is camera center in world coordinates, so use as is
                #np.linalg.inv(W2C)
                tx, ty, tz = t_norm.tolist()
                # COLMAP expects world-to-camera quaternion, so invert rotation
                rot = R.from_quat([qx, qy, qz, qw])
                rot_inv = rot.inv().as_matrix()
                C2W = np.zeros([4,4])
                C2W[:3, :3] = rot_inv
                C2W[:3, 3] = t_norm
                C2W[3, 3] = 1.0
                W2C = np.linalg.inv(C2W)
                t_norm = W2C[:3, 3]
                # Convert to COLMAP quaternion format
                #rot_inv = rot_inv.inv()
                qx_c, qy_c, qz_c, qw_c = R.from_matrix(rot_inv).as_quat()
                images_txt_tx.append(tx)
                images_txt_ty.append(ty)
                images_txt_tz.append(tz)
                images_txt_qx.append(qx_c)
                images_txt_qy.append(qy_c)
                images_txt_qz.append(qz_c)
                images_txt_qw.append(qw_c)
                f.write(f'{image_id} {qw_c} {qx_c} {qy_c} {qz_c} {tx} {ty} {tz} {cam_id} {img["path"]}\n\n')
                trajectory['positions'].append(t_norm.tolist())
                trajectory['quaternions'].append([qx_c, qy_c, qz_c, qw_c])
            else:
                # t is world-to-camera translation, need to compute camera center
                Rmat = R.from_quat([ qx, qy, qz, qw]).inv().as_matrix()
                #Rmat= Rmat
                C = Rmat.T @ t_norm
                tx, ty, tz = C.tolist()
                images_txt_tx.append(tx)
                images_txt_ty.append(ty)
                images_txt_tz.append(tz)
                images_txt_qx.append(qx)
                images_txt_qy.append(qy)
                images_txt_qz.append(qz)
                images_txt_qw.append(qw)
                f.write(f'{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {img["path"]}\n\n')
            image_id += 1


    trajectory['positions'] = np.array(trajectory['positions'])
    trajectory['quaternions'] = np.array(trajectory['quaternions'])
    save_basis_vectors_ply(trajectory, os.path.join(args.outdir, 'trajectory.ply'), offset=2)

    images_txt_tx = np.array(images_txt_tx)
    images_txt_ty = np.array(images_txt_ty)
    images_txt_tz = np.array(images_txt_tz)
    images_txt_qx = np.array(images_txt_qx)
    images_txt_qy = np.array(images_txt_qy)
    images_txt_qz = np.array(images_txt_qz)
    images_txt_qw = np.array(images_txt_qw)
    print(f"images.txt ranges: tx=[{images_txt_tx.min():.3f}, {images_txt_tx.max():.3f}], ty=[{images_txt_ty.min():.3f}, {images_txt_ty.max():.3f}], tz=[{images_txt_tz.min():.3f}, {images_txt_tz.max():.3f}]")
    print(f"images.txt quaternion ranges: qx=[{images_txt_qx.min():.3f}, {images_txt_qx.max():.3f}], qy=[{images_txt_qy.min():.3f}, {images_txt_qy.max():.3f}], qz=[{images_txt_qz.min():.3f}, {images_txt_qz.max():.3f}], qw=[{images_txt_qw.min():.3f}, {images_txt_qw.max():.3f}]")
if __name__ == '__main__':
    main()