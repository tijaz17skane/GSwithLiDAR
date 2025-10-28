import argparse
import json
import csv
import os
import laspy
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Convert FHF dataset to COLMAP format with normalization (translations and LAS points only).")
    parser.add_argument('--meta', default='/mnt/data/tijaz/data/AlignData/LiDAROutput/metaFiltered.json', help='Path to meta.json')
    parser.add_argument('--calib', default='/mnt/data/tijaz/data/AlignData/LiDAROutput/calibration.csv', help='Path to calibration.csv')
    parser.add_argument('--las', default='/mnt/data/tijaz/data/AlignData/LiDAROutput/points3D_withoutBB.las', help='Path to points3D_withoutBB.las')
    parser.add_argument('--outdir', default='/mnt/data/tijaz/data/AlignData/LiDAROutput', help='Output directory for COLMAP files')
    parser.add_argument('--extrinsics-type', choices=['cam_to_world','world_to_cam'], default='cam_to_world', help='Type of extrinsics') # cam to world means camera coordinates, images.txt needs to be in camera coordinates as colmap outputs. 

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
    # Vectorized operations for better performance
    xs = las.x
    ys = las.y
    zs = las.z
    
    # Handle color data efficiently
    try:
        rs = las.red
        gs = las.green
        bs = las.blue
    except AttributeError:
        rs = gs = bs = np.full(len(xs), 128, dtype=np.uint8)
    
    # Calculate bounds efficiently
    min_x, min_y, min_z = xs.min(), ys.min(), zs.min()
    
    # Stack coordinates and colors for vectorized processing
    coords = np.column_stack([xs, ys, zs])
    colors = np.column_stack([rs, gs, bs])
    
    return coords, colors, min_x, min_y, min_z

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    # Calibration
    sensor_to_camid, cam_params = read_calibration(args.calib)
    
    # LAS (use laspy for binary LAS) - vectorized
    coords, colors, min_x, min_y, min_z = read_las_laspy(args.las)
    
    # Normalize LAS points efficiently
    norm_coords = coords - np.array([min_x, min_y, min_z])
    
    # Write points3D.txt efficiently
    print("Writing points3D.txt...")
    with open(os.path.join(args.outdir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        
        # Process colors efficiently
        colors_normalized = np.clip(colors // 256, 0, 255)
        
        for i, (coord, color) in enumerate(zip(norm_coords, colors_normalized)):
            x, y, z = coord
            r, g, b = color
            f.write(f'{i+1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 1.0\n')
    
    # Write cameras.txt (intrinsics and resolution are NOT normalized)
    print("Writing cameras.txt...")
    with open(os.path.join(args.outdir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for cam_id, params in cam_params.items():
            f.write(f'{cam_id} {params["model"]} {params["width"]} {params["height"]} {params["fx"]} {params["fy"]} {params["cx"]} {params["cy"]}\n')
    
    # Read metadata
    with open(args.meta) as f:
        meta = json.load(f)
    
    # Prepare data structures for efficient processing
    image_data = []
    camera_positions = []
    
    print("Processing camera poses...")
    for img in meta['images']:
        if 'pose' not in img or 'translation' not in img['pose'] or 'orientation_xyzw' not in img['pose']:
            continue
            
        sensor_id = img['sensor_id']
        if sensor_id not in sensor_to_camid:
            continue
            
        cam_id = sensor_to_camid[sensor_id]
        cam_params_dict = cam_params[cam_id]
        
        t = np.array(img['pose']['translation'], dtype=np.float64)
        q = img['pose']['orientation_xyzw']
        
        if len(q) != 4:
            continue
            
        qx, qy, qz, qw = q
        
        # Normalize translation
        t_norm = t - np.array([min_x, min_y, min_z], dtype=np.float64)
        
        # Extract just the filename from the path
        img_name = os.path.basename(img["path"])
        
        # Convert extrinsics to COLMAP convention
        if args.extrinsics_type == 'cam_to_world':
            # t is camera center in world coordinates
            rot = R.from_quat([qx, qy, qz, qw])
            rot_inv = rot.inv()
            t_colmap = -rot_inv.apply(t_norm)
            tx, ty, tz = t_colmap.tolist()
            qx_c, qy_c, qz_c, qw_c = rot_inv.as_quat()
            
            # Store for images.txt
            image_data.append({
                'id': len(image_data) + 1,
                'qw': qw_c, 'qx': qx_c, 'qy': qy_c, 'qz': qz_c,
                'tx': tx, 'ty': ty, 'tz': tz,
                'cam_id': cam_id,
                'name': img_name
            })
            
            
        else:
            # t is world-to-camera translation, need to compute camera center
            rot = R.from_quat([qx, qy, qz, qw])
            Rmat = rot.as_matrix()
            C = -Rmat.T @ t_norm
            tx, ty, tz = C.tolist()
            
            # Store for images.txt
            image_data.append({
                'id': len(image_data) + 1,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'cam_id': cam_id,
                'name': img_name
            })
            
    # Write images.txt efficiently
    print("Writing images.txt...")
    with open(os.path.join(args.outdir, 'images.txt'), 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write(f'# Number of images: {len(image_data)}\n')
        for img_data in image_data:
            f.write(f'{img_data["id"]} {img_data["qw"]} {img_data["qx"]} {img_data["qy"]} {img_data["qz"]} {img_data["tx"]} {img_data["ty"]} {img_data["tz"]} {img_data["cam_id"]} {img_data["name"]}\n\n')
    
    
    # Print summary statistics efficiently
    print("\n=== SUMMARY ===")
    print(f"Total points: {len(norm_coords):,}")
    print(f"Total cameras: {len(image_data)}")
    
    # Calculate ranges efficiently
    x_coords, y_coords, z_coords = norm_coords[:, 0], norm_coords[:, 1], norm_coords[:, 2]
    print(f"\n=== POINTS3D.TXT RANGES ===")
    print(f"X range: {x_coords.min():.6f} to {x_coords.max():.6f}")
    print(f"Y range: {y_coords.min():.6f} to {y_coords.max():.6f}")
    print(f"Z range: {z_coords.min():.6f} to {z_coords.max():.6f}")
    
    # Calculate camera position ranges
    if camera_positions:
        positions = np.array([cam['position'] for cam in camera_positions])
        tx_coords, ty_coords, tz_coords = positions[:, 0], positions[:, 1], positions[:, 2]
        print(f"\n=== CAMERA POSITIONS RANGES ===")
        print(f"TX range: {tx_coords.min():.6f} to {tx_coords.max():.6f}")
        print(f"TY range: {ty_coords.min():.6f} to {ty_coords.max():.6f}")
        print(f"TZ range: {tz_coords.min():.6f} to {tz_coords.max():.6f}")
    
    # Save normalization transformation matrix
    norm_transform = np.eye(4)
    norm_transform[:3, 3] = [min_x, min_y, min_z]
    np.savetxt(os.path.join(args.outdir, 'normalization_transform.txt'), norm_transform, fmt='%.8f')
    print(f"Normalization transform saved to {os.path.join(args.outdir, 'normalization_transform.txt')}")

if __name__ == '__main__':
    main()
