import argparse
import json
import csv
import os
import laspy
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
from createEmptyDatabase import create_empty_database
import sqlite3
from cam_world_conversions import world2cam

def parse_args():
    parser = argparse.ArgumentParser(description="Convert FHF dataset to COLMAP format with normalization (translations and LAS points only).")
    parser.add_argument('--meta', default='/mnt/data/tijaz/data/section_3mappedOnColmap/metaFiltered.json', help='Path to meta.json')
    parser.add_argument('--calib', default='/mnt/data/tijaz/data/section_3mappedOnColmap/calibration.csv', help='Path to calibration.csv')
    parser.add_argument('--las', default='/mnt/data/tijaz/data/section_3mappedOnColmap/annotated_ftth_downsampled.las', help='Path to 3D point cloud LAS file')
    parser.add_argument('--outdir', default='/mnt/data/tijaz/data/section_3mappedOnColmap/converted2Colmap', help='Output directory for COLMAP files')
    parser.add_argument('--normalize', action='store_true', help='Apply normalization to translation and LAS points')
    parser.add_argument('--images_folder', type=str, default='/mnt/data/tijaz/data/section_3mappedOnColmap/images', help='Path to folder containing images to filter output')

    return parser.parse_args()

'''
    --meta is the meta.json file from fhf dataset
    --calib is the calibration.csv file from fhf dataset inside the tabular folder
    --las is the annotated_ftth.las file from fhf dataset
    --outdir is the output directory to save colmap files
    --normalize applies normalization to translation and LAS points. Without translation the gradients explode, but with this normalization it doesn't work well.
    rather transform it to colmap and it should work better on that scale.
    --images_folder is the path to the folder containing images to filter output. So that we only get poses for the images that are present in the folder. 
'''



MODEL_NAME_TO_ID = {
    'SIMPLE_PINHOLE': 0,
    'PINHOLE': 1,
    'SIMPLE_RADIAL': 2,
    'RADIAL': 3,
    'OPENCV': 4,
    'FULL_OPENCV': 5,
    'FOV': 6,
    'THIN_PRISM_FISHEYE': 7,
    'SIMPLE_RADIAL_FISHEYE': 8,
    'RADIAL_FISHEYE': 9,
    'OPENCV_FISHEYE': 10,
}

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

def write_colmap_database(images_txt_path, cameras_txt_path, db_path):
    # Create empty COLMAP-style database
    create_empty_database(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Insert cameras
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:  # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
                continue

            camera_id = int(parts[0])
            model_str = parts[1]
            model_id = MODEL_NAME_TO_ID.get(model_str)
            if model_id is None:
                raise ValueError(f"Unknown camera model: {model_str}")

            width = int(parts[2])
            height = int(parts[3])
            # Remaining are params; convert to float64 and store as BLOB
            param_vals = list(map(float, parts[4:]))
            params_blob = np.array(param_vals, dtype=np.float64).tobytes()

            prior_focal_length = 1
            c.execute(
                "INSERT OR REPLACE INTO cameras (camera_id, model, width, height, params, prior_focal_length) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (camera_id, model_id, width, height, sqlite3.Binary(params_blob), prior_focal_length),
            )

    # Insert images
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]

            c.execute(
                "INSERT OR REPLACE INTO images "
                "(image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz),
            )

    conn.commit()
    conn.close()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    # Calibration
    sensor_to_camid, cam_params = read_calibration(args.calib)
    
    # LAS (use laspy for binary LAS) - vectorized
    coords, colors, min_x, min_y, min_z = read_las_laspy(args.las)

    # Normalize LAS points efficiently if requested
    if args.normalize:
        norm_coords = coords - np.array([min_x, min_y, min_z])
    else:
        norm_coords = coords.copy()

    # Write points3D.txt efficiently
    print("Writing points3D.txt...")
    with open(os.path.join(args.outdir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
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

    # If images_folder is provided, get set of image names
    images_set = None
    if args.images_folder:
        images_set = set(os.listdir(args.images_folder))

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
        # Normalize translation if requested
        if args.normalize:
            t_norm = t - np.array([min_x, min_y, min_z], dtype=np.float64)
        else:
            t_norm = t.copy()
        
        # Extract just the filename from the path
        img_name = os.path.basename(img["path"])
        # Filter by images_folder if provided
        if images_set is not None and img_name not in images_set:
            continue

        # Convert extrinsics to COLMAP convention using world2cam function
        # t_norm is camera center in world coordinates
        norm_offset = np.array([min_x, min_y, min_z]) if args.normalize else None
        t_cam, R_w2c = world2cam(qw, qx, qy, qz, t_norm[0], t_norm[1], t_norm[2], norm_offset)
        tx, ty, tz = t_cam.tolist()
        qx_c, qy_c, qz_c, qw_c = R.from_matrix(R_w2c).as_quat()
        
        # Store for images.txt
        image_data.append({
            'id': len(image_data) + 1,
            'qw': qw_c, 'qx': qx_c, 'qy': qy_c, 'qz': qz_c,
            'tx': tx, 'ty': ty, 'tz': tz,
            'cam_id': cam_id,
            'name': img_name
        })
            
    # Write images.txt and cameras.txt efficiently
    print("Writing images.txt...")
    images_txt_path = os.path.join(args.outdir, 'images.txt')
    with open(images_txt_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write(f'# Number of images: {len(image_data)}\n')
        for img_data in image_data:
            f.write(f'{img_data["id"]} {img_data["qw"]} {img_data["qx"]} {img_data["qy"]} {img_data["qz"]} {img_data["tx"]} {img_data["ty"]} {img_data["tz"]} {img_data["cam_id"]} {img_data["name"]}\n\n')
    cameras_txt_path = os.path.join(args.outdir, 'cameras.txt')
    # Write to database.db after writing txt files
    db_path = os.path.join(args.outdir, 'database.db')
    write_colmap_database(images_txt_path, cameras_txt_path, db_path)
    print(f"database.db written to {db_path}")
    
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
    
    # Save normalization transformation matrix if requested
    if args.normalize:
        norm_transform = np.eye(4)
        norm_transform[:3, 3] = [min_x, min_y, min_z]
        np.savetxt(os.path.join(args.outdir, 'normalization_transform.txt'), norm_transform, fmt='%.8f')
        print(f"Normalization transform saved to {os.path.join(args.outdir, 'normalization_transform.txt')}")

if __name__ == '__main__':
    main()
