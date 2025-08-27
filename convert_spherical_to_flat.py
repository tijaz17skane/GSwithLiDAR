import os
import json
import math
import cv2
import numpy as np
from glob import glob
from datetime import datetime
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert spherical images to flat images with calibration and meta output.")
    parser.add_argument('--spherical_images_folder', type=str, required=True)
    parser.add_argument('--meta_json', type=str, required=True)
    parser.add_argument('--fov', type=float, default=60.0, help='Field of view for flat images (degrees)')
    parser.add_argument('--skip_angles', type=str, default="lbf:180,240,300;lbb:180,240,300", help='Angles to skip for each camera, e.g. "lbf:180,240,300;lbb:180,240,300"')
    parser.add_argument('--flat_images_folder', type=str, required=True)
    parser.add_argument('--calibration_out', type=str, required=True)
    parser.add_argument('--meta_out', type=str, required=True)
    return parser.parse_args()

def equirect_to_perspective(img, fov, theta, phi, out_hw=(1024, 1024)):
    # img: input equirectangular image (H, W, 3)
    # fov: field of view in degrees
    # theta: yaw (azimuth) in degrees
    # phi: pitch (elevation) in degrees
    # out_hw: output image size (h, w)
    h, w = out_hw
    fov_rad = math.radians(fov)
    cx, cy = w / 2, h / 2
    fx = fy = w / (2 * math.tan(fov_rad / 2))
    # Build direction vectors for each pixel
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)
    x_cam = (xv - cx) / fx
    y_cam = (yv - cy) / fy
    z_cam = np.ones_like(x_cam)
    dirs = np.stack([x_cam, y_cam, z_cam], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    # Rotation
    theta_rad = math.radians(theta)
    phi_rad = math.radians(phi)
    R_yaw = np.array([
        [math.cos(theta_rad), 0, math.sin(theta_rad)],
        [0, 1, 0],
        [-math.sin(theta_rad), 0, math.cos(theta_rad)]
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(phi_rad), -math.sin(phi_rad)],
        [0, math.sin(phi_rad), math.cos(phi_rad)]
    ])
    R = R_yaw @ R_pitch
    dirs = dirs @ R.T
    # Convert to spherical coordinates
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1, 1))
    # Map to equirectangular
    equ_h, equ_w = img.shape[:2]
    u = (lon / np.pi + 1) / 2 * equ_w
    v = (lat / (0.5 * np.pi) + 0.5) * equ_h
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    persp = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return persp

def main():
    args = parse_args()
    os.makedirs(args.flat_images_folder, exist_ok=True)
    with open(args.meta_json, 'r') as f:
        meta = json.load(f)
    # Find spherical images info
    sph_imgs = meta.get('spherical_images', [])
    if not sph_imgs:
        raise ValueError('No spherical_images found in meta.json')
    # Output meta for flat images
    flat_meta_images = []
    # Calibration CSV header
    calib_header = [
        '', 'id', 'sensor_id', 'ipm_ignore', 'intr_calibration_date',
        'focal_length_px_x', 'focal_length_px_y', 'principal_point_px_x', 'principal_point_px_y',
        'lens_distortion_calibration_date', 'calibration_type',
        'k1', 'k2', 'k3', 'p1', 'p2', 's1', 's2'
    ]
    # For each spherical image, create flat crops
    fov = args.fov
    n_crops = int(360 // fov)
    # Parse skip_angles argument into a dict
    skip_angles = { 'lbf': set(), 'lbb': set() }
    for entry in args.skip_angles.split(';'):
        if ':' in entry:
            cam, angles = entry.split(':')
            skip_angles[cam.strip()] = set(int(a) for a in angles.split(',') if a.strip())
    calib_row_map = {}
    for img_info in sph_imgs:
        img_name = img_info['path']
        sensor_id = img_info['sensor_id']
        img_path = os.path.join(args.spherical_images_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        for i in range(n_crops):
            yaw = int(i * fov)
            pitch = 0  # Only horizontal crops
            cam_short = 'lbf' if 'front' in sensor_id else 'lbb'
            if yaw in skip_angles.get(cam_short, set()):
                continue
            flat_img = equirect_to_perspective(img, fov, yaw, pitch)
            base = os.path.splitext(os.path.basename(img_name))[0]
            out_name = f"{cam_short}_{yaw}of360_fov{int(fov)}_{base}.jpg"
            out_path = os.path.join(args.flat_images_folder, out_name)
            cv2.imwrite(out_path, flat_img)
            # Meta for flat image
            flat_meta_images.append({
                'sensor_id': cam_short,
                'path': f"flat_images/{out_name}",
                'time_stamp': img_info.get('time_stamp', None),
                'pose': img_info.get('pose', None)
            })
            # Calibration row (dummy values, should be replaced with real calibration if available)
            calib_key = f"{cam_short}_{yaw}of360_fov{int(fov)}"
            sensor_id_with_angle = f"{cam_short}_{yaw}of360"
            if calib_key not in calib_row_map:
                calib_row = [
                    i, i, sensor_id_with_angle, False, datetime.now().isoformat(),
                    1455.0, 1455.0, 1024/2, 1024/2, datetime.now().isoformat(),
                    'opencv', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]
                calib_row_map[calib_key] = calib_row
    # Write calibrationOut.csv (one row per cam_short/yaw/fov)
    with open(args.calibration_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(calib_header)
        for key, row in calib_row_map.items():
            writer.writerow(row)
    # Write metaOut.json
    meta_out = {'images': flat_meta_images}
    with open(args.meta_out, 'w') as f:
        json.dump(meta_out, f, indent=2)
if __name__ == '__main__':
    main()
