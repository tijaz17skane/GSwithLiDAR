import os
import json
import argparse
import numpy as np
import cv2
from datetime import datetime, timezone
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(description="Convert spherical images to flat images with calibration and meta output.")
    parser.add_argument('--spherical_images_folder', type=str, default='/mnt/data/tijaz/datasets/section_3useful/spherical_images', help='Folder containing input spherical images')
    parser.add_argument('--meta_json', type=str, default='/mnt/data/tijaz/datasets/section_3useful/meta_filtered_max.json', help='Input meta.json file')
    parser.add_argument('--fov', type=float, default=60.0, help='Field of view for flat images (degrees)')
    parser.add_argument('--skip_angles', type=str, default="lbf:180,240,300;lbb:180,240,300", help='Angles to skip for each camera, e.g. \"lbf:180,240,300;lbb:180,240,300\"')
    parser.add_argument('--flat_images_folder', type=str, default='/mnt/data/tijaz/datasets/section_3useful/flat_images2', help='Output folder for flat images')
    parser.add_argument('--calibration_out', type=str, default='/mnt/data/tijaz/datasets/section_3useful/calibrations_flat.csv', help='Output calibration CSV file')
    parser.add_argument('--meta_out', type=str, default='/mnt/data/tijaz/datasets/section_3useful/meta_flat3.json', help='Output meta JSON file')
    return parser.parse_args()


def rotate_quaternion_by_yaw(quat_xyzw, yaw_degrees):
    """Rotate a quaternion to convert from vertical ladybug to horizontal camera."""
    # Convert quaternion to scipy Rotation object
    # Note: scipy uses xyzw order, so we're good
    original_rot = R.from_quat(quat_xyzw)
    
    # First, rotate -90 degrees around X-axis to convert from vertical to horizontal
    # This makes the camera point forward instead of upward
    x_rot = R.from_rotvec([np.deg2rad(-90), 0, 0])
    
    # Then apply yaw rotation around Y-axis for the horizontal direction
    yaw_rad = np.deg2rad(yaw_degrees)
    yaw_rot = R.from_rotvec([0, yaw_rad, 0])
    
    # Combine rotations: original * x_vertical_to_horizontal * yaw
    new_rot = original_rot * x_rot * yaw_rot
    
    # Return as xyzw quaternion
    return new_rot.as_quat()


def equirectangular_to_perspective(equi, fov, theta, phi, out_hw=(1024, 1024)):
    """Convert equirectangular image to perspective projection."""
    h, w = out_hw
    fov = np.deg2rad(fov)

    # pixel coordinates in output image
    x = np.linspace(-np.tan(fov/2), np.tan(fov/2), w)
    y = np.linspace(-np.tan(fov/2), np.tan(fov/2), h)
    xv, yv = np.meshgrid(x, -y)

    zv = np.ones_like(xv)
    dirs = np.stack([xv, yv, zv], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # rotation matrices: yaw (phi) around Y-axis, pitch (theta) around X-axis
    phi_r = np.deg2rad(phi)
    theta_r = np.deg2rad(theta)
    R_yaw = np.array([
        [ np.cos(phi_r), 0, np.sin(phi_r)],
        [ 0,            1, 0            ],
        [-np.sin(phi_r), 0, np.cos(phi_r)]
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(theta_r), -np.sin(theta_r)],
        [0, np.sin(theta_r),  np.cos(theta_r)]
    ])
    R = R_yaw @ R_pitch
    dirs = dirs @ R.T

    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1, 1))

    equ_h, equ_w = equi.shape[:2]
    uf = (lon / (2*np.pi) + 0.5) * equ_w
    vf = (0.5 - lat / np.pi) * equ_h

    persp = cv2.remap(equi, uf.astype(np.float32), vf.astype(np.float32),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return persp


def main():
    args = parse_args()
    os.makedirs(args.flat_images_folder, exist_ok=True)

    with open(args.meta_json, "r") as f:
        meta = json.load(f)

    skip_dict = {}
    if args.skip_angles:
        for group in args.skip_angles.split(";"):
            cam, vals = group.split(":")
            skip_dict[cam] = set(map(int, vals.split(",")))

    new_entries = []
    calib_entries = []
    calib_cameras = set()  # Track which camera IDs we've already calibrated

    for entry in meta["spherical_images"]:
        img_path = os.path.join(args.spherical_images_folder, os.path.basename(entry["path"]))
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: {img_path} could not be loaded")
            continue

        num_views = int(360 // args.fov)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        sensor_id = entry["sensor_id"]
        if sensor_id == "ladybug_front":
            prefix = "lbf"
        elif sensor_id == "ladybug_back":
            prefix = "lbb"
        else:
            prefix = sensor_id  # fallback

        for i in range(num_views):
            yaw = i * args.fov
            if prefix in skip_dict and int(yaw) in skip_dict[prefix]:
                continue

            pitch = 0.0
            persp = equirectangular_to_perspective(img, args.fov, theta=pitch, phi=yaw)

            out_name = f"{prefix}_{int(yaw)}of360_fov{int(args.fov)}_{base_name}.jpg"
            out_path = os.path.join(args.flat_images_folder, out_name)
            cv2.imwrite(out_path, persp)

            camera_id = f"{prefix}_{int(yaw)}of360"
            new_entry = {
                "sensor_id": camera_id,
                "path": f"images/{out_name}",
                "time_stamp": entry["time_stamp"],
                "pose": {
                    "translation": entry["pose"]["translation"],
                    "orientation_xyzw": rotate_quaternion_by_yaw(
                        entry["pose"]["orientation_xyzw"], yaw
                    ).tolist()
                }
            }
            new_entries.append(new_entry)

            # Calculate intrinsics for this specific crop (only once per camera ID)
            if camera_id not in calib_cameras:
                fx = fy = (persp.shape[1] / 2) / np.tan(np.deg2rad(args.fov / 2))
                cx, cy = persp.shape[1] / 2, persp.shape[0] / 2
                calib_entries.append({
                    "sensor_id": camera_id,
                    "focal_length_px_x": float(fx),
                    "focal_length_px_y": float(fy),
                    "principal_point_px_x": float(cx),
                    "principal_point_px_y": float(cy)
                })
                calib_cameras.add(camera_id)

    # Save meta_out - merge with original meta.json content
    out_meta = meta.copy()  # Start with original content
    if "images" in out_meta:
        out_meta["images"].extend(new_entries)  # Add new flat images to existing images
    else:
        out_meta["images"] = new_entries  # Create images list if it doesn't exist
    with open(args.meta_out, "w") as f:
        json.dump(out_meta, f, indent=2)

    # Save calibration CSV with required schema
    import csv
    now_iso = datetime.now(timezone.utc).isoformat()
    header = [
        "",
        "id",
        "sensor_id",
        "ipm_ignore",
        "intr_calibration_date",
        "focal_length_px_x",
        "focal_length_px_y",
        "principal_point_px_x",
        "principal_point_px_y",
        "lens_distortion_calibration_date",
        "calibration_type",
        "k1",
        "k2",
        "k3",
        "p1",
        "p2",
        "s1",
        "s2"
    ]
    with open(args.calibration_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, row in enumerate(calib_entries):
            writer.writerow([
                idx,
                idx,
                row["sensor_id"],
                False,
                now_iso,
                row["focal_length_px_x"],
                row["focal_length_px_y"],
                row["principal_point_px_x"],
                row["principal_point_px_y"],
                now_iso,
                "opencv",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ])

    print(f"Saved {len(new_entries)} flat images to {args.flat_images_folder}")


if __name__ == "__main__":
    main()
