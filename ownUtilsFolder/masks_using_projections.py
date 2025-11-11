#!/usr/bin/env python3
"""
Generate image masks using LiDAR point reprojections.
Creates masks that define an envelope for all reprojected points in images.
"""

import numpy as np
import json
import cv2
import laspy
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from PIL import Image
import argparse
from tqdm import tqdm

# Import the world2cam function
import sys
sys.path.append('/mnt/data/tijaz/gaussian-splatting/ownUtilsFolder')
from cam_world_conversions import world2cam


def load_las_points(las_path):
    """Load LiDAR points from LAS file."""
    print(f"Loading LAS file: {las_path}")
    las = laspy.read(las_path)
    
    # Extract XYZ coordinates
    points = np.vstack([las.x, las.y, las.z]).T
    print(f"Loaded {len(points)} LiDAR points")
    print(f"Point bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"              Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"              Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    return points


def load_meta_data(meta_path):
    """Load camera poses from meta.json."""
    print(f"Loading meta data: {meta_path}")
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    # Check if meta_data is a list or dict
    if isinstance(meta_data, dict):
        # If it's a dict, it might have a key containing the list
        print(f"Meta data is a dictionary with keys: {list(meta_data.keys())}")
        # Look for common keys that might contain the image data
        for key in ['images', 'data', 'poses', 'frames']:
            if key in meta_data:
                meta_data = meta_data[key]
                print(f"Using data from key '{key}'")
                break
        else:
            # If no common key found, use the dict as is
            print("No common key found, trying to use dict directly")
    
    if isinstance(meta_data, list):
        print(f"Loaded {len(meta_data)} camera poses")
        # Verify structure of first item
        if len(meta_data) > 0:
            first_item = meta_data[0]
            print(f"First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
            if isinstance(first_item, dict) and 'pose' in first_item:
                print(f"Pose keys: {list(first_item['pose'].keys())}")
    else:
        print(f"Meta data type: {type(meta_data)}")
        if isinstance(meta_data, dict):
            print(f"Keys: {list(meta_data.keys())}")
        raise TypeError(f"Expected list of poses, got {type(meta_data)}")
    
    return meta_data

def load_calibration_data(calibration_path):
    """Load camera calibration data from CSV file."""
    # Try multiple possible calibration file locations
    possible_paths = [
        calibration_path,
        '/mnt/data/tijaz/data/section_3/tabular/calibration.csv',
        '/mnt/data/tijaz/data/section_3useful/calibration.csv',
        '/mnt/data/tijaz/data/section_3mappedOnColmap/calibration.csv'
    ]
    
    for path in possible_paths:
        try:
            calibration_df = pd.read_csv(path)
            print(f"Loaded calibration data from: {path}")
            
            # Create a dictionary mapping sensor names to calibration data
            calibration = {}
            for _, row in calibration_df.iterrows():
                sensor_name = row['sensor_id']  # Use sensor_id column
                calibration[sensor_name] = {
                    'focal_x': row['focal_length_px_x'],
                    'focal_y': row['focal_length_px_y'],
                    'principal_x': row['principal_point_px_x'],
                    'principal_y': row['principal_point_px_y'],
                    'k1': row['k1'],
                    'k2': row['k2'],
                    'p1': row['p1'],
                    'p2': row['p2'],
                    'k3': row['k3']
                }
            
            print(f"Loaded calibration for sensors: {list(calibration.keys())}")
            return calibration
        
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading calibration data from {path}: {e}")
            continue
    
    print("Warning: No calibration file found in any expected location")
    return None


def normalize_coordinates(points, camera_poses):
    """
    Normalize coordinates by finding a common translation offset.
    Apply the same normalization to both points and camera poses.
    """
    # Find approximate center of points
    point_center = np.mean(points, axis=0)
    
    # Find approximate center of camera positions
    camera_positions = []
    for pose_data in camera_poses:
        pos = np.array(pose_data['pose']['translation'])
        camera_positions.append(pos)
    camera_center = np.mean(camera_positions, axis=0)
    
    # Use the average as the normalization offset
    offset = (point_center + camera_center) / 2
    print(f"Normalization offset: [{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}]")
    
    # Normalize points
    points_norm = points - offset
    
    # Normalize camera poses
    poses_norm = []
    for pose_data in camera_poses:
        pose_norm = pose_data.copy()
        pos = np.array(pose_data['pose']['translation']) - offset
        pose_norm['pose']['translation'] = pos.tolist()
        poses_norm.append(pose_norm)
    
    print(f"Normalized point bounds: X[{points_norm[:, 0].min():.2f}, {points_norm[:, 0].max():.2f}]")
    print(f"                        Y[{points_norm[:, 1].min():.2f}, {points_norm[:, 1].max():.2f}]")
    print(f"                        Z[{points_norm[:, 2].min():.2f}, {points_norm[:, 2].max():.2f}]")
    
    return points_norm, poses_norm, offset


def quaternion_to_rotation_matrix(quat_xyzw):
    """Convert quaternion (x,y,z,w) to rotation matrix."""
    x, y, z, w = quat_xyzw
    return R.from_quat([x, y, z, w]).as_matrix()


def world_to_camera_transform(points_world, translation, quaternion):
    """Transform world coordinates to camera coordinates using world2cam function."""
    # Extract quaternion components (input is xyzw, function expects wxyz)
    qx, qy, qz, qw = quaternion
    tx, ty, tz = translation
    
    # Get the camera transformation parameters
    t_cam, R_cam, q_cam = world2cam(qw, qx, qy, qz, tx, ty, tz)
    
    # Transform points: x_camera = R_cam @ x_world + t_cam
    points_camera = (R_cam @ points_world.T).T + t_cam
    
    return points_camera


def project_points_to_image(points_camera, intrinsic_matrix):
    """Project 3D camera points to 2D image coordinates."""
    # Only keep points in front of camera (positive Z)
    valid_mask = points_camera[:, 2] > 0
    valid_points = points_camera[valid_mask]
    
    if len(valid_points) == 0:
        return np.array([]), np.array([]), valid_mask
    
    # Project to image plane
    # [u, v, 1]^T = K @ [X, Y, Z]^T / Z
    homogeneous = valid_points / valid_points[:, 2:3]  # Normalize by Z
    projected = (intrinsic_matrix @ homogeneous.T).T
    
    # Extract pixel coordinates
    pixels = projected[:, :2]
    depths = valid_points[:, 2]
    
    return pixels, depths, valid_mask


def get_camera_intrinsics(image_name, calibration_data, image_width=2448, image_height=2048):
    """Get camera intrinsics from calibration data based on image name."""
    # Extract sensor name from image filename
    # Assuming format like "front_12345.png" or similar
    sensor_name = None
    for sensor in ['front', 'left', 'right', 'retro']:
        if sensor in image_name.lower():
            sensor_name = sensor
            break
    
    # Check if calibration data is available
    if calibration_data is None or sensor_name is None or sensor_name not in calibration_data:
        if calibration_data is None:
            print(f"Warning: No calibration data available, using estimated intrinsics for {image_name}")
        else:
            print(f"Warning: Could not find calibration for sensor '{sensor_name}' in {image_name}, using estimated intrinsics")
        
        # Fallback to estimation
        focal_length = max(image_width, image_height) * 0.8
        camera_matrix = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ])
        dist_coeffs = np.zeros(5)
        return camera_matrix, dist_coeffs
    
    # Use calibration data
    calib = calibration_data[sensor_name]
    camera_matrix = np.array([
        [calib['focal_x'], 0, calib['principal_x']],
        [0, calib['focal_y'], calib['principal_y']],
        [0, 0, 1]
    ])
    
    # OpenCV distortion coefficients [k1, k2, p1, p2, k3]
    dist_coeffs = np.array([
        calib['k1'], calib['k2'], calib['p1'], calib['p2'], calib['k3']
    ])
    
    return camera_matrix, dist_coeffs


def create_height_mask(pixels, depths, image_shape, subsample_factor=10):
    """
    Create a mask based on the highest LiDAR points in each pixel column.
    Uses only the highest points to create an envelope mask.
    """
    height, width = image_shape[:2]
    
    # Filter pixels within image bounds
    valid_mask = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    
    if not np.any(valid_mask):
        return np.zeros((height, width), dtype=np.uint8)
    
    valid_pixels = pixels[valid_mask]
    valid_depths = depths[valid_mask]
    
    # Convert to integer pixel coordinates
    pixel_coords = np.round(valid_pixels).astype(int)
    
    # For each column (x-coordinate), find the highest point (minimum depth/closest to camera)
    # or maximum z-coordinate in camera frame, depending on your preference
    
    # Create a grid to store the minimum depth for each pixel
    depth_grid = np.full((height, width), np.inf)
    
    # Update grid with minimum depths (closest points)
    for i in range(len(pixel_coords)):
        x, y = pixel_coords[i]
        if 0 <= x < width and 0 <= y < height:
            depth_grid[y, x] = min(depth_grid[y, x], valid_depths[i])
    
    # Create height envelope
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # For each column, find the highest projected point and create mask below it
    for x in range(0, width, subsample_factor):  # Subsample for efficiency
        column_depths = depth_grid[:, x]
        valid_y_indices = np.where(column_depths != np.inf)[0]
        
        if len(valid_y_indices) > 0:
            # Find the highest point (smallest y-coordinate, assuming y increases downward)
            highest_y = np.min(valid_y_indices)
            # Create mask from this point downward
            mask[highest_y:, x:x+subsample_factor] = 255
    
    # Smooth the mask
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    
    return mask


def process_image(image_path, pose_data, points_lidar, output_dir, calibration_data, fov_degrees=60):
    """Process a single image to generate its mask."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"Processing {image_path.name} ({width}x{height})")
    
    # Get camera pose
    translation = np.array(pose_data['pose']['translation'])
    orientation_xyzw = pose_data['pose']['orientation_xyzw']
    
    print(f"Camera position: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
    print(f"Camera orientation (xyzw): {orientation_xyzw}")
    
    # Get camera intrinsics from calibration data
    camera_matrix, dist_coeffs = get_camera_intrinsics(image_path.name, calibration_data, width, height)
    print(f"Camera intrinsics focal length: {camera_matrix[0,0]:.2f}")
    
    # Transform LiDAR points to camera coordinates
    points_camera = world_to_camera_transform(points_lidar, translation, orientation_xyzw)
    
    # Filter points that are in front of the camera and within reasonable distance
    front_mask = points_camera[:, 2] > 0.1  # At least 10cm in front
    distance_mask = points_camera[:, 2] < 100.0  # Within 100m
    valid_mask = front_mask & distance_mask
    
    print(f"Points in front of camera: {np.sum(front_mask)}/{len(points_camera)}")
    print(f"Points within distance: {np.sum(valid_mask)}/{len(points_camera)}")
    
    if np.sum(valid_mask) == 0:
        print(f"Warning: No valid points for image {image_path.name}")
        mask = np.zeros((height, width), dtype=np.uint8)
        mask_filename = image_path.stem + '_mask.png'
        mask_path = output_dir / mask_filename
        cv2.imwrite(str(mask_path), mask)
        return
    
    # Use only valid points
    points_camera_valid = points_camera[valid_mask]
    
    # Project to image
    pixels, depths, _ = project_points_to_image(points_camera_valid, camera_matrix)
    
    if len(pixels) == 0:
        print(f"Warning: No points projected for image {image_path.name}")
        mask = np.zeros((height, width), dtype=np.uint8)
    else:
        # Create height-based mask
        mask = create_height_mask(pixels, depths, (height, width))
        print(f"Processed {len(pixels)} projected points for {image_path.name}")
        
        # Count how many pixels are within image bounds
        in_bounds = np.sum(
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        print(f"Points within image bounds: {in_bounds}/{len(pixels)}")
    
    # Save mask
    mask_filename = image_path.stem + '_mask.png'
    mask_path = output_dir / mask_filename
    cv2.imwrite(str(mask_path), mask)
    
    # Save visualization with better debugging
    vis_image = image.copy()
    if len(pixels) > 0:
        valid_pixels = pixels[
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        ]
        
        # Draw projected points
        for i, pixel in enumerate(valid_pixels[::max(1, len(valid_pixels)//1000)]):  # Subsample for visualization
            cv2.circle(vis_image, tuple(pixel.astype(int)), 3, (0, 255, 0), -1)
        
        # Draw some text info
        cv2.putText(vis_image, f"Points: {len(valid_pixels)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    vis_filename = image_path.stem + '_projection_vis.jpg'
    vis_path = output_dir / vis_filename
    cv2.imwrite(str(vis_path), vis_image)


def main():
    parser = argparse.ArgumentParser(description="Generate image masks using LiDAR reprojections")
    parser.add_argument('--las_file', type=str, 
                       default='/mnt/data/tijaz/data/section_3withMasks/annotated_ftth.las',
                       help='Path to LAS file')
    parser.add_argument('--meta_file', type=str,
                       default='/mnt/data/tijaz/data/section_3withMasks/metaFiltered.json',
                       help='Path to meta.json file')
    parser.add_argument('--images_dir', type=str,
                       default='/mnt/data/tijaz/data/section_3withMasks/images',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/data/tijaz/data/section_3withMasks/masks',
                       help='Output directory for masks')
    parser.add_argument('--calibration_file', type=str,
                       default='/mnt/data/tijaz/data/section_3withMasks/calibration.csv',
                       help='Path to calibration.csv file')
    parser.add_argument('--fov', type=float, default=60.0,
                       help='Camera field of view in degrees (for intrinsics estimation)')
    parser.add_argument('--max_points', type=int, default=1000000,
                       help='Maximum number of LiDAR points to use (for memory efficiency)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    points_lidar = load_las_points(args.las_file)
    meta_data = load_meta_data(args.meta_file)
    calibration_data = load_calibration_data(args.calibration_file)
    
    # Subsample points if too many
    if len(points_lidar) > args.max_points:
        print(f"Subsampling {len(points_lidar)} points to {args.max_points}")
        indices = np.random.choice(len(points_lidar), args.max_points, replace=False)
        points_lidar = points_lidar[indices]
    
    # Normalize coordinates
    points_norm, poses_norm, offset = normalize_coordinates(points_lidar, meta_data)
    
    print(f"\nProcessing {len(poses_norm)} images...")
    
    # Process each image
    images_dir = Path(args.images_dir)
    processed_count = 0
    
    for pose_data in tqdm(poses_norm, desc="Processing images"):
        image_path = images_dir / Path(pose_data['path']).name
        
        if image_path.exists():
            process_image(image_path, pose_data, points_norm, output_dir, calibration_data, args.fov)
            processed_count += 1
        else:
            print(f"Warning: Image not found: {image_path}")
    
    print(f"\nCompleted! Processed {processed_count} images.")
    print(f"Masks saved to: {output_dir}")


if __name__ == "__main__":
    main()