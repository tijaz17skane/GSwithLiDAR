#!/usr/bin/env python3
"""
Compare two COLMAP images.txt files and report differences.

This script compares camera poses (odd-numbered rows) from two COLMAP images.txt files,
matching them by image NAME, and reports position and orientation differences.

Camera coordinates (qw, qx, qy, qz, tx, ty, tz) are converted to world coordinates
(position, rotation matrix, and quaternion) using cam2world conversion before computing
all differences. All comparisons are performed in world coordinate system.
"""

import numpy as np
import argparse
import os
import sys

# Add parent directory to path to import cam_world_conversions
sys.path.insert(0, '/mnt/data/tijaz/gaussian-splatting/ownUtilsFolder')
from cam_world_conversions import cam2world


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (in degrees)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.degrees([x, y, z])


def quaternion_angular_difference(q1, q2):
    """
    Compute angular difference between two quaternions in degrees.
    
    Parameters:
    -----------
    q1, q2 : array-like (4,)
        Quaternions in [qw, qx, qy, qz] format
    
    Returns:
    --------
    angle : float
        Angular difference in degrees
    """
    # Normalize quaternions
    q1 = np.array(q1) / np.linalg.norm(q1)
    q2 = np.array(q2) / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    
    # Angular difference
    angle = 2 * np.arccos(dot)
    return np.degrees(angle)


def parse_colmap_images(filepath):
    """
    Parse COLMAP images.txt file.
    
    Returns:
    --------
    cameras : dict
        Dictionary mapping NAME to camera data dict with keys:
        'image_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'camera_id', 'name', 'points2d'
    """
    cameras = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # Parse camera pose line (odd row)
        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            
            # Get POINTS2D line (even row)
            points2d = ''
            if i + 1 < len(lines):
                points2d = lines[i + 1].strip()
            
            cameras[name] = {
                'image_id': image_id,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'camera_id': camera_id,
                'name': name,
                'points2d': points2d
            }
            
            # Skip the POINTS2D line
            i += 2
        else:
            i += 1
    
    return cameras


def compare_cameras(cam_a, cam_b, name):
    """
    Compare two camera poses and return difference metrics.
    
    Converts camera coordinates to world coordinates before comparison.
    
    Returns:
    --------
    metrics : dict
        Dictionary containing difference metrics
    """
    # Convert camera coordinates to world coordinates for A
    pos_a_world, R_a_world, quat_a_world = cam2world(
        cam_a['qw'], cam_a['qx'], cam_a['qy'], cam_a['qz'],
        cam_a['tx'], cam_a['ty'], cam_a['tz']
    )
    
    # Convert camera coordinates to world coordinates for B
    pos_b_world, R_b_world, quat_b_world = cam2world(
        cam_b['qw'], cam_b['qx'], cam_b['qy'], cam_b['qz'],
        cam_b['tx'], cam_b['ty'], cam_b['tz']
    )
    
    # Position difference (in world coordinates)
    pos_diff = pos_b_world - pos_a_world
    pos_dist = np.linalg.norm(pos_diff)
    
    # Quaternion difference (world quaternions)
    angular_diff = quaternion_angular_difference(quat_a_world, quat_b_world)
    
    # Rotation matrices and Euler angles (from world rotation matrices)
    euler_a = rotation_matrix_to_euler(R_a_world)
    euler_b = rotation_matrix_to_euler(R_b_world)
    euler_diff = euler_b - euler_a
    
    return {
        'name': name,
        'pos_diff': pos_diff,
        'pos_dist': pos_dist,
        'angular_diff': angular_diff,
        'euler_a': euler_a,
        'euler_b': euler_b,
        'euler_diff': euler_diff,
        'quat_a': quat_a_world,  # World quaternions
        'quat_b': quat_b_world,  # World quaternions
        'pos_a': pos_a_world,    # World coordinates
        'pos_b': pos_b_world     # World coordinates
    }


def write_comparison_report(output_path, metrics_list, cameras_a, cameras_b):
    """
    Write detailed comparison report to file.
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COLMAP CAMERA POSE COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total cameras in A:        {len(cameras_a)}\n")
        f.write(f"Total cameras in B:        {len(cameras_b)}\n")
        f.write(f"Common cameras (matched):  {len(metrics_list)}\n")
        f.write(f"Cameras only in A:         {len(cameras_a) - len(metrics_list)}\n")
        f.write(f"Cameras only in B:         {len(cameras_b) - len(metrics_list)}\n\n")
        
        if not metrics_list:
            f.write("No common cameras found for comparison.\n")
            return
        
        # Compute statistics
        pos_dists = [m['pos_dist'] for m in metrics_list]
        angular_diffs = [m['angular_diff'] for m in metrics_list]
        
        # Compute bounding box size from all camera positions in both A and B
        all_positions = []
        for m in metrics_list:
            all_positions.append(m['pos_a'])
            all_positions.append(m['pos_b'])
        all_positions = np.array(all_positions)
        
        bbox_min = np.min(all_positions, axis=0)
        bbox_max = np.max(all_positions, axis=0)
        bbox_size = np.linalg.norm(bbox_max - bbox_min)
        
        # Compute percentage differences
        pos_dists_percent = [(dist / bbox_size * 100) if bbox_size > 0 else 0.0 for dist in pos_dists]
        
        f.write("BOUNDING BOX INFORMATION (World Coordinates)\n")
        f.write("-"*80 + "\n")
        f.write(f"Min: [{bbox_min[0]:10.6f}, {bbox_min[1]:10.6f}, {bbox_min[2]:10.6f}]\n")
        f.write(f"Max: [{bbox_max[0]:10.6f}, {bbox_max[1]:10.6f}, {bbox_max[2]:10.6f}]\n")
        f.write(f"Diagonal size: {bbox_size:.6f}\n\n")
        
        f.write("POSITION DIFFERENCES (Euclidean distance in World Coordinates)\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean:    {np.mean(pos_dists):.6f}  ({np.mean(pos_dists_percent):.4f}% of bbox)\n")
        f.write(f"Median:  {np.median(pos_dists):.6f}  ({np.median(pos_dists_percent):.4f}% of bbox)\n")
        f.write(f"Std:     {np.std(pos_dists):.6f}\n")
        f.write(f"Min:     {np.min(pos_dists):.6f}  ({np.min(pos_dists_percent):.4f}% of bbox)\n")
        f.write(f"Max:     {np.max(pos_dists):.6f}  ({np.max(pos_dists_percent):.4f}% of bbox)\n\n")
        
        f.write("ORIENTATION DIFFERENCES (Angular distance in degrees)\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean:    {np.mean(angular_diffs):.4f}°\n")
        f.write(f"Median:  {np.median(angular_diffs):.4f}°\n")
        f.write(f"Std:     {np.std(angular_diffs):.4f}°\n")
        f.write(f"Min:     {np.min(angular_diffs):.4f}°\n")
        f.write(f"Max:     {np.max(angular_diffs):.4f}°\n\n")
        
        # Per-camera details
        f.write("="*80 + "\n")
        f.write("DETAILED PER-CAMERA COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        # Sort by position distance (largest differences first)
        metrics_sorted = sorted(metrics_list, key=lambda x: x['pos_dist'], reverse=True)
        
        for i, m in enumerate(metrics_sorted, 1):
            pos_dist_percent = (m['pos_dist'] / bbox_size * 100) if bbox_size > 0 else 0.0
            
            f.write(f"[{i}/{len(metrics_sorted)}] {m['name']}\n")
            f.write("-"*80 + "\n")
            
            f.write("Position (World Coordinates):\n")
            f.write(f"  A: [{m['pos_a'][0]:10.6f}, {m['pos_a'][1]:10.6f}, {m['pos_a'][2]:10.6f}]\n")
            f.write(f"  B: [{m['pos_b'][0]:10.6f}, {m['pos_b'][1]:10.6f}, {m['pos_b'][2]:10.6f}]\n")
            f.write(f"  Δ: [{m['pos_diff'][0]:10.6f}, {m['pos_diff'][1]:10.6f}, {m['pos_diff'][2]:10.6f}]\n")
            f.write(f"  Distance: {m['pos_dist']:.6f} ({pos_dist_percent:.4f}% of bbox)\n\n")
            
            f.write("Quaternion (World Coordinates):\n")
            f.write(f"  A: [{m['quat_a'][0]:9.6f}, {m['quat_a'][1]:9.6f}, {m['quat_a'][2]:9.6f}, {m['quat_a'][3]:9.6f}]\n")
            f.write(f"  B: [{m['quat_b'][0]:9.6f}, {m['quat_b'][1]:9.6f}, {m['quat_b'][2]:9.6f}, {m['quat_b'][3]:9.6f}]\n")
            f.write(f"  Angular difference: {m['angular_diff']:.4f}°\n\n")
            
            f.write("Euler angles (from World Rotation Matrix, degrees):\n")
            f.write(f"  A: [Roll={m['euler_a'][0]:8.3f}°, Pitch={m['euler_a'][1]:8.3f}°, Yaw={m['euler_a'][2]:8.3f}°]\n")
            f.write(f"  B: [Roll={m['euler_b'][0]:8.3f}°, Pitch={m['euler_b'][1]:8.3f}°, Yaw={m['euler_b'][2]:8.3f}°]\n")
            f.write(f"  Δ: [Roll={m['euler_diff'][0]:8.3f}°, Pitch={m['euler_diff'][1]:8.3f}°, Yaw={m['euler_diff'][2]:8.3f}°]\n\n")
        
        # Cameras only in A
        only_in_a = set(cameras_a.keys()) - set(cameras_b.keys())
        if only_in_a:
            f.write("="*80 + "\n")
            f.write(f"CAMERAS ONLY IN A ({len(only_in_a)})\n")
            f.write("="*80 + "\n")
            for name in sorted(only_in_a):
                f.write(f"  - {name}\n")
            f.write("\n")
        
        # Cameras only in B
        only_in_b = set(cameras_b.keys()) - set(cameras_a.keys())
        if only_in_b:
            f.write("="*80 + "\n")
            f.write(f"CAMERAS ONLY IN B ({len(only_in_b)})\n")
            f.write("="*80 + "\n")
            for name in sorted(only_in_b):
                f.write(f"  - {name}\n")
            f.write("\n")
        
        # Compact one-line-per-image comparison
        f.write("="*80 + "\n")
        f.write("COMPACT COMPARISON (ONE LINE PER IMAGE)\n")
        f.write("="*80 + "\n")
        f.write("Format: Serial | Image Name | Position Distance | % of bbox | Angular Distance\n")
        f.write("-"*80 + "\n")
        
        # Sort by original order (by name)
        metrics_by_name = sorted(metrics_list, key=lambda x: x['name'])
        
        for i, m in enumerate(metrics_by_name, 1):
            pos_dist_percent = (m['pos_dist'] / bbox_size * 100) if bbox_size > 0 else 0.0
            f.write(f"{i:3d}: {m['name']:50s} | Pos: {m['pos_dist']:10.6f} | {pos_dist_percent:7.4f}% | Ang: {m['angular_diff']:8.4f}°\n")
        
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare two COLMAP images.txt files and report differences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  This tool compares camera poses from two COLMAP images.txt files by matching
  them by image NAME and computing position and orientation differences.
  
  Only odd-numbered rows (camera pose lines) are compared. Even-numbered rows
  (POINTS2D data) are ignored.

EXAMPLE:
  python compare_txt.py \\
    --inputA imagesA.txt \\
    --inputB imagesB.txt \\
    --output_txt comparison_report.txt
        """
    )
    
    parser.add_argument('--inputA', type=str, required=True,
                       help='Path to first COLMAP images.txt file')
    parser.add_argument('--inputB', type=str, required=True,
                       help='Path to second COLMAP images.txt file')
    parser.add_argument('--output_txt', type=str, required=True,
                       help='Path to output comparison report file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COLMAP CAMERA POSE COMPARISON")
    print("="*80)
    
    # Load input files
    print(f"\nLoading input A: {args.inputA}")
    if not os.path.exists(args.inputA):
        print(f"ERROR: File not found: {args.inputA}")
        return
    cameras_a = parse_colmap_images(args.inputA)
    print(f"  ✓ Loaded {len(cameras_a)} cameras")
    
    print(f"\nLoading input B: {args.inputB}")
    if not os.path.exists(args.inputB):
        print(f"ERROR: File not found: {args.inputB}")
        return
    cameras_b = parse_colmap_images(args.inputB)
    print(f"  ✓ Loaded {len(cameras_b)} cameras")
    
    # Find common cameras
    print("\nFinding common cameras...")
    common_names = set(cameras_a.keys()) & set(cameras_b.keys())
    print(f"  ✓ Found {len(common_names)} common cameras")
    
    if not common_names:
        print("\n⚠ Warning: No common cameras found between the two files!")
        print("  Check that image filenames (NAME column) match between files.")
    
    # Compare cameras
    print("\nComparing camera poses...")
    metrics_list = []
    for name in common_names:
        metrics = compare_cameras(cameras_a[name], cameras_b[name], name)
        metrics_list.append(metrics)
    
    # Compute summary statistics
    if metrics_list:
        pos_dists = [m['pos_dist'] for m in metrics_list]
        angular_diffs = [m['angular_diff'] for m in metrics_list]
        
        # Compute bounding box for percentage calculation
        all_positions = []
        for m in metrics_list:
            all_positions.append(m['pos_a'])
            all_positions.append(m['pos_b'])
        all_positions = np.array(all_positions)
        bbox_min = np.min(all_positions, axis=0)
        bbox_max = np.max(all_positions, axis=0)
        bbox_size = np.linalg.norm(bbox_max - bbox_min)
        
        print("\n" + "-"*80)
        print("QUICK SUMMARY")
        print("-"*80)
        print(f"Bounding box diagonal:           {bbox_size:.6f}")
        print(f"Position difference (mean):      {np.mean(pos_dists):.6f}  ({np.mean(pos_dists)/bbox_size*100:.4f}% of bbox)")
        print(f"Position difference (max):       {np.max(pos_dists):.6f}  ({np.max(pos_dists)/bbox_size*100:.4f}% of bbox)")
        print(f"Angular difference (mean):       {np.mean(angular_diffs):.4f}°")
        print(f"Angular difference (max):        {np.max(angular_diffs):.4f}°")
    
    # Write report
    print(f"\nWriting detailed report to: {args.output_txt}")
    write_comparison_report(args.output_txt, metrics_list, cameras_a, cameras_b)
    print(f"  ✓ Report saved")
    
    print("\n" + "="*80)
    print("✓ COMPARISON COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
