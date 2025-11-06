#!/usr/bin/env python3
"""
Quaternion Debug Tool for SRTaligner.py
Compare quaternions between input and output files to debug rotation issues.
"""

import numpy as np
import argparse
import os
import sys

def parse_colmap_file(filepath):
    """Parse COLMAP images.txt file and extract quaternions and names."""
    data = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments
        if line.startswith('#'):
            i += 1
            continue
        
        # Parse camera pose line
        parts = line.split()
        if len(parts) >= 10:
            image_id = parts[0]
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = parts[8]
            name = parts[9]
            
            data.append({
                'image_id': image_id,
                'quaternion': np.array([qw, qx, qy, qz]),
                'position': np.array([tx, ty, tz]),
                'camera_id': camera_id,
                'name': name
            })
        
        # Skip next line (POINTS2D data) or move to next if at end
        i += 2
    
    return data

def quaternion_difference(q1, q2):
    """
    Compute the angular difference between two quaternions in degrees.
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product (cosine of half-angle)
    dot_product = np.abs(np.dot(q1, q2))
    
    # Clamp to avoid numerical issues
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    # Angular difference in radians, then convert to degrees
    angle_rad = 2 * np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def quaternion_to_euler(q):
    """Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw) in degrees."""
    qw, qx, qy, qz = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees([roll, pitch, yaw])

def main():
    parser = argparse.ArgumentParser(
        description='Debug quaternion differences between input and output files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--inputB', type=str, required=True,
                       help='Target/reference camera poses file (inputB.txt)')
    parser.add_argument('--aligned_output', type=str, required=True,
                       help='Aligned output file (aligned_in_cam.txt)')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples to show in detail')
    parser.add_argument('--threshold_deg', type=float, default=5.0,
                       help='Threshold in degrees to highlight large differences')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QUATERNION DEBUGGING TOOL")
    print("="*80)
    
    # Parse input files
    print(f"Loading target quaternions from: {args.inputB}")
    target_data = parse_colmap_file(args.inputB)
    print(f"  → Loaded {len(target_data)} cameras")
    
    print(f"\nLoading aligned quaternions from: {args.aligned_output}")
    aligned_data = parse_colmap_file(args.aligned_output)
    print(f"  → Loaded {len(aligned_data)} cameras")
    
    # Create dictionaries for fast lookup by name
    target_dict = {item['name']: item for item in target_data}
    aligned_dict = {item['name']: item for item in aligned_data}
    
    # Find common cameras
    common_names = set(target_dict.keys()) & set(aligned_dict.keys())
    common_names = sorted(list(common_names))
    
    print(f"\nFound {len(common_names)} common cameras for comparison")
    
    if len(common_names) == 0:
        print("ERROR: No common cameras found!")
        return
    
    # Compare quaternions
    print("\n" + "-"*80)
    print("QUATERNION COMPARISON ANALYSIS")
    print("-"*80)
    
    differences = []
    
    for name in common_names:
        target_q = target_dict[name]['quaternion']
        aligned_q = aligned_dict[name]['quaternion']
        
        # Compute angular difference
        angle_diff = quaternion_difference(target_q, aligned_q)
        
        differences.append({
            'name': name,
            'target_q': target_q,
            'aligned_q': aligned_q,
            'angle_diff': angle_diff,
            'target_euler': quaternion_to_euler(target_q),
            'aligned_euler': quaternion_to_euler(aligned_q)
        })
    
    # Sort by largest differences
    differences.sort(key=lambda x: x['angle_diff'], reverse=True)
    
    # Statistics
    angle_diffs = [d['angle_diff'] for d in differences]
    mean_diff = np.mean(angle_diffs)
    max_diff = np.max(angle_diffs)
    min_diff = np.min(angle_diffs)
    std_diff = np.std(angle_diffs)
    
    print(f"ANGULAR DIFFERENCE STATISTICS:")
    print(f"  Mean difference:    {mean_diff:.3f}°")
    print(f"  Max difference:     {max_diff:.3f}°")
    print(f"  Min difference:     {min_diff:.3f}°")
    print(f"  Std deviation:      {std_diff:.3f}°")
    
    # Count cameras with large differences
    large_diffs = [d for d in differences if d['angle_diff'] > args.threshold_deg]
    print(f"  Cameras with >{args.threshold_deg}° diff: {len(large_diffs)}/{len(differences)} ({100*len(large_diffs)/len(differences):.1f}%)")
    
    # Show detailed comparison for worst cases
    print(f"\n" + "-"*80)
    print(f"DETAILED ANALYSIS (Top {min(args.max_samples, len(differences))} worst cases)")
    print("-"*80)
    
    for i, diff in enumerate(differences[:args.max_samples]):
        print(f"\n{i+1}. Camera: {diff['name']}")
        print(f"   Angular difference: {diff['angle_diff']:.3f}°")
        
        print(f"   Target quaternion:    [{diff['target_q'][0]:8.5f}, {diff['target_q'][1]:8.5f}, {diff['target_q'][2]:8.5f}, {diff['target_q'][3]:8.5f}]")
        print(f"   Aligned quaternion:   [{diff['aligned_q'][0]:8.5f}, {diff['aligned_q'][1]:8.5f}, {diff['aligned_q'][2]:8.5f}, {diff['aligned_q'][3]:8.5f}]")
        
        print(f"   Target Euler (r,p,y): [{diff['target_euler'][0]:7.2f}°, {diff['target_euler'][1]:7.2f}°, {diff['target_euler'][2]:7.2f}°]")
        print(f"   Aligned Euler (r,p,y):[{diff['aligned_euler'][0]:7.2f}°, {diff['aligned_euler'][1]:7.2f}°, {diff['aligned_euler'][2]:7.2f}°]")
        
        # Compute Euler differences
        euler_diff = diff['aligned_euler'] - diff['target_euler']
        # Handle angle wrapping
        euler_diff = ((euler_diff + 180) % 360) - 180
        print(f"   Euler difference:     [{euler_diff[0]:7.2f}°, {euler_diff[1]:7.2f}°, {euler_diff[2]:7.2f}°]")
    
    # Analysis of patterns
    print(f"\n" + "-"*80)
    print("PATTERN ANALYSIS")
    print("-"*80)
    
    # Check if quaternions are just negated (equivalent rotations)
    negated_count = 0
    for diff in differences:
        target_q = diff['target_q']
        aligned_q = diff['aligned_q']
        
        # Check if aligned_q ≈ -target_q (same rotation, different sign)
        negated_q = -target_q
        angle_to_negated = quaternion_difference(aligned_q, negated_q)
        
        if angle_to_negated < 1.0:  # Very small difference to negated
            negated_count += 1
    
    print(f"Quaternions that are negated versions: {negated_count}/{len(differences)}")
    
    # Check for systematic rotation bias
    if len(differences) >= 3:
        all_target_eulers = np.array([d['target_euler'] for d in differences])
        all_aligned_eulers = np.array([d['aligned_euler'] for d in differences])
        
        mean_target_euler = np.mean(all_target_eulers, axis=0)
        mean_aligned_euler = np.mean(all_aligned_eulers, axis=0)
        
        print(f"\nMean Target Euler:    [{mean_target_euler[0]:7.2f}°, {mean_target_euler[1]:7.2f}°, {mean_target_euler[2]:7.2f}°]")
        print(f"Mean Aligned Euler:   [{mean_aligned_euler[0]:7.2f}°, {mean_aligned_euler[1]:7.2f}°, {mean_aligned_euler[2]:7.2f}°]")
        
        systematic_bias = mean_aligned_euler - mean_target_euler
        systematic_bias = ((systematic_bias + 180) % 360) - 180
        print(f"Systematic bias:      [{systematic_bias[0]:7.2f}°, {systematic_bias[1]:7.2f}°, {systematic_bias[2]:7.2f}°]")
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
