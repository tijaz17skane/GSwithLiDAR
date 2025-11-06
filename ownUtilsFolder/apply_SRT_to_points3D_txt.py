#!/usr/bin/env python3
"""
Apply SRT transformation to COLMAP points3D.txt file.

This script applies a similarity transformation (Scale + Rotation + Translation) 
to 3D points in a COLMAP points3D.txt file using a combined transformation matrix.
"""

import argparse
import numpy as np
import os
import sys


def parse_transformation_matrix(matrix_file):
    """
    Parse the combined transformation matrix from file.
    
    Parameters:
    -----------
    matrix_file : str
        Path to the transformation matrix file
        
    Returns:
    --------
    R : np.ndarray
        3x3 rotation matrix
    t : np.ndarray  
        3x1 translation vector
    s : float
        Scale factor
    """
    try:
        # Load the 4x4 transformation matrix
        transform_4x4 = np.loadtxt(matrix_file)
        
        if transform_4x4.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {transform_4x4.shape}")
        
        # Extract rotation and scale from top-left 3x3 block
        RS_matrix = transform_4x4[:3, :3]
        
        # Extract translation from top-right column
        t = transform_4x4[:3, 3]
        
        # Decompose scale and rotation using SVD
        U, S, Vt = np.linalg.svd(RS_matrix)
        
        # Scale factor is the geometric mean of singular values
        s = np.cbrt(np.prod(S))  # Cube root for 3D
        
        # Rotation matrix
        R = U @ Vt
        
        # Ensure proper rotation matrix (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        
        print(f"✓ Parsed transformation matrix:")
        print(f"  Scale factor: {s:.8f}")
        print(f"  Translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
        print(f"  Rotation matrix determinant: {np.linalg.det(R):.8f}")
        
        return R, t, s
        
    except Exception as e:
        print(f"Error parsing transformation matrix: {e}")
        sys.exit(1)


def transform_points(points, R, t, s):
    """Apply similarity transformation: output = s * R * points + t"""
    return s * (points @ R.T) + t


def parse_points3d_txt(input_file):
    """
    Parse COLMAP points3D.txt file.
    
    Returns:
    --------
    points : np.ndarray
        3D points as (N, 3) array
    colors : np.ndarray
        RGB colors as (N, 3) array
    headers : list
        Header lines to preserve
    point_data : list
        Complete point data including IDs, errors, tracks
    """
    points = []
    colors = []
    headers = []
    point_data = []
    
    print(f"Reading points3D.txt from: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Store header lines
                if line.startswith('#'):
                    headers.append(line)
                    continue
                
                # Skip empty lines
                if not line:
                    continue
                
                # Parse point data
                parts = line.split()
                if len(parts) < 8:
                    print(f"Warning: Line {line_num} has insufficient data: {len(parts)} parts")
                    continue
                
                try:
                    point_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                    error = float(parts[7])
                    
                    # Store 3D coordinates and colors
                    points.append([x, y, z])
                    colors.append([r, g, b])
                    
                    # Store complete line data for reconstruction
                    track_data = ' '.join(parts[8:]) if len(parts) > 8 else ""
                    point_data.append({
                        'id': point_id,
                        'error': error,
                        'track': track_data
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
        
        points = np.array(points)
        colors = np.array(colors, dtype=int)
        
        print(f"✓ Loaded {len(points)} 3D points")
        
        if len(points) == 0:
            print("Error: No valid points found in input file!")
            sys.exit(1)
        
        return points, colors, headers, point_data
        
    except Exception as e:
        print(f"Error reading points3D.txt: {e}")
        sys.exit(1)


def save_points3d_txt(output_file, transformed_points, colors, headers, point_data):
    """
    Save transformed points to COLMAP points3D.txt format.
    
    Parameters:
    -----------
    output_file : str
        Output file path
    transformed_points : np.ndarray
        Transformed 3D points
    colors : np.ndarray
        RGB colors
    headers : list
        Header lines
    point_data : list
        Original point metadata
    """
    print(f"Saving transformed points to: {output_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with open(output_file, 'w') as f:
            # Write headers
            for header in headers:
                f.write(header + '\n')
            
            # Update point count in header if present
            if not any('Number of points:' in h for h in headers):
                f.write(f"# Number of points: {len(transformed_points)}\n")
            
            # Write transformed points
            for i, (point, color, data) in enumerate(zip(transformed_points, colors, point_data)):
                x, y, z = point[0], point[1], point[2]
                r, g, b = color[0], color[1], color[2]
                
                # Reconstruct line with transformed coordinates
                line_parts = [
                    str(data['id']),
                    f"{x:.8f}",
                    f"{y:.8f}", 
                    f"{z:.8f}",
                    str(r),
                    str(g),
                    str(b),
                    f"{data['error']:.8f}"
                ]
                
                # Add track data if present
                if data['track']:
                    line_parts.append(data['track'])
                
                f.write(' '.join(line_parts) + '\n')
        
        print(f"✓ Saved {len(transformed_points)} transformed points")
        
    except Exception as e:
        print(f"Error saving points3D.txt: {e}")
        sys.exit(1)


def print_transformation_summary(original_points, transformed_points, R, t, s):
    """Print summary of transformation results."""
    print("\n" + "="*70)
    print("TRANSFORMATION SUMMARY")
    print("="*70)
    
    # Calculate bounding boxes
    orig_min = np.min(original_points, axis=0)
    orig_max = np.max(original_points, axis=0)
    orig_center = np.mean(original_points, axis=0)
    
    trans_min = np.min(transformed_points, axis=0)
    trans_max = np.max(transformed_points, axis=0)
    trans_center = np.mean(transformed_points, axis=0)
    
    print(f"Number of points:        {len(original_points)}")
    print(f"Scale factor:            {s:.8f}")
    print()
    
    print("ORIGINAL POINT CLOUD:")
    print(f"  Bounding box min:      [{orig_min[0]:12.6f}, {orig_min[1]:12.6f}, {orig_min[2]:12.6f}]")
    print(f"  Bounding box max:      [{orig_max[0]:12.6f}, {orig_max[1]:12.6f}, {orig_max[2]:12.6f}]")
    print(f"  Center:                [{orig_center[0]:12.6f}, {orig_center[1]:12.6f}, {orig_center[2]:12.6f}]")
    print()
    
    print("TRANSFORMED POINT CLOUD:")
    print(f"  Bounding box min:      [{trans_min[0]:12.6f}, {trans_min[1]:12.6f}, {trans_min[2]:12.6f}]")
    print(f"  Bounding box max:      [{trans_max[0]:12.6f}, {trans_max[1]:12.6f}, {trans_max[2]:12.6f}]")
    print(f"  Center:                [{trans_center[0]:12.6f}, {trans_center[1]:12.6f}, {trans_center[2]:12.6f}]")
    print()
    
    # Calculate displacement
    center_displacement = trans_center - orig_center
    print(f"Center displacement:     [{center_displacement[0]:12.6f}, {center_displacement[1]:12.6f}, {center_displacement[2]:12.6f}]")
    print(f"Center displacement magnitude: {np.linalg.norm(center_displacement):.6f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Apply SRT transformation to COLMAP points3D.txt file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  This tool applies a similarity transformation (Scale + Rotation + Translation) to 3D points
  in a COLMAP points3D.txt file using a combined 4x4 transformation matrix.
  
  The transformation follows the formula: output = s * R * input + t
  
EXAMPLES:
  # Apply transformation to points3D.txt
  python apply_SRT_to_points3D_txt.py --input_txt points3D.txt --output_txt transformed_points3D.txt --SRT_transf_matrix combined_transform.txt
  
  # With verbose output
  python apply_SRT_to_points3D_txt.py --input_txt points3D.txt --output_txt transformed_points3D.txt --SRT_transf_matrix combined_transform.txt --verbose

INPUT FORMATS:
  Transformation matrix (4x4):
    # Combined Transformation Matrix: T*R*S
    9.83989219 -1.08272846 9.14403370 413299.13466555
    -9.20771308 -1.06893286 9.78184716 5318012.46106755
    -0.06060489 -13.39007891 -1.52027808 303.79622280
    0.00000000 0.00000000 0.00000000 1.00000000
  
  Points3D.txt format:
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    4427 0.16861396 -0.27044964 3.39335689 253 254 241 0.59431852 23 4807 128 7814
        """
    )
    
    parser.add_argument('--input_txt', type=str, required=True,
                       help='Input COLMAP points3D.txt file')
    parser.add_argument('--output_txt', type=str, required=True,
                       help='Output path for transformed points3D.txt file')
    parser.add_argument('--SRT_transf_matrix', type=str, required=True,
                       help='Path to 4x4 combined transformation matrix (T*R*S) file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed transformation summary')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COLMAP POINTS3D.TXT TRANSFORMATION TOOL")
    print("="*70)
    print("Applies SRT similarity transformation to 3D point coordinates")
    print("="*70 + "\n")
    
    # Validate input files
    if not os.path.exists(args.input_txt):
        print(f"Error: Input file not found: {args.input_txt}")
        sys.exit(1)
    
    if not os.path.exists(args.SRT_transf_matrix):
        print(f"Error: Transformation matrix file not found: {args.SRT_transf_matrix}")
        sys.exit(1)
    
    # Parse transformation matrix
    print("LOADING TRANSFORMATION MATRIX")
    print("-" * 70)
    R, t, s = parse_transformation_matrix(args.SRT_transf_matrix)
    
    # Load points3D.txt
    print("\nLOADING 3D POINTS")
    print("-" * 70)
    points, colors, headers, point_data = parse_points3d_txt(args.input_txt)
    
    # Apply transformation
    print("\nAPPLYING TRANSFORMATION")
    print("-" * 70)
    print("Transformation formula: output = s * R * input + t")
    transformed_points = transform_points(points, R, t, s)
    print(f"✓ Transformed {len(points)} points")
    
    # Save results
    print("\nSAVING RESULTS")
    print("-" * 70)
    save_points3d_txt(args.output_txt, transformed_points, colors, headers, point_data)
    
    # Print summary
    if args.verbose:
        print_transformation_summary(points, transformed_points, R, t, s)
    
    print(f"\n✓ TRANSFORMATION COMPLETED SUCCESSFULLY!")
    print(f"✓ Output saved to: {args.output_txt}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
