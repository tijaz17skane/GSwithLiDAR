
# SRTaligner.py

import numpy as np
import argparse
import os
import sys
from cam_world_conversions import cam2world, world2cam

def load_colmap_cameras(filepath, input_format='world'):
    """
    Load camera positions (TX, TY, TZ) and names from COLMAP images.txt.
    
    Parameters:
    -----------
    filepath : str
        Path to images.txt file
    input_format : str
        'world' for world coordinates (current logic) or 'cam' for camera coordinates
    
    Returns:
    --------
    positions : np.ndarray
        Camera positions as (N, 3) array [TX, TY, TZ]
    names : list
        Corresponding names for each position
    all_lines : list (only for 'cam' format)
        All lines from the file for reconstruction
    """
    positions = []
    names = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if input_format == 'world':
        for line in lines:
            line = line.strip()
            
            # Skip only comments
            if line.startswith('#'):
                continue
            
            # Parse camera pose line
            parts = line.split()
            if len(parts) >= 10:
                # Extract TX, TY, TZ (indices 5, 6, 7) and NAME (index 9)
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                name = parts[9]
                positions.append([tx, ty, tz])
                names.append(name)
        
        return np.array(positions), names
    
    elif input_format == 'cam':
        # For camera format, we need to parse odd rows and keep all lines for reconstruction
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip comments
            if line.startswith('#'):
                i += 1
                continue
            
            # Parse camera pose line (odd row)
            parts = line.split()
            if len(parts) >= 10:
                # Extract quaternion and translation
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                name = parts[9]
                
                # Convert from camera coordinates to world coordinates
                pos_world, _ = cam2world(qw, qx, qy, qz, tx, ty, tz)
                positions.append(pos_world)
                names.append(name)
            
            # Skip next line (POINTS2D data) or move to next if at end
            i += 2
        
        return np.array(positions), names, lines
    
    else:
        raise ValueError(f"Unknown input_format: {input_format}")


def load_point_cloud_with_names(filepath, file_format='auto', input_format='world'):
    """
    Load point cloud with names for correspondence matching.
    
    Returns:
    --------
    points : np.ndarray
        Point cloud as (N, 3) array
    names : list
        Corresponding names for each point
    all_lines : list (optional, for 'cam' format)
        All lines from the file for reconstruction
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    # Auto-detect COLMAP format
    if file_format == 'auto':
        with open(filepath, 'r') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        # Check if it looks like COLMAP format
        if any('IMAGE_ID' in line and 'QW' in line for line in first_lines):
            file_format = 'colmap'
            print("  Detected COLMAP images.txt format")
    
    if file_format == 'colmap':
        result = load_colmap_cameras(filepath, input_format)
        if input_format == 'cam':
            return result  # Returns (positions, names, all_lines)
        else:
            return result  # Returns (positions, names)
    
    elif ext in ['.txt', '.xyz']:
        # Try to load as space/tab separated with last column as name
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            points = []
            names = []
            
            # Check if this looks like COLMAP format by examining the structure
            is_colmap_format = False
            for line in lines[:10]:  # Check first 10 lines
                if line.strip().startswith('#') and 'IMAGE_ID' in line and 'QW' in line:
                    is_colmap_format = True
                    break
            
            if is_colmap_format:
                print("  Detected COLMAP images.txt format in simple txt file")
                # Parse like COLMAP format - extract from odd rows only
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Skip comments
                    if line.startswith('#'):
                        i += 1
                        continue
                    
                    # Parse camera pose line (odd row)
                    parts = line.split()
                    if len(parts) >= 10:
                        if input_format == 'cam':
                            # Extract quaternion and translation, convert to world
                            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                            name = parts[9]
                            
                            # Convert from camera coordinates to world coordinates
                            pos_world, _ = cam2world(qw, qx, qy, qz, tx, ty, tz)
                            points.append(pos_world)
                            names.append(name)
                        else:
                            # Extract world coordinates directly
                            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                            name = parts[9]
                            points.append([tx, ty, tz])
                            names.append(name)
                    
                    # Skip next line (POINTS2D data) or move to next if at end
                    i += 2
            else:
                # Parse as simple format
                for line in lines:
                    line = line.strip()
                    # Skip only comments
                    if line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 4:
                        # First 3 columns are x, y, z; last column is name
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        name = parts[-1]  # Last column is the name
                        points.append([x, y, z])
                        names.append(name)
                    elif len(parts) == 3:
                        # No names provided, use line number
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                        names.append(f"point_{len(points)}")
                
                # For simple txt format, if input_format is 'cam', we need to handle it
                # But simple txt format doesn't have quaternions, so we assume these are already world coordinates
                # and warn the user if they specified 'cam' format for simple txt
                if input_format == 'cam' and not is_colmap_format:
                    print(f"  Warning: Simple txt format doesn't support camera coordinate conversion.")
                    print(f"  Assuming coordinates in {filepath} are already world coordinates.")
            
            return np.array(points), names
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            sys.exit(1)
    
    elif ext == '.npy':
        # For .npy, we can't have names, so generate them
        try:
            points = np.load(filepath)
            if points.shape[1] != 3:
                print(f"Error: Expected (N,3) array, got {points.shape}")
                sys.exit(1)
            names = [f"point_{i}" for i in range(len(points))]
            return points, names
        except Exception as e:
            print(f"Error loading NPY file: {e}")
            sys.exit(1)
    
    else:
        print(f"Error: Unsupported file format '{ext}'")
        sys.exit(1)


def match_point_correspondences(points_A, names_A, points_B, names_B):
    """
    Match points from A and B based on their names.
    
    Returns:
    --------
    matched_A : np.ndarray
        Matched points from A
    matched_B : np.ndarray
        Matched points from B
    matched_names : list
        Names of matched points
    """
    # Create dictionaries for fast lookup
    dict_A = {name: idx for idx, name in enumerate(names_A)}
    dict_B = {name: idx for idx, name in enumerate(names_B)}
    
    # Find common names
    common_names = set(names_A) & set(names_B)
    
    if len(common_names) == 0:
        print("Error: No matching names found between the two point clouds!")
        sys.exit(1)
    
    # Sort for consistent ordering
    common_names = sorted(list(common_names))
    
    # Extract matched points
    matched_A = np.array([points_A[dict_A[name]] for name in common_names])
    matched_B = np.array([points_B[dict_B[name]] for name in common_names])
    
    return matched_A, matched_B, common_names


def save_point_cloud_with_names(filepath, points, names):
    """Save point cloud with names to file."""
    ext = os.path.splitext(filepath)[1].lower()
    
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if ext in ['.txt', '.xyz']:
        with open(filepath, 'w') as f:
            f.write("# TX TY TZ NAME\n")
            for point, name in zip(points, names):
                f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {name}\n")
        print(f"Saved to: {filepath}")
    
    elif ext == '.npy':
        np.save(filepath, points)
        # Save names separately
        names_file = filepath.replace('.npy', '_names.txt')
        with open(names_file, 'w') as f:
            for name in names:
                f.write(f"{name}\n")
        print(f"Saved to: {filepath}")
        print(f"Names saved to: {names_file}")
    
    else:
        print(f"Warning: Unsupported format '{ext}', saving as .txt")
        filepath = filepath + '.txt'
        with open(filepath, 'w') as f:
            f.write("# TX TY TZ NAME\n")
            for point, name in zip(points, names):
                f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {name}\n")


def save_colmap_format(filepath, aligned_positions, names, original_lines=None, input_format='world'):
    """Save aligned positions back to original format."""
    if input_format == 'world':
        # Simple format: just save world positions
        with open(filepath, 'w') as f:
            f.write("# TX TY TZ NAME\n")
            for point, name in zip(aligned_positions, names):
                f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {name}\n")
    
    elif input_format == 'cam':
        # Complex format: reconstruct original COLMAP format with aligned positions
        with open(filepath, 'w') as f:
            aligned_idx = 0
            i = 0
            
            while i < len(original_lines):
                line = original_lines[i].strip()
                
                # Copy comment lines as-is
                if line.startswith('#'):
                    f.write(original_lines[i])
                    i += 1
                    continue
                
                # Process camera pose line (odd row)
                parts = line.split()
                if len(parts) >= 10:
                    # Get original quaternion and other data
                    image_id = parts[0]
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    camera_id = parts[8]
                    name = parts[9]
                    
                    # Convert aligned world position back to camera coordinates
                    aligned_world_pos = aligned_positions[aligned_idx]
                    t_cam, _ = world2cam(qw, qx, qy, qz, aligned_world_pos[0], aligned_world_pos[1], aligned_world_pos[2])
                    
                    # Write updated camera pose line
                    f.write(f"{image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {t_cam[0]:.8f} {t_cam[1]:.8f} {t_cam[2]:.8f} {camera_id} {name}\n")
                    aligned_idx += 1
                    
                    # Copy POINTS2D line as-is (next line)
                    if i + 1 < len(original_lines):
                        f.write(original_lines[i + 1])
                    
                    i += 2
                else:
                    f.write(original_lines[i])
                    i += 1


def save_colmap_world_format(filepath, aligned_positions, names, num_images=None):
    """Save aligned positions in full COLMAP format with world coordinates."""
    # Calculate statistics
    if num_images is None:
        num_images = len(aligned_positions)
    
    with open(filepath, 'w') as f:
        # Write COLMAP header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}, mean observations per image: 0.0\n")
        
        # Write camera poses in world coordinates
        for i, (point, name) in enumerate(zip(aligned_positions, names)):
            image_id = i + 1
            # Identity quaternion for world coordinates (no rotation)
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            # World position
            tx, ty, tz = point[0], point[1], point[2]
            camera_id = 1  # Default camera ID
            
            # Write image line
            f.write(f"{image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} {camera_id} {name}\n")
            # Write empty POINTS2D line
            f.write("\n")


def save_colmap_world_format_as_cam(filepath, aligned_positions, names, num_images=None):
    """Save aligned positions in COLMAP format treating world coordinates as camera positions with identity quaternions."""
    # Calculate statistics
    if num_images is None:
        num_images = len(aligned_positions)
    
    with open(filepath, 'w') as f:
        # Write COLMAP header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}\n")
        
        # Write camera poses using world coordinates with identity quaternions
        for i, (point, name) in enumerate(zip(aligned_positions, names)):
            image_id = i + 1
            # Identity quaternion (no rotation from world to camera)
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            # Use world position as camera translation
            tx, ty, tz = point[0], point[1], point[2]
            camera_id = 1  # Default camera ID
            
            # Write image line
            f.write(f"{image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} {camera_id} {name}\n")
            # Write empty POINTS2D line
            f.write("\n")


def save_ply_point_cloud(filepath, points, names=None):
    """Save point cloud as PLY file."""
    with open(filepath, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices (with default white color)
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} 255 255 255\n")


def save_srt_matrices(output_dir, R, t, s):
    """Save Scale, Rotation, Translation as separate matrices and combined 4x4 matrix."""
    
    # Scale matrix (4x4)
    S = np.eye(4)
    S[:3, :3] *= s
    
    # Rotation matrix (4x4)
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = R
    
    # Translation matrix (4x4)
    T_4x4 = np.eye(4)
    T_4x4[:3, 3] = t
    
    # Combined transformation matrix: T * R * S
    combined = T_4x4 @ R_4x4 @ S
    
    # Save individual matrices
    np.savetxt(os.path.join(output_dir, 'scale_matrix.txt'), S, fmt='%.8f', header="Scale Matrix (4x4)")
    np.savetxt(os.path.join(output_dir, 'rotation_matrix.txt'), R_4x4, fmt='%.8f', header="Rotation Matrix (4x4)")
    np.savetxt(os.path.join(output_dir, 'translation_matrix.txt'), T_4x4, fmt='%.8f', header="Translation Matrix (4x4)")
    np.savetxt(os.path.join(output_dir, 'combined_transform.txt'), combined, fmt='%.8f', header="Combined Transformation Matrix: T*R*S")
    
    print(f"✓ SRT matrices saved:")
    print(f"  Scale:       {os.path.join(output_dir, 'scale_matrix.txt')}")
    print(f"  Rotation:    {os.path.join(output_dir, 'rotation_matrix.txt')}")
    print(f"  Translation: {os.path.join(output_dir, 'translation_matrix.txt')}")
    print(f"  Combined:    {os.path.join(output_dir, 'combined_transform.txt')}")


def umeyama_alignment(A, B):
    """
    Estimate similarity transformation using Umeyama's method.
    
    Returns: R, t, s
    """
    assert A.shape == B.shape, "Point sets must have the same shape"
    assert A.shape[1] == 3, "Points must be 3-dimensional"
    
    N = A.shape[0]
    
    # Compute means
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)
    
    # Center the data
    X = A - mu_A
    Y = B - mu_B
    
    # Cross-covariance matrix
    Sigma_xy = (Y.T @ X) / N
    
    # SVD
    U, D, Vt = np.linalg.svd(Sigma_xy)
    
    # Rotation with reflection correction
    S = np.eye(3)
    if np.linalg.det(Sigma_xy) < 0:
        S[2, 2] = -1
    
    R = U @ S @ Vt
    
    # Scale
    var_A = np.sum(np.linalg.norm(X, axis=1)**2) / N
    s = (1 / var_A) * np.trace(np.diag(D) @ S)
    
    # Translation
    t = mu_B - s * (R @ mu_A)
    
    return R, t, s


def transform_points(points, R, t, s):
    """Apply similarity transformation: output = s * R * points + t"""
    return s * (points @ R.T) + t


def compute_squared_error(A, B):
    """
    Compute total squared error between point sets.
    
    Returns:
    --------
    total_error : float
        Sum of squared distances
    rmse : float
        Root mean squared error
    mean_error : float
        Mean Euclidean distance
    """
    diff = A - B
    squared_distances = np.sum(diff**2, axis=1)
    total_error = np.sum(squared_distances)
    rmse = np.sqrt(np.mean(squared_distances))
    mean_error = np.mean(np.sqrt(squared_distances))
    
    return total_error, rmse, mean_error


def save_transformation_params(output_dir, R, t, s, initial_errors, final_errors, 
                               num_matched, num_A, num_B):
    """Save transformation parameters and errors to file."""
    param_file = os.path.join(output_dir, 'transformation_params.txt')
    
    with open(param_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SIMILARITY TRANSFORMATION PARAMETERS (Umeyama's Method)\n")
        f.write("="*70 + "\n\n")
        
        f.write("POINT CORRESPONDENCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Points in A:              {num_A}\n")
        f.write(f"Points in B:              {num_B}\n")
        f.write(f"Matched correspondences:  {num_matched}\n")
        f.write(f"Unmatched in A:           {num_A - num_matched}\n")
        f.write(f"Unmatched in B:           {num_B - num_matched}\n\n")
        
        f.write("ALIGNMENT ERRORS\n")
        f.write("-"*70 + "\n")
        f.write(f"Initial Total Squared Error:  {initial_errors[0]:.6f}\n")
        f.write(f"Initial RMSE:                 {initial_errors[1]:.6f}\n")
        f.write(f"Initial Mean Error:           {initial_errors[2]:.6f}\n\n")
        
        f.write(f"Final Total Squared Error:    {final_errors[0]:.6f}\n")
        f.write(f"Final RMSE:                   {final_errors[1]:.6f}\n")
        f.write(f"Final Mean Error:             {final_errors[2]:.6f}\n\n")
        
        f.write(f"Error Reduction:              {(1 - final_errors[0]/initial_errors[0])*100:.2f}%\n")
        f.write("\n")
        
        f.write("TRANSFORMATION PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"Scale factor (s):             {s:.8f}\n\n")
        
        f.write("Rotation matrix (R):\n")
        for row in R:
            f.write("  " + "  ".join([f"{val:12.8f}" for val in row]) + "\n")
        f.write("\n")
        
        f.write("Translation vector (t):\n")
        f.write("  " + "  ".join([f"{val:12.8f}" for val in t]) + "\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("Transformation formula: B = s * R * A + t\n")
        f.write("="*70 + "\n")
    
    print(f"Transformation parameters saved to: {param_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Align two COLMAP point clouds using Umeyama similarity transformation with name-based correspondence matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  This tool aligns two COLMAP camera pose files by finding corresponding cameras based on image names
  and computing a similarity transformation (Scale + Rotation + Translation) using Umeyama's method.
  All alignment computations are performed in world coordinates for numerical stability.

EXAMPLES:
  # Basic usage with COLMAP files (TX,TY,TZ treated as world coordinates)
  python SRTaligner.py --inputA cameras_source.txt --inputB cameras_reference.txt --output_dir ./results
  
  # When input coordinates are in camera coordinate system (converts to world internally)
  python SRTaligner.py --inputA cameras_A.txt --inputB cameras_B.txt --input_format cam --output_dir ./results
  
  # Save detailed transformation parameters and verbose statistics
  python SRTaligner.py --inputA cameras_A.txt --inputB cameras_B.txt --output_dir ./results --save_params -v

WORKFLOW:
  1. Load input files (point clouds in txt  formatted like colmap requires. use --input_format world if in COLMAP format but world coordinates, otherwise use cam)
  2. If --input_format=cam: Convert camera coordinates to world coordinates using quaternions
  3. Match corresponding cameras by image filename (e.g., "image001.jpg")
  4. Compute similarity transformation in world coordinates (Source A → Target B)
  5. Apply transformation to all cameras from Source A
  6. Save outputs in specified directory

OUTPUTS (always generated):
  aligned_in_cam.txt      - Aligned cameras in COLMAP format (preserves original structure)
  aligned_in_world.ply    - 3D point cloud visualization of aligned camera positions
  scale_matrix.txt        - 4x4 scale transformation matrix
  rotation_matrix.txt     - 4x4 rotation transformation matrix  
  translation_matrix.txt  - 4x4 translation transformation matrix
  combined_transform.txt  - 4x4 combined transformation matrix (T*R*S)

INPUT FORMATS:
  COLMAP images.txt format:
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    127 0.904382 0.066545 -0.414289 -0.077652 1.707377 0.347907 -1.341681 1 image001.jpg
    
  Simple TXT format:
    # TX TY TZ NAME
    1.707377 0.347907 -1.341681 image001.jpg

COORDINATE SYSTEMS:
  --input_format=world (default): TX,TY,TZ are world coordinates (camera centers)
  --input_format=cam: TX,TY,TZ are camera coordinates (require quaternion conversion)
        """
    )
    
    parser.add_argument('--inputA', type=str, required=True,
                       help='Source camera poses file (COLMAP images.txt format or simple TXT)')
    parser.add_argument('--inputB', type=str, required=True,
                       help='Target/reference camera poses file (COLMAP images.txt format or simple TXT)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for aligned results and transformation matrices')
    parser.add_argument('--save_params', action='store_true',
                       help='Save detailed transformation parameters and statistics to file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed statistics about the alignment process')
    parser.add_argument('--input_format', type=str, default='cam',
                       choices=['world', 'cam'],
                       help='Coordinate system of TX,TY,TZ values: "world"=camera centers, "cam"=camera coordinates')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COLMAP CAMERA POSE ALIGNMENT TOOL")
    print("(Umeyama Similarity Transformation with Name-Based Matching)")
    print("="*70)
    print("Outputs: aligned_in_cam.txt, aligned_in_world.ply, and SRT transformation matrices")
    print("="*70 + "\n")
    
    # Load camera pose files
    print(f"Loading source camera poses from: {args.inputA}")
    if args.input_format == 'cam':
        result_A = load_point_cloud_with_names(args.inputA, 'colmap', args.input_format)
        points_A_world, names_A, original_lines_A = result_A
        print(f"  → Loaded {len(points_A_world)} cameras (converted from camera to world coordinates)")
    else:
        points_A_world, names_A = load_point_cloud_with_names(args.inputA, 'colmap', args.input_format)
        original_lines_A = None
        print(f"  → Loaded {len(points_A_world)} cameras (using world coordinates)")
    
    print(f"\nLoading target camera poses from: {args.inputB}")
    points_B_world, names_B = load_point_cloud_with_names(args.inputB, 'txt', args.input_format)
    if args.input_format == 'cam':
        print(f"  → Loaded {len(points_B_world)} cameras (converted from camera to world coordinates)")
    else:
        print(f"  → Loaded {len(points_B_world)} cameras (using world coordinates)")
    
    print(f"\n✓ Alignment will be computed in WORLD COORDINATE SYSTEM")
    
    # Match cameras by image filename
    print("\n" + "-"*70)
    print("MATCHING CAMERAS BY IMAGE FILENAME")
    print("-"*70)
    A_matched, B_matched, matched_names = match_point_correspondences(
        points_A_world, names_A, points_B_world, names_B
    )
    
    print(f"Cameras in source (A):       {len(points_A_world)}")
    print(f"Cameras in target (B):       {len(points_B_world)}")
    print(f"Matched camera pairs:        {len(matched_names)}")
    print(f"Unmatched cameras in A:      {len(points_A_world) - len(matched_names)}")
    print(f"Unmatched cameras in B:      {len(points_B_world) - len(matched_names)}")
    
    if len(matched_names) < 3:
        print("\n⚠ Error: Need at least 3 matched camera pairs for alignment!")
        print("  Check that image filenames match between the two files.")
        sys.exit(1)
    
    # Compute initial alignment error
    print("\n" + "-"*70)
    print("COMPUTING INITIAL ALIGNMENT ERROR")
    print("-"*70)
    initial_total_error, initial_rmse, initial_mean = compute_squared_error(A_matched, B_matched)
    print(f"Total Squared Error:  {initial_total_error:.6f}")
    print(f"RMSE:                 {initial_rmse:.6f}")
    print(f"Mean Error:           {initial_mean:.6f}")
    
    # Compute similarity transformation using Umeyama's method
    print("\n" + "-"*70)
    print("COMPUTING SIMILARITY TRANSFORMATION (Source A → Target B)")
    print("-"*70)
    R, t, s = umeyama_alignment(A_matched, B_matched)
    
    print(f"\nScale factor (s): {s:.8f}")
    print(f"\nRotation matrix (R):")
    print(R)
    print(f"\nTranslation vector (t):")
    print(f"  [{t[0]:12.8f}, {t[1]:12.8f}, {t[2]:12.8f}]")
    
    # Apply transformation to ALL cameras from source A
    print("\n" + "-"*70)
    print("APPLYING TRANSFORMATION TO ALL SOURCE CAMERAS")
    print("-"*70)
    C_all_world = transform_points(points_A_world, R, t, s)
    print(f"✓ Transformed all {len(points_A_world)} cameras from source A")
    
    # Apply transformation to matched points for error computation
    C_matched = transform_points(A_matched, R, t, s)
    
    # Compute final alignment error on matched cameras
    print("\n" + "-"*70)
    print("COMPUTING FINAL ALIGNMENT ERROR")
    print("-"*70)
    final_total_error, final_rmse, final_mean = compute_squared_error(C_matched, B_matched)
    print(f"Total Squared Error:  {final_total_error:.6f}")
    print(f"RMSE:                 {final_rmse:.6f}")
    print(f"Mean Error:           {final_mean:.6f}")
    
    # Error reduction
    error_reduction = (1 - final_total_error / initial_total_error) * 100
    print(f"\n✓ Alignment Error Reduction: {error_reduction:.2f}%")
    
    # Save all results to output directory
    print("\n" + "-"*70)
    print("SAVING ALIGNMENT RESULTS")
    print("-"*70)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always save SRT matrices
    save_srt_matrices(args.output_dir, R, t, s)
    
    # Always save aligned_in_world.ply for 3D visualization
    aligned_ply_path = os.path.join(args.output_dir, 'aligned_in_world.ply')
    save_ply_point_cloud(aligned_ply_path, C_all_world, names_A)
    print(f"✓ 3D visualization (PLY): {aligned_ply_path}")
    
    # Always save aligned_in_cam.txt in COLMAP format
    # Reconstructs original COLMAP structure with aligned camera positions
    aligned_cam_path = os.path.join(args.output_dir, 'aligned_in_cam.txt')
    if original_lines_A is not None:
        # Use original file structure to maintain exact COLMAP format
        save_colmap_format(aligned_cam_path, C_all_world, names_A, original_lines_A, 'cam')
    else:
        # Create COLMAP format from world coordinates with identity quaternions
        save_colmap_world_format_as_cam(aligned_cam_path, C_all_world, names_A, len(points_A_world))
    print(f"✓ Aligned cameras (COLMAP): {aligned_cam_path}")
    
    if args.save_params:
        initial_errors = (initial_total_error, initial_rmse, initial_mean)
        final_errors = (final_total_error, final_rmse, final_mean)
        save_transformation_params(args.output_dir, R, t, s, 
                                   initial_errors, final_errors,
                                   len(matched_names), len(points_A_world), len(points_B_world))
    
    # Verbose statistics
    if args.verbose:
        print("\n" + "="*70)
        print("DETAILED ALIGNMENT STATISTICS")
        print("="*70)
        print("\nSource cameras (matched subset):")
        print(f"  Mean position:    [{A_matched.mean(axis=0)[0]:10.4f}, {A_matched.mean(axis=0)[1]:10.4f}, {A_matched.mean(axis=0)[2]:10.4f}]")
        print(f"  Position spread:  [{A_matched.std(axis=0)[0]:10.4f}, {A_matched.std(axis=0)[1]:10.4f}, {A_matched.std(axis=0)[2]:10.4f}]")
        
        print("\nTarget cameras (matched subset):")
        print(f"  Mean position:    [{B_matched.mean(axis=0)[0]:10.4f}, {B_matched.mean(axis=0)[1]:10.4f}, {B_matched.mean(axis=0)[2]:10.4f}]")
        print(f"  Position spread:  [{B_matched.std(axis=0)[0]:10.4f}, {B_matched.std(axis=0)[1]:10.4f}, {B_matched.std(axis=0)[2]:10.4f}]")
        
        print("\nAligned cameras (transformed):")
        print(f"  Mean position:    [{C_matched.mean(axis=0)[0]:10.4f}, {C_matched.mean(axis=0)[1]:10.4f}, {C_matched.mean(axis=0)[2]:10.4f}]")
        print(f"  Position spread:  [{C_matched.std(axis=0)[0]:10.4f}, {C_matched.std(axis=0)[1]:10.4f}, {C_matched.std(axis=0)[2]:10.4f}]")
        
        # Show sample of matched camera names
        print(f"\nSample of matched camera names:")
        for name in matched_names[:10]:
            print(f"  - {name}")
        if len(matched_names) > 10:
            print(f"  ... and {len(matched_names) - 10} more camera pairs")
    
    print("\n" + "="*70)
    print("✓ CAMERA ALIGNMENT COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()