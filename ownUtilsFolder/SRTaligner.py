# SRTaligner.py

import numpy as np
import argparse
import os
import sys
from cam_world_conversions import cam2world, cam2world_batch, world2cam, world2cam_batch
from scipy.spatial.transform import Rotation as R

def load_point_cloud_in_colmap_format(filepath):
    """
    Load camera translations, quaternions and names from a COLMAP images.txt file.
    
    Expected format:
      # Image list with two lines of data per image:
      #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
      #   POINTS2D[] as (X, Y, POINT3D_ID)
      # Number of images: N
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D line (optional, skipped)
      ...

    The function skips the first 4 header lines (comments starting with #),
    then reads odd-numbered lines (after header) containing camera poses in format:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    
    Even-numbered lines contain POINTS2D data and are skipped.

    Returns:
      translations : np.ndarray (N,3) - [TX, TY, TZ]
      quaternions  : np.ndarray (N,4) - [QW, QX, QY, QZ]
      names        : list of str - image filenames
      lines        : all lines read from file (list of str)
    """
    translations = []
    quaternions = []
    names = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip first 4 rows (header comments)
    line_idx = 4
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            line_idx += 1
            continue

        parts = line.split()
        
        # COLMAP format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        # Need at least 10 tokens (IMAGE_ID + 4 quat + 3 trans + CAMERA_ID + NAME)
        if len(parts) < 10:
            line_idx += 1
            continue

        try:
            # Parse COLMAP images.txt format
            # parts[0] = IMAGE_ID (skip)
            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])
            # parts[8] = CAMERA_ID (skip)
            name = parts[9]  # Image filename
            
            translations.append([tx, ty, tz])
            quaternions.append([qw, qx, qy, qz])
            names.append(name)
            
        except (ValueError, IndexError):
            # Skip lines that don't match expected format
            pass
        
        # Skip next line (POINTS2D line) and move to next camera pose
        line_idx += 2

    return np.array(translations, dtype=float), np.array(quaternions, dtype=float), names, lines

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

def save_colmap_format(filepath, aligned_quaternions, aligned_positions, names, even_lines=None):
    """Takes aligned poses in camera frame and quaternions in wxyz format and saves them in COLMAP format.
        Quaternions are already in wxyz (qw qx qy qz) format, so no reordering needed.
        
        If even_lines is provided, outputs:
        - First 4 comment lines (standard COLMAP header)
        - Odd-numbered lines (camera poses) with aligned data
        - Even-numbered lines from even_lines after each odd line
    """
    with open(filepath, 'w') as f:
        if even_lines is None:
            # Standard header (no mean observations available)
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(aligned_positions)}\n")
            # Simple output: write aligned poses (odd lines) with empty even lines
            for idx in range(len(aligned_positions)):
                trans = aligned_positions[idx]
                quat = aligned_quaternions[idx]
                name = names[idx]
                f.write(f"{idx+1} {quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f} {trans[0]:.8f} {trans[1]:.8f} {trans[2]:.8f} {idx+1} {name}\n")
                f.write("\n")
        else:
            # Parse even_lines to extract original header and observation lines
            header_lines = []
            odd_lines = []
            even_line_data = []
            line_idx = 0
            for line in even_lines:
                stripped = line.strip()
                if line_idx < 4:
                    header_lines.append(line)
                    line_idx += 1
                    continue
                if stripped and not stripped.startswith('#'):
                    if (line_idx - 4) % 2 == 0:
                        odd_lines.append(line)
                    else:
                        even_line_data.append(line)
                line_idx += 1

            # Build unified 4-line header (avoid duplication). Use mean observations if present in original 4th line.
            # First three lines: use canonical form (ignore any deviations).
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            mean_suffix = ""
            if len(header_lines) >= 4:
                fourth = header_lines[3].strip()
                # Detect mean observations segment
                if "," in fourth:
                    # Preserve everything after the first comma
                    mean_suffix = "," + fourth.split(",", 1)[1].strip()

            f.write(f"# Number of images: {len(aligned_positions)}{(' ' + mean_suffix) if mean_suffix else ''}\n")

            # Write aligned poses with original IMAGE_ID and CAMERA_ID when available.
            for idx in range(len(aligned_positions)):
                trans = aligned_positions[idx]
                quat = aligned_quaternions[idx]
                name = names[idx]
                if idx < len(odd_lines):
                    parts = odd_lines[idx].split()
                    if len(parts) >= 10:
                        image_id = parts[0]
                        camera_id = parts[8]
                    else:
                        image_id = str(idx + 1)
                        camera_id = str(idx + 1)
                else:
                    image_id = str(idx + 1)
                    camera_id = str(idx + 1)
                f.write(f"{image_id} {quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f} {trans[0]:.8f} {trans[1]:.8f} {trans[2]:.8f} {camera_id} {name}\n")
                if idx < len(even_line_data):
                    even_line = even_line_data[idx]
                    if not even_line.endswith('\n'):
                        even_line += '\n'
                    f.write(even_line)
                else:
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
    
    # Reflection handling (prevent unintended 180° flips)
    # Previous version used det(Sigma_xy); now use det(U @ Vt) and adjust U directly.
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1  # Flip last singular vector to ensure proper rotation (det=+1)
    
    # Rotation
    R = U @ Vt
    
    # Scale (trace of singular values after reflection correction)
    var_A = np.sum(np.linalg.norm(X, axis=1)**2) / N
    s = (1 / var_A) * np.sum(D)
    
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

def compare_aligned_with_reference(aligned_filepath, reference_filepath, output_filepath):
    """
    Compare aligned cameras with reference cameras line by line.
    Computes errors in both camera coordinates and world coordinates.
    
    Parameters:
    -----------
    aligned_filepath : str
        Path to aligned_in_cam.txt
    reference_filepath : str
        Path to reference file (inputB)
    output_filepath : str
        Path to save comparison results
    """
    print("\n" + "-"*70)
    print("COMPARING ALIGNED CAMERAS WITH REFERENCE")
    print("-"*70)
    
    # Load both files
    aligned_cams = {}
    with open(aligned_filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 10:
                name = parts[9]
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                aligned_cams[name] = {
                    'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                    'tx': tx, 'ty': ty, 'tz': tz
                }
            i += 2
    
    ref_cams = {}
    with open(reference_filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 10:
                name = parts[9]
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                ref_cams[name] = {
                    'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                    'tx': tx, 'ty': ty, 'tz': tz
                }
            i += 2
    
    # Find common cameras
    common_names = sorted(set(aligned_cams.keys()) & set(ref_cams.keys()))
    
    if len(common_names) == 0:
        print("⚠ Warning: No matching cameras found between aligned and reference!")
        return
    
    print(f"Found {len(common_names)} matching cameras for comparison")
    
    # Compute errors for each camera
    cam_errors = []
    world_errors = []
    
    with open(output_filepath, 'w') as f:
        f.write("="*100 + "\n")
        f.write("DETAILED COMPARISON: ALIGNED vs REFERENCE CAMERAS\n")
        f.write("="*100 + "\n\n")
        f.write(f"Total cameras compared: {len(common_names)}\n")
        f.write(f"Aligned file:   {aligned_filepath}\n")
        f.write(f"Reference file: {reference_filepath}\n\n")
        
        f.write("="*100 + "\n")
        f.write("LINE-BY-LINE COMPARISON\n")
        f.write("="*100 + "\n\n")
        
        for name in common_names:
            aligned = aligned_cams[name]
            ref = ref_cams[name]
            
            # Camera coordinate errors
            cam_t_error = np.array([
                aligned['tx'] - ref['tx'],
                aligned['ty'] - ref['ty'],
                aligned['tz'] - ref['tz']
            ])
            cam_t_error_mag = np.linalg.norm(cam_t_error)
            
            cam_q_error = np.array([
                aligned['qw'] - ref['qw'],
                aligned['qx'] - ref['qx'],
                aligned['qy'] - ref['qy'],
                aligned['qz'] - ref['qz']
            ])
            cam_q_error_mag = np.linalg.norm(cam_q_error)

            q_aligned = np.array([aligned['qw'], aligned['qx'], aligned['qy'], aligned['qz']])
            t_aligned = np.array([aligned['tx'], aligned['ty'], aligned['tz']])
            # Convert to world coordinates
            aligned_t_world, aligned_R_world, aligned_q_world = cam2world_batch(q_aligned[np.newaxis, :], t_aligned[np.newaxis, :])
            aligned_t_world = aligned_t_world[0]
            aligned_R_world = aligned_R_world[0]
            aligned_q_world = aligned_q_world[0]
            
            q_ref = np.array([ref['qw'], ref['qx'], ref['qy'], ref['qz']])
            t_ref = np.array([ref['tx'], ref['ty'], ref['tz']])
            ref_t_world, ref_R_world, ref_q_world = cam2world_batch(q_ref[np.newaxis, :], t_ref[np.newaxis, :])
            ref_t_world = ref_t_world[0]
            ref_R_world = ref_R_world[0]
            ref_q_world = ref_q_world[0]
            
            # World coordinate errors
            world_t_error = aligned_t_world - ref_t_world
            world_t_error_mag = np.linalg.norm(world_t_error)
            
            world_q_error = aligned_q_world - ref_q_world
            world_q_error_mag = np.linalg.norm(world_q_error)
            
            # Compute angular error between quaternions (world coordinates)
            # Use dot product of quaternions (handle both q and -q representing same rotation)
            dot_product = np.abs(np.dot(aligned_q_world, ref_q_world))
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angular_error_rad = 2 * np.arccos(dot_product)
            angular_error_deg = np.degrees(angular_error_rad)
            
            # Store for summary statistics
            cam_errors.append({
                'name': name,
                't_error': cam_t_error_mag,
                'q_error': cam_q_error_mag
            })
            
            world_errors.append({
                'name': name,
                't_error': world_t_error_mag,
                'q_error': world_q_error_mag,
                'angular_error_deg': angular_error_deg
            })
            
            # Write detailed comparison for this camera
            f.write("-"*100 + "\n")
            f.write(f"Camera: {name}\n")
            f.write("-"*100 + "\n\n")
            
            f.write("CAMERA COORDINATES:\n")
            f.write(f"  Translation (tx, ty, tz):\n")
            f.write(f"    Aligned:    [{aligned['tx']:12.8f}, {aligned['ty']:12.8f}, {aligned['tz']:12.8f}]\n")
            f.write(f"    Reference:  [{ref['tx']:12.8f}, {ref['ty']:12.8f}, {ref['tz']:12.8f}]\n")
            f.write(f"    Error:      [{cam_t_error[0]:12.8f}, {cam_t_error[1]:12.8f}, {cam_t_error[2]:12.8f}]\n")
            f.write(f"    Error Mag:  {cam_t_error_mag:12.8f}\n\n")
            
            f.write(f"  Quaternion (qw, qx, qy, qz):\n")
            f.write(f"    Aligned:    [{aligned['qw']:12.8f}, {aligned['qx']:12.8f}, {aligned['qy']:12.8f}, {aligned['qz']:12.8f}]\n")
            f.write(f"    Reference:  [{ref['qw']:12.8f}, {ref['qx']:12.8f}, {ref['qy']:12.8f}, {ref['qz']:12.8f}]\n")
            f.write(f"    Error:      [{cam_q_error[0]:12.8f}, {cam_q_error[1]:12.8f}, {cam_q_error[2]:12.8f}, {cam_q_error[3]:12.8f}]\n")
            f.write(f"    Error Mag:  {cam_q_error_mag:12.8f}\n\n")
            
            f.write("WORLD COORDINATES:\n")
            f.write(f"  Position (world):\n")
            f.write(f"    Aligned:    [{aligned_t_world[0]:12.8f}, {aligned_t_world[1]:12.8f}, {aligned_t_world[2]:12.8f}]\n")
            f.write(f"    Reference:  [{ref_t_world[0]:12.8f}, {ref_t_world[1]:12.8f}, {ref_t_world[2]:12.8f}]\n")
            f.write(f"    Error:      [{world_t_error[0]:12.8f}, {world_t_error[1]:12.8f}, {world_t_error[2]:12.8f}]\n")
            f.write(f"    Error Mag:  {world_t_error_mag:12.8f}\n\n")
            
            f.write(f"  Quaternion (world):\n")
            f.write(f"    Aligned:    [{aligned_q_world[0]:12.8f}, {aligned_q_world[1]:12.8f}, {aligned_q_world[2]:12.8f}, {aligned_q_world[3]:12.8f}]\n")
            f.write(f"    Reference:  [{ref_q_world[0]:12.8f}, {ref_q_world[1]:12.8f}, {ref_q_world[2]:12.8f}, {ref_q_world[3]:12.8f}]\n")
            f.write(f"    Error:      [{world_q_error[0]:12.8f}, {world_q_error[1]:12.8f}, {world_q_error[2]:12.8f}, {world_q_error[3]:12.8f}]\n")
            f.write(f"    Error Mag:  {world_q_error_mag:12.8f}\n")
            f.write(f"    Angular Error: {angular_error_deg:12.6f} degrees\n\n")
        
        # Write summary statistics
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        # Camera coordinate statistics
        cam_t_errors = np.array([e['t_error'] for e in cam_errors])
        cam_q_errors = np.array([e['q_error'] for e in cam_errors])
        
        f.write("CAMERA COORDINATES:\n")
        f.write(f"  Translation Errors:\n")
        f.write(f"    Mean:    {np.mean(cam_t_errors):12.8f}\n")
        f.write(f"    Median:  {np.median(cam_t_errors):12.8f}\n")
        f.write(f"    Std Dev: {np.std(cam_t_errors):12.8f}\n")
        f.write(f"    Min:     {np.min(cam_t_errors):12.8f}\n")
        f.write(f"    Max:     {np.max(cam_t_errors):12.8f}\n")
        f.write(f"    RMSE:    {np.sqrt(np.mean(cam_t_errors**2)):12.8f}\n\n")
        
        f.write(f"  Quaternion Errors:\n")
        f.write(f"    Mean:    {np.mean(cam_q_errors):12.8f}\n")
        f.write(f"    Median:  {np.median(cam_q_errors):12.8f}\n")
        f.write(f"    Std Dev: {np.std(cam_q_errors):12.8f}\n")
        f.write(f"    Min:     {np.min(cam_q_errors):12.8f}\n")
        f.write(f"    Max:     {np.max(cam_q_errors):12.8f}\n")
        f.write(f"    RMSE:    {np.sqrt(np.mean(cam_q_errors**2)):12.8f}\n\n")
        
        # World coordinate statistics
        world_t_errors = np.array([e['t_error'] for e in world_errors])
        world_q_errors = np.array([e['q_error'] for e in world_errors])
        angular_errors = np.array([e['angular_error_deg'] for e in world_errors])
        
        f.write("WORLD COORDINATES:\n")
        f.write(f"  Position Errors:\n")
        f.write(f"    Mean:    {np.mean(world_t_errors):12.8f}\n")
        f.write(f"    Median:  {np.median(world_t_errors):12.8f}\n")
        f.write(f"    Std Dev: {np.std(world_t_errors):12.8f}\n")
        f.write(f"    Min:     {np.min(world_t_errors):12.8f}\n")
        f.write(f"    Max:     {np.max(world_t_errors):12.8f}\n")
        f.write(f"    RMSE:    {np.sqrt(np.mean(world_t_errors**2)):12.8f}\n\n")
        
        f.write(f"  Quaternion Errors:\n")
        f.write(f"    Mean:    {np.mean(world_q_errors):12.8f}\n")
        f.write(f"    Median:  {np.median(world_q_errors):12.8f}\n")
        f.write(f"    Std Dev: {np.std(world_q_errors):12.8f}\n")
        f.write(f"    Min:     {np.min(world_q_errors):12.8f}\n")
        f.write(f"    Max:     {np.max(world_q_errors):12.8f}\n")
        f.write(f"    RMSE:    {np.sqrt(np.mean(world_q_errors**2)):12.8f}\n\n")
        
        f.write(f"  Angular Errors (degrees):\n")
        f.write(f"    Mean:    {np.mean(angular_errors):12.6f}\n")
        f.write(f"    Median:  {np.median(angular_errors):12.6f}\n")
        f.write(f"    Std Dev: {np.std(angular_errors):12.6f}\n")
        f.write(f"    Min:     {np.min(angular_errors):12.6f}\n")
        f.write(f"    Max:     {np.max(angular_errors):12.6f}\n")
        f.write(f"    RMSE:    {np.sqrt(np.mean(angular_errors**2)):12.6f}\n\n")
        
        # Find worst cases
        f.write("WORST CASES:\n")
        worst_t_idx = np.argmax(world_t_errors)
        worst_angular_idx = np.argmax(angular_errors)
        
        f.write(f"  Largest Position Error:\n")
        f.write(f"    Camera:  {world_errors[worst_t_idx]['name']}\n")
        f.write(f"    Error:   {world_t_errors[worst_t_idx]:12.8f}\n\n")
        
        f.write(f"  Largest Angular Error:\n")
        f.write(f"    Camera:  {world_errors[worst_angular_idx]['name']}\n")
        f.write(f"    Error:   {angular_errors[worst_angular_idx]:12.6f} degrees\n\n")
        
        f.write("="*100 + "\n")
    
    print(f"✓ Comparison saved to: {output_filepath}")
    print(f"\n  Summary (World Coordinates):")
    print(f"    Position Error (RMSE):  {np.sqrt(np.mean(world_t_errors**2)):12.8f}")
    print(f"    Angular Error (RMSE):   {np.sqrt(np.mean(angular_errors**2)):12.6f} degrees")
    print(f"    Max Position Error:     {np.max(world_t_errors):12.8f}")
    print(f"    Max Angular Error:      {np.max(angular_errors):12.6f} degrees")


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
    # Basic usage with COLMAP images.txt files
    python SRTaligner.py --inputA cameras_source.txt --inputB cameras_reference.txt --output_dir ./results
    
    # Save detailed transformation parameters and verbose statistics
    python SRTaligner.py --inputA cameras_A.txt --inputB cameras_B.txt --output_dir ./results --save_params -v

    WORKFLOW:
    1. Load COLMAP images.txt files (camera coordinates are converted to world coordinates internally)
    2. Match corresponding cameras by image filename (e.g., "image001.jpg")
    3. Compute similarity transformation in world coordinates (Source A → Target B)
    4. Apply transformation to all cameras from Source A
    5. Save outputs in specified directory

    OUTPUTS (always generated):
    aligned_in_cam.txt      - Aligned cameras in COLMAP format (preserves original structure) - Actual Output
    aligned_in_world.ply    - 3D point cloud visualization of aligned camera positions
    scale_matrix.txt        - 4x4 scale transformation matrix
    rotation_matrix.txt     - 4x4 rotation transformation matrix  
    translation_matrix.txt  - 4x4 translation transformation matrix
    combined_transform.txt  - 4x4 combined transformation matrix (T*R*S)

    INPUT FORMAT:
    COLMAP images.txt format (required):
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        127 0.904382 0.066545 -0.414289 -0.077652 1.707377 0.347907 -1.341681 1 image001.jpg
        
    NOTE: TX,TY,TZ are camera coordinates and will be converted to world coordinates internally using quaternions
        """
    )
    
# =======================================================================
# ======= Argument Parsing =======
# =======================================================================

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
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COLMAP CAMERA POSE ALIGNMENT TOOL")
    print("(Umeyama Similarity Transformation with Name-Based Matching)")
    print("="*70)
    print("Outputs: aligned_in_cam.txt, aligned_in_world.ply, and SRT transformation matrices")
    print("="*70 + "\n")
    

# =======================================================================
# ======= Load Images =======
# =======================================================================

    print(f"Loading source camera poses from: {args.inputA} in camera frame")
    trans_A_camera, quats_A_camera, names_A, even_lines_from_A = load_point_cloud_in_colmap_format(args.inputA)
    print(f"  → Loaded {len(trans_A_camera)} Image Extrinsics in camera frame")
    
    print(f"\nLoading target camera poses from: {args.inputB} in camera frame")
    trans_B_camera, quats_B_camera, names_B, even_lines_from_B = load_point_cloud_in_colmap_format(args.inputB)
    print(f"  → Loaded {len(trans_B_camera)} Image Extrinsics in camera frame")
    
    
    # =======================================================================
    # ======= Compute SRT =======
    # =======================================================================
    
    print(f"\n✓ Alignment will be computed in WORLD COORDINATE SYSTEM")

    trans_A_world, _,quats_A_world = cam2world_batch(quats_A_camera, trans_A_camera)
    trans_B_world, _,quats_B_world = cam2world_batch(quats_B_camera, trans_B_camera)

    
    # Match cameras by image filename
    print("\n" + "-"*70)
    print("MATCHING CAMERAS BY IMAGE FILENAME")
    print("-"*70)
    A_matched_world, B_matched_world, matched_names = match_point_correspondences(trans_A_world, names_A, trans_B_world, names_B)
    
    print(f"Cameras in source (A):       {len(trans_A_world)}")
    print(f"Cameras in target (B):       {len(trans_B_world)}")
    print(f"Matched camera pairs:        {len(matched_names)}")
    print(f"Unmatched cameras in A:      {len(trans_A_world) - len(matched_names)}")
    print(f"Unmatched cameras in B:      {len(trans_B_world) - len(matched_names)}")
    
    if len(matched_names) < 3:
        print("\n⚠ Error: Need at least 3 matched camera pairs for alignment!")
        print("  Check that image filenames match between the two files.")
        sys.exit(1)


    # Compute initial alignment error
    print("\n" + "-"*70)
    print("COMPUTING INITIAL ALIGNMENT ERROR")
    print("-"*70)
    initial_total_error, initial_rmse, initial_mean = compute_squared_error(A_matched_world, B_matched_world)
    print(f"Total Squared Error:  {initial_total_error:.6f}")
    print(f"RMSE:                 {initial_rmse:.6f}")
    print(f"Mean Error:           {initial_mean:.6f}")
    
    # Compute similarity transformation using Umeyama's method
    print("\n" + "-"*70)
    print("COMPUTING SIMILARITY TRANSFORMATION (Source A → Target B)")
    print("-"*70)
    Rot_align, trans_align, scale_align = umeyama_alignment(A_matched_world, B_matched_world)
    
    print(f"\nScale factor (s): {scale_align:.8f}")
    print(f"\nRotation matrix (R):")
    print(Rot_align)
    print(f"\nTranslation vector (t):")
    print(f"  [{trans_align[0]:12.8f}, {trans_align[1]:12.8f}, {trans_align[2]:12.8f}]")
    
# =======================================================================
# ======= Apply SRT on Source =======
# =======================================================================


    # Apply transformation to ALL cameras from source A
    print("\n" + "-"*70)
    print("APPLYING TRANSFORMATION TO ALL SOURCE CAMERAS")
    print("-"*70)
    trans_C_all_world = transform_points(trans_A_world, Rot_align, trans_align, scale_align) #trans_C_all_world = scale_align * (points_A_world @ Rot_align.T) + trans_align
    print(f"✓ Transformed all {len(trans_A_world)} camera positions from source A")

    
    # Transform quaternions
    #quats_A_transformed = np.array([transform_quaternion(q, R) for q in quats_A]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR HERE> quats_A are in camera coords !!!!!!!!!!!!!!!!!

    q_Rot_align_world = R.from_matrix(Rot_align) # convert rotation matrix to scipy Rotation object. q_xyzw format, since scipy uses that
    quats_A_scipy = R.from_quat(quats_A_world[:, [1, 2, 3, 0]]) # convert quats_A from wxyz to xyzw for scipy

    quats_A_transformed = q_Rot_align_world * quats_A_scipy  # apply rotation to quaternions (equivalent to quats_A followed by q_rot_align))
    quats_A_transformed = quats_A_transformed.as_quat()[:, [3, 0, 1, 2]]  # convert back to wxyz format
    quats_C_all_world = quats_A_transformed
    
    #quats_A_transformed = np.array([multiply_quaternions_wxyz(quaternion_conjugate_wxyz(q), rotation_matrix_to_quaternion(R)) for q in quats_A])

    print(f"✓ Transformed all {len(quats_A_world)} camera orientations from source A")
    
    # Apply transformation to matched points for error computation
    C_matched_world = transform_points(A_matched_world, Rot_align, trans_align, scale_align) #C_matched = scale_align * (A_matched_world @ Rot_align.T) + trans_align
    
    # Compute final alignment error on matched cameras
    print("\n" + "-"*70)
    print("COMPUTING FINAL ALIGNMENT ERROR")
    print("-"*70)
    final_total_error, final_rmse, final_mean = compute_squared_error(C_matched_world, B_matched_world)
    print(f"Total Squared Error:  {final_total_error:.6f}")
    print(f"RMSE:                 {final_rmse:.6f}")
    print(f"Mean Error:           {final_mean:.6f}")
    
    # Error reduction
    error_reduction = (1 - final_total_error / initial_total_error) * 100
    print(f"\n✓ Alignment Error Reduction: {error_reduction:.2f}%")

# =======================================================================
# ===== output to txt for processing and to ply for visualization =======
# =======================================================================


    # Save all results to output directory
    print("\n" + "-"*70)
    print("SAVING ALIGNMENT RESULTS")
    print("-"*70)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always save SRT matrices
    save_srt_matrices(args.output_dir, Rot_align, trans_align, scale_align)
    
    # Always save aligned_in_world.ply for 3D visualization, Showing transformed camera positions
    aligned_ply_path = os.path.join(args.output_dir, 'aligned_in_world.ply')
    save_ply_point_cloud(aligned_ply_path, trans_C_all_world, names_A)
    print(f"✓ 3D visualization (PLY): {aligned_ply_path} for transformed camera positions")

    # Save full aligned set (all cameras)
    aligned_cam_path = os.path.join(args.output_dir, 'aligned_in_cam.txt')
    trans_C_all_cam, _, quats_C_cam = world2cam_batch(quats_C_all_world, trans_C_all_world)
    save_colmap_format(aligned_cam_path, quats_C_cam, trans_C_all_cam, names_A, even_lines_from_A)
    print(f"✓ Aligned cameras (COLMAP): {aligned_cam_path}")

    # Compare aligned cameras with reference (inputB)
    comparison_output_path = os.path.join(args.output_dir, 'comparisonOutput_vs_inputB.txt')
    compare_aligned_with_reference(aligned_cam_path, args.inputB, comparison_output_path)
    
    if args.save_params:
        initial_errors = (initial_total_error, initial_rmse, initial_mean)
        final_errors = (final_total_error, final_rmse, final_mean)
        save_transformation_params(args.output_dir, Rot_align, trans_align, scale_align, 
                                   initial_errors, final_errors,
                                   len(matched_names), len(trans_A_world), len(trans_B_world))
    
    # Verbose statistics
    if args.verbose:
        print("\n" + "="*70)
        print("DETAILED ALIGNMENT STATISTICS")
        print("="*70)
        print("\nSource cameras (matched subset):")
        print(f"  Mean position:    [{A_matched_world.mean(axis=0)[0]:10.4f}, {A_matched_world.mean(axis=0)[1]:10.4f}, {A_matched_world.mean(axis=0)[2]:10.4f}]")
        print(f"  Position spread:  [{A_matched_world.std(axis=0)[0]:10.4f}, {A_matched_world.std(axis=0)[1]:10.4f}, {A_matched_world.std(axis=0)[2]:10.4f}]")
        
        print("\nTarget cameras (matched subset):")
        print(f"  Mean position:    [{B_matched_world.mean(axis=0)[0]:10.4f}, {B_matched_world.mean(axis=0)[1]:10.4f}, {B_matched_world.mean(axis=0)[2]:10.4f}]")
        print(f"  Position spread:  [{B_matched_world.std(axis=0)[0]:10.4f}, {B_matched_world.std(axis=0)[1]:10.4f}, {B_matched_world.std(axis=0)[2]:10.4f}]")
        
        print("\nAligned cameras (transformed):")
        print(f"  Mean position:    [{C_matched_world.mean(axis=0)[0]:10.4f}, {C_matched_world.mean(axis=0)[1]:10.4f}, {C_matched_world.mean(axis=0)[2]:10.4f}]")
        print(f"  Position spread:  [{C_matched_world.std(axis=0)[0]:10.4f}, {C_matched_world.std(axis=0)[1]:10.4f}, {C_matched_world.std(axis=0)[2]:10.4f}]")
        
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