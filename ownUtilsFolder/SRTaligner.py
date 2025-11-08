# SRTaligner.py

import numpy as np
import argparse
import os
import sys
from cam_world_conversions import cam2world, world2cam
from scipy.spatial.transform import Rotation as R


def rotation_matrix_to_quaternion(R_mat, order='wxyz'):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Args:
        R_mat (np.ndarray): 3x3 rotation matrix.
        order (str): 'xyzw' (scipy default) or 'wxyz' (common in CV).

    Returns:
        np.ndarray: Quaternion in specified order.
    """
    # Ensure it's a numpy array
    R_mat = np.asarray(R_mat)

    # Create rotation object
    rot = R.from_matrix(R_mat)

    # Get quaternion in [x, y, z, w] order (scipy default)
    q_xyzw = rot.as_quat()

    # Reorder if needed
    if order == 'wxyz':
        q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    else:
        q = q_xyzw

    return q


def quat_wxyz_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion in wxyz order to 3x3 rotation matrix."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # xyzw
    return r.as_matrix()


def multiply_quaternions_wxyz(q1, q2):
    """Hamilton product q = q1 * q2 for quaternions in wxyz order.
    Rotation applied: first q2 then q1 (since R(q1*q2)=R(q1)@R(q2)).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def transform_quaternion(q_world, R_align):
    """Apply alignment rotation R_align (3x3) to an existing camera-to-world quaternion.
    q_world is wxyz for R_world. We want R_world' = R_align * R_world.
    Returns new world quaternion in wxyz.
    """
    # Convert inputs to SciPy rotations
    r_align = R.from_matrix(R_align)
    r_world = R.from_quat([q_world[1], q_world[2], q_world[3], q_world[0]])  # xyzw
    r_new = r_align * r_world  # apply world alignment first (left-multiply)
    q_new_xyzw = r_new.as_quat()
    q_new_wxyz = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]], dtype=np.float64)
    # Normalize for safety
    q_new_wxyz /= np.linalg.norm(q_new_wxyz)
    return q_new_wxyz

'''
def multiply_quaternions(q1, q2):
    Multiply two quaternions in wxyz order.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])



def transform_quaternion(q_original, R_matrix):
    """Transform a camera quaternion by composing it with rotation matrix R."""
    # Convert rotation matrix to quaternion
    q_R = rotation_matrix_to_quaternion(R_matrix)
    
    # Multiply q_R * q_original
    q_transformed = multiply_quaternions(q_R, q_original)

    # Normalize result
    q_transformed = q_transformed / np.linalg.norm(q_transformed)
    
    return q_transformed
'''

def load_point_cloud(filepath):
    """
    Load camera positions (TX, TY, TZ), quaternions, and names from COLMAP images.txt.
    Converts from camera coordinates to world coordinates internally.
    
    Parameters:
    -----------
    filepath : str
        Path to images.txt file
    
    Returns:
    --------
    positions : np.ndarray
        Camera positions in world coordinates as (N, 3) array
    quaternions : np.ndarray
        Camera quaternions as (N, 4) array [QW, QX, QY, QZ]
    names : list
        Corresponding names for each position
    all_lines : list
        All lines from the file for reconstruction
    """
    positions = []
    quaternions = []
    names = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse odd rows and keep all lines for reconstruction
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
            pos_world, _, q_world = cam2world(qw, qx, qy, qz, tx, ty, tz)
            positions.append(pos_world)
            quaternions.append(q_world)  # Use world quaternions
            names.append(name)
        
        # Skip next line (POINTS2D data) or move to next if at end
        i += 2
    
    return np.array(positions), np.array(quaternions), names, lines


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


def save_colmap_format(filepath, aligned_positions, aligned_quaternions, names, original_lines):
    """Save aligned positions and quaternions back to COLMAP format.
    aligned_quaternions are camera-to-world (R_world) quaternions (wxyz).
    COLMAP expects world-to-camera (R_cam). We convert via conjugation.
    Translation TX,TY,TZ stored by COLMAP is t_cam such that x_c = R_cam x_w + t_cam.
    Given camera center C_world, t_cam = -R_cam @ C_world.
    """
    with open(filepath, 'w') as f:
        aligned_idx = 0
        i = 0
        total = len(aligned_positions)
        while i < len(original_lines):
            line = original_lines[i].strip()
            if line.startswith('#'):
                f.write(original_lines[i])
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 10 and aligned_idx < total:
                image_id = parts[0]
                camera_id = parts[8]
                name = parts[9]
                # Use corresponding aligned pose (assumes order preserved by names list)
                C_world = aligned_positions[aligned_idx]
                q_world = aligned_quaternions[aligned_idx]  # camera-to-world
                # Conjugate for world-to-camera
                qw, qx, qy, qz = q_world
                q_cam = np.array([qw, -qx, -qy, -qz], dtype=np.float64)
                # Rotation matrix for q_cam
                q_cam_xyzw = [q_cam[1], q_cam[2], q_cam[3], q_cam[0]]
                R_cam = R.from_quat(q_cam_xyzw).as_matrix()
                # Camera translation
                t_cam = -R_cam @ C_world
                f.write(f"{image_id} {q_cam[0]:.8f} {q_cam[1]:.8f} {q_cam[2]:.8f} {q_cam[3]:.8f} {t_cam[0]:.8f} {t_cam[1]:.8f} {t_cam[2]:.8f} {camera_id} {name}\n")
                aligned_idx += 1
                if i + 1 < len(original_lines):
                    f.write(original_lines[i+1])
                i += 2
            else:
                # Copy any trailing lines (e.g. if unmatched cameras exist) unchanged
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


def save_colmap_world_format_as_cam(filepath, aligned_positions, aligned_quaternions, names, num_images=None):
    """Save aligned positions and quaternions in COLMAP format."""
    # Calculate statistics
    if num_images is None:
        num_images = len(aligned_positions)
    
    with open(filepath, 'w') as f:
        # Write COLMAP header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}\n")
        
        # Write camera poses using aligned quaternions and world coordinates
        for i, (point, quat, name) in enumerate(zip(aligned_positions, aligned_quaternions, names)):
            image_id = i + 1
            # Use aligned quaternions
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
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
            
            # Convert to world coordinates
            aligned_t_world, aligned_R_world, aligned_q_world = cam2world(
                aligned['qw'], aligned['qx'], aligned['qy'], aligned['qz'],
                aligned['tx'], aligned['ty'], aligned['tz']
            )
            
            ref_t_world, ref_R_world, ref_q_world = cam2world(
                ref['qw'], ref['qx'], ref['qy'], ref['qz'],
                ref['tx'], ref['ty'], ref['tz']
            )
            
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
  aligned_in_cam.txt      - Aligned cameras in COLMAP format (preserves original structure)
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
    
    # Load camera pose files
    print(f"Loading source camera poses from: {args.inputA}")
    points_A_world, quats_A, names_A, original_lines_A = load_point_cloud(args.inputA)
    print(f"  → Loaded {len(points_A_world)} cameras (converted from camera to world coordinates)")
    
    print(f"\nLoading target camera poses from: {args.inputB}")
    points_B_world, quats_B, names_B, _ = load_point_cloud(args.inputB)
    print(f"  → Loaded {len(points_B_world)} cameras (converted from camera to world coordinates)")
    
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

    # Optional: orientation-aware global 180° axis correction.
    # Try C in {I, Rx(π), Ry(π), Rz(π)} and pick the one minimizing mean angular error between
    # matched world rotations.
    dict_A_names = {n: i for i, n in enumerate(names_A)}
    dict_B_names = {n: i for i, n in enumerate(names_B)}
    qA_matched = np.array([quats_A[dict_A_names[n]] for n in matched_names])
    qB_matched = np.array([quats_B[dict_B_names[n]] for n in matched_names])

    def mean_angular_error_for_correction(C: np.ndarray) -> float:
        angles = []
        for qa, qb in zip(qA_matched, qB_matched):
            Rw_a = quat_wxyz_to_matrix(qa)
            Rw_b = quat_wxyz_to_matrix(qb)
            R_est = C @ R @ Rw_a  # predicted world rotation after alignment
            R_res = Rw_b @ R_est.T
            # Clamp trace to valid range
            trace = np.clip((np.trace(R_res) - 1) / 2.0, -1.0, 1.0)
            angle = np.degrees(np.arccos(trace)) * 2.0  # convert from half-angle relation? Use robust method below
            # Prefer robust angle from Rotation
            try:
                angle = R.from_matrix(R_res).magnitude() * 180.0 / np.pi
            except Exception:
                pass
            angles.append(angle)
        return float(np.mean(angles)) if angles else 180.0

    I = np.eye(3)
    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    Ry = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    candidates = [("I", I), ("Rx180", Rx), ("Ry180", Ry), ("Rz180", Rz)]

    best_name, best_C, best_err = None, None, 1e9
    for name_c, C in candidates:
        err = mean_angular_error_for_correction(C)
        if err < best_err:
            best_name, best_C, best_err = name_c, C, err

    if best_name is not None and best_name != "I":
        print(f"\nApplying orientation correction: {best_name} (mean angular error ~ {best_err:.3f}°)")
        R = best_C @ R
        # Recompute translation with updated rotation to preserve centroid alignment
        mu_A = np.mean(A_matched, axis=0)
        mu_B = np.mean(B_matched, axis=0)
        t = mu_B - s * (R @ mu_A)
    else:
        print(f"\nOrientation correction not needed (best mean angular error ~ {best_err:.3f}°)")
    
    # Apply transformation to ALL cameras from source A
    print("\n" + "-"*70)
    print("APPLYING TRANSFORMATION TO ALL SOURCE CAMERAS")
    print("-"*70)
    C_all_world = transform_points(points_A_world, R, t, s)
    print(f"✓ Transformed all {len(points_A_world)} camera positions from source A")

    # Transform quaternions with updated R (after any orientation correction)
    quats_A_transformed = np.array([transform_quaternion(q, R) for q in quats_A])
    print(f"✓ Transformed all {len(quats_A)} camera orientations from source A")
    
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

    # Save full aligned set (all cameras)
    aligned_cam_path = os.path.join(args.output_dir, 'aligned_in_cam.txt')
    save_colmap_format(aligned_cam_path, C_all_world, quats_A_transformed, names_A, original_lines_A)
    print(f"✓ Aligned cameras (COLMAP): {aligned_cam_path}")

    # Compare aligned cameras with reference (inputB)
    comparison_output_path = os.path.join(args.output_dir, 'comparisonOutput_vs_inputB.txt')
    compare_aligned_with_reference(aligned_cam_path, args.inputB, comparison_output_path)
    
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