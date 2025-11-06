# SRTaligner_translation_only.py

import numpy as np
import argparse
import os
from cam_world_conversions import cam2world


def load_camera_positions(filepath):
    """
    Load camera positions (tx, ty, tz) from COLMAP images.txt format.
    
    Returns:
    --------
    positions : np.ndarray
        Camera positions as (N, 3) array [TX, TY, TZ]
    names : list
        Corresponding names for each pose
    """
    positions = []
    names = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip comments
        if line.startswith('#'):
            continue
        
        # Parse camera pose line
        parts = line.split()
        if len(parts) >= 10:
            # Extract position (indices 5-7)
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]
            
            positions.append([tx, ty, tz])
            names.append(name)
    
    return np.array(positions, dtype=np.float64), names


def load_world_positions_from_images(filepath):
    """
    Load COLMAP images.txt and convert each pose from camera coordinates to world camera centers using cam2world.
    Uses quaternion (QW,QX,QY,QZ) and translation (TX,TY,TZ) per line, returns world camera center positions and image names.
    """
    positions_world = []
    names = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            # Parse COLMAP order: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]
            # Convert to world camera center
            C_world, _ = cam2world(qw, qx, qy, qz, tx, ty, tz)
            positions_world.append(C_world)
            names.append(name)
    return np.asarray(positions_world, dtype=np.float64), names


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
    common_names = sorted(list(set(names_A) & set(names_B)))
    
    if len(common_names) == 0:
        print("Error: No matching names found between the two point clouds!")
        exit(1)
    
    # Extract matched points
    matched_A = np.array([points_A[dict_A[name]] for name in common_names], dtype=np.float64)
    matched_B = np.array([points_B[dict_B[name]] for name in common_names], dtype=np.float64)
    
    return matched_A, matched_B, common_names


def compute_srt_matrices(source_points, target_points):
    """
    Compute SRT transformation matrices to align source to target (Procrustes / Umeyama-style on positions).
    Returns R (3x3), t (3,), s (float).
    """
    # Compute centroids
    centroid_A = np.mean(source_points, axis=0)
    centroid_B = np.mean(target_points, axis=0)
    
    # Center the points
    X = source_points - centroid_A
    Y = target_points - centroid_B
    
    # Cross-covariance
    H = X.T @ Y
    
    # SVD
    U, S_sv, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    # Scale (Umeyama variant)
    var_X = np.sum(np.sum(X**2, axis=1)) / X.shape[0]
    s = np.trace(np.diag(S_sv)) / (X.shape[0] * var_X) if var_X > 0 else 1.0
    
    # Translation
    t = centroid_B - s * (R @ centroid_A)
    return R, t, s


def transform_points(points, R, t, s):
    """Apply similarity transformation: p' = s * R * p + t for a set of row-vector points."""
    return (points @ R.T) * s + t


def trimmed_srt_fit(A, B, trim_ratio=0.9, max_iters=2):
    """
    Robust SRT fit by trimming to the best trim_ratio fraction of correspondences.
    Returns R, t, s and indices of inliers used in the final fit.
    """
    assert A.shape == B.shape and A.shape[0] >= 3
    N = A.shape[0]
    K = max(3, int(np.ceil(trim_ratio * N)))
    # Initial fit
    R, t, s = compute_srt_matrices(A, B)
    keep_idx = np.arange(N)
    for _ in range(max_iters):
        A_to_B = transform_points(A, R, t, s)
        errors = np.linalg.norm(A_to_B - B, axis=1)
        keep_idx = np.argsort(errors)[:K]
        R, t, s = compute_srt_matrices(A[keep_idx], B[keep_idx])
    return R, t, s, keep_idx


def ransac_srt_fit(A, B, thresh=0.1, max_iters=1000, min_samples=3, seed=None):
    """
    RANSAC for SRT estimation using 3-point minimal sets.
    thresh: inlier distance threshold in world units.
    Returns R, t, s, inlier_indices.
    """
    assert A.shape == B.shape and A.shape[0] >= min_samples
    N = A.shape[0]
    rng = np.random.default_rng(seed)

    best_inliers = np.array([], dtype=int)
    best_R, best_t, best_s = None, None, None

    for _ in range(max_iters):
        # Sample minimal set
        idx = rng.choice(N, size=min_samples, replace=False)
        try:
            R_i, t_i, s_i = compute_srt_matrices(A[idx], B[idx])
        except Exception:
            continue
        # Score
        A_to_B = transform_points(A, R_i, t_i, s_i)
        errors = np.linalg.norm(A_to_B - B, axis=1)
        inliers = np.where(errors <= thresh)[0]
        if inliers.size >= max(best_inliers.size, min_samples):
            # Refit on inliers
            try:
                R_ref, t_ref, s_ref = compute_srt_matrices(A[inliers], B[inliers])
            except Exception:
                continue
            # Evaluate RMSE on inliers for tie-break
            A_to_B_ref = transform_points(A[inliers], R_ref, t_ref, s_ref)
            rmse_ref = float(np.sqrt(np.mean(np.linalg.norm(A_to_B_ref - B[inliers], axis=1)**2)))
            if (inliers.size > best_inliers.size) or (inliers.size == best_inliers.size and best_R is not None and rmse_ref < np.inf):
                best_inliers = inliers
                best_R, best_t, best_s = R_ref, t_ref, s_ref

    # Fallback if nothing good found
    if best_R is None or best_inliers.size < min_samples:
        R_f, t_f, s_f = compute_srt_matrices(A, B)
        return R_f, t_f, s_f, np.arange(N)

    return best_R, best_t, best_s, best_inliers


def save_srt_matrices(output_dir, R, t, s):
    """Save Scale, Rotation, Translation as separate matrices and combined 4x4 matrix."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Scale matrix (4x4)
    S = np.eye(4, dtype=np.float64)
    S[:3, :3] *= s
    
    # Rotation matrix (4x4)
    R_4x4 = np.eye(4, dtype=np.float64)
    R_4x4[:3, :3] = R
    
    # Translation matrix (4x4)
    T_4x4 = np.eye(4, dtype=np.float64)
    T_4x4[:3, 3] = t
    
    # Combined transformation matrix: T * R * S
    combined = T_4x4 @ R_4x4 @ S
    
    # Save individual matrices
    np.savetxt(os.path.join(output_dir, 'scale_matrix.txt'), S, fmt='%.8f', header="Scale Matrix (4x4)")
    np.savetxt(os.path.join(output_dir, 'rotation_matrix.txt'), R_4x4, fmt='%.8f', header="Rotation Matrix (4x4)")
    np.savetxt(os.path.join(output_dir, 'translation_matrix.txt'), T_4x4, fmt='%.8f', header="Translation Matrix (4x4)")
    np.savetxt(os.path.join(output_dir, 'combined_transform.txt'), combined, fmt='%.8f', header="Combined Transformation Matrix: T*R*S")
    
    print(f"✓ SRT matrices saved to {output_dir}:")
    print(f"  - Scale matrix")
    print(f"  - Rotation matrix")
    print(f"  - Translation matrix")
    print(f"  - Combined transformation matrix")


def main():
    parser = argparse.ArgumentParser(description='Compute SRT alignment matrices from camera poses (cam2world → align on positions)')
    parser.add_argument('--source', type=str, required=True, help='Source COLMAP images.txt file')
    parser.add_argument('--target', type=str, required=True, help='Target COLMAP images.txt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for transformation matrices')
    # Robust method selection
    parser.add_argument('--robust', type=str, default='trimmed', choices=['none', 'trimmed', 'ransac'], help='Robust fitting method')
    parser.add_argument('--trim', type=float, default=0.9, help='Trim ratio for trimmed LS (0<trim<=1)')
    parser.add_argument('--ransac_thresh', type=float, default=0.10, help='RANSAC inlier threshold in world units')
    parser.add_argument('--ransac_iters', type=int, default=1000, help='RANSAC max iterations')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for RANSAC')
    args = parser.parse_args()
    
    # 1) Load and convert to world coordinates (camera centers)
    print('Loading poses and converting to world camera centers (cam2world)...')
    A_world, names_A = load_world_positions_from_images(args.source)
    B_world, names_B = load_world_positions_from_images(args.target)
    print(f"  Source cameras: {len(A_world)}  Target cameras: {len(B_world)}")
    
    # 2) Match by names
    print('Matching cameras by name...')
    A_matched, B_matched, matched_names = match_point_correspondences(A_world, names_A, B_world, names_B)
    print(f"  Matched pairs: {len(matched_names)}")
    if len(matched_names) < 3:
        print('Need at least 3 matched pairs for alignment!')
        exit(1)
    
    # 3) Robust SRT on positions only
    method = args.robust
    if method == 'none':
        print('Estimating SRT (no robust method)...')
        R, t, s = compute_srt_matrices(A_matched, B_matched)
        inliers = np.arange(A_matched.shape[0])
    elif method == 'trimmed':
        trim = max(0.5, min(1.0, float(args.trim)))
        print(f'Estimating SRT with trimmed set (trim={trim:.2f})...')
        R, t, s, inliers = trimmed_srt_fit(A_matched, B_matched, trim_ratio=trim, max_iters=2)
    else:  # ransac
        print(f'Estimating SRT with RANSAC (thresh={args.ransac_thresh}, iters={args.ransac_iters})...')
        R, t, s, inliers = ransac_srt_fit(
            A_matched, B_matched,
            thresh=float(args.ransac_thresh),
            max_iters=int(args.ransac_iters),
            min_samples=3,
            seed=args.seed,
        )
    
    # Report errors
    A_to_B = transform_points(A_matched, R, t, s)
    errors = np.linalg.norm(A_to_B - B_matched, axis=1)
    rmse_all = float(np.sqrt(np.mean(errors**2)))
    rmse_in = float(np.sqrt(np.mean(errors[inliers]**2))) if inliers.size > 0 else rmse_all
    print(f"  Inliers used: {len(inliers)}/{len(matched_names)}")
    print(f"  RMSE (all): {rmse_all:.6f}   RMSE (inliers): {rmse_in:.6f}")
    
    # 4) Save matrices (S, R, T, and combined T*R*S)
    print('Saving transformation matrices...')
    save_srt_matrices(args.output_dir, R, t, s)
    print('Done!')


if __name__ == "__main__":
    main()