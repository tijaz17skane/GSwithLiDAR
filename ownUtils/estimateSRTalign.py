import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

def read_images_txt(file_path):
    """
    Read COLMAP images.txt file and return list of images with poses and points2d.
    
    Returns:
    - images: list of dicts, each with 'image_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'camera_id', 'name', 'points2d'
    """
    images = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        # First line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = ' '.join(parts[9:])
        
        # Second line: POINTS2D[] as (X, Y, POINT3D_ID)
        i += 1
        points2d_line = lines[i].strip()
        points2d = []
        if points2d_line:
            # Parse points2d, assuming format like "x1 y1 p3d1 x2 y2 p3d2 ..."
            parts = points2d_line.split()
            for j in range(0, len(parts), 3):
                x, y, p3d_id = float(parts[j]), float(parts[j+1]), int(parts[j+2])
                points2d.append((x, y, p3d_id))
        
        images.append({
            'image_id': image_id,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
            'camera_id': camera_id,
            'name': name,
            'points2d': points2d
        })
        i += 1
    
    return images

def compute_camera_centers(images):
    """
    Compute camera centers from poses.
    
    Camera center c = -R^T * t, where R is from quaternion, t is translation.
    """
    centers = []
    for img in images:
        # Quaternion to rotation matrix
        qw, qx, qy, qz = img['qw'], img['qx'], img['qy'], img['qz']
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([img['tx'], img['ty'], img['tz']])
        center = -np.dot(R.T, t)
        centers.append(center)
    return np.array(centers)

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Convert quaternion to rotation matrix.
    """
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def estimate_srt(reference_centers, target_centers):
    """
    Estimate Scale, Rotation, and Translation (SRT) to align target_centers to reference_centers.
    """
    # Compute centroids
    ref_centroid = np.mean(reference_centers, axis=0)
    tgt_centroid = np.mean(target_centers, axis=0)
    
    # Center the points
    ref_centered = reference_centers - ref_centroid
    tgt_centered = target_centers - tgt_centroid
    
    # Compute covariance matrix
    H = np.dot(tgt_centered.T, ref_centered)
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R_matrix = np.dot(Vt.T, U.T)
    
    # Ensure right-handed rotation
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)
    
    # Compute scale
    scale = np.sum(S) / np.sum(np.linalg.norm(tgt_centered, axis=1)**2)
    
    # Compute translation
    translation = ref_centroid - scale * np.dot(R_matrix, tgt_centroid)
    
    return scale, R_matrix, translation

def apply_srt_to_pose(qw, qx, qy, qz, tx, ty, tz, scale, R_s, t_s):
    """
    Apply SRT to a pose.
    New R' = R_s * R, t' = scale * R_s * t + t_s
    """
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    t = np.array([tx, ty, tz])
    
    R_new = np.dot(R_s, R)
    t_new = scale * np.dot(R_s, t) + t_s
    
    # Convert back to quaternion
    qw_new, qx_new, qy_new, qz_new = rotation_matrix_to_quaternion(R_new)
    
    return qw_new, qx_new, qy_new, qz_new, t_new[0], t_new[1], t_new[2]

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.
    """
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4 * qw)
    qy = (R[0,2] - R[2,0]) / (4 * qw)
    qz = (R[1,0] - R[0,1]) / (4 * qw)
    return qw, qx, qy, qz

def write_images_txt(images, file_path):
    """
    Write images to COLMAP images.txt format.
    """
    with open(file_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        for img in images:
            f.write(f"{img['image_id']} {img['qw']:.6f} {img['qx']:.6f} {img['qy']:.6f} {img['qz']:.6f} {img['tx']:.6f} {img['ty']:.6f} {img['tz']:.6f} {img['camera_id']} {img['name']}\n")
            points2d_str = ' '.join(f"{x:.2f} {y:.2f} {p3d_id}" for x, y, p3d_id in img['points2d'])
            f.write(f"{points2d_str}\n")

def main():
    parser = argparse.ArgumentParser(description='Estimate SRT from reference to target images.txt and apply to target.')
    parser.add_argument('--reference', required=True, help='Path to reference images.txt')
    parser.add_argument('--target', required=True, help='Path to target images.txt')
    parser.add_argument('--output', required=True, help='Path to output transformed images.txt')
    parser.add_argument('--output_matrix', help='Path to output SRT matrix file')
    
    args = parser.parse_args()
    
    # Read reference and target
    ref_images = read_images_txt(args.reference)
    tgt_images = read_images_txt(args.target)
    
    # Create dicts of name to image for matching
    ref_dict = {img['name']: img for img in ref_images}
    tgt_dict = {img['name']: img for img in tgt_images}
    
    # Find common names
    common_names = set(ref_dict.keys()) & set(tgt_dict.keys())
    if not common_names:
        print("No common image names found between reference and target.")
        return
    
    print(f"Found {len(common_names)} common images.")
    
    # Compute camera centers for common images
    ref_centers = []
    tgt_centers = []
    for name in common_names:
        ref_img = ref_dict[name]
        tgt_img = tgt_dict[name]
        
        # Compute ref center
        qw, qx, qy, qz = ref_img['qw'], ref_img['qx'], ref_img['qy'], ref_img['qz']
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([ref_img['tx'], ref_img['ty'], ref_img['tz']])
        ref_center = -np.dot(R.T, t)
        ref_centers.append(ref_center)
        
        # Compute tgt center
        qw, qx, qy, qz = tgt_img['qw'], tgt_img['qx'], tgt_img['qy'], tgt_img['qz']
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tgt_img['tx'], tgt_img['ty'], tgt_img['tz']])
        tgt_center = -np.dot(R.T, t)
        tgt_centers.append(tgt_center)
    
    ref_centers = np.array(ref_centers)
    tgt_centers = np.array(tgt_centers)
    
    # Estimate SRT
    scale, rotation, translation = estimate_srt(ref_centers, tgt_centers)
    
    print(f"Estimated Scale: {scale}")
    print(f"Estimated Rotation:\n{rotation}")
    print(f"Estimated Translation: {translation}")
    
    # Output matrix if specified
    if args.output_matrix:
        with open(args.output_matrix, 'w') as f:
            f.write(f"Scale: {scale}\n")
            f.write("Rotation:\n")
            for row in rotation:
                f.write(' '.join(f"{x:.6f}" for x in row) + '\n')
            f.write("Translation:\n")
            f.write(' '.join(f"{x:.6f}" for x in translation) + '\n')
        print(f"SRT matrix written to {args.output_matrix}")
    
    # Apply SRT to target poses
    transformed_images = []
    for img in tgt_images:
        qw, qx, qy, qz, tx, ty, tz = apply_srt_to_pose(
            img['qw'], img['qx'], img['qy'], img['qz'], img['tx'], img['ty'], img['tz'],
            scale, rotation, translation
        )
        transformed_images.append({
            'image_id': img['image_id'],
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
            'camera_id': img['camera_id'],
            'name': img['name'],
            'points2d': img['points2d']
        })
    
    # Write output
    write_images_txt(transformed_images, args.output)
    print(f"Transformed images.txt written to {args.output}")

if __name__ == "__main__":
    main()