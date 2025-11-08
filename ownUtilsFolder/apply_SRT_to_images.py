import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from cam_world_conversions import cam2world


def parse_images_file(images_path):
    """
    Parse COLMAP images.txt file (two lines per image: pose/meta line + points2D line).
    Returns a list of dict entries with keys:
      image_id, qw,qx,qy,qz, tx,ty,tz, camera_id, name, points2d
    Preserves order.
    """
    entries = []
    headers = []
    with open(images_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]
    i = 0
    N = len(lines)
    # collect header/comment lines
    while i < N and (not lines[i] or lines[i].startswith('#')):
        headers.append(lines[i])
        i += 1
    # parse pairs
    while i < N:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            # Not a valid pose line, skip
            i += 1
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        # next line is points2D (may be empty)
        points2d = ''
        if i + 1 < N:
            points2d = lines[i + 1].strip()
        entries.append({
            'image_id': image_id,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
            'camera_id': camera_id,
            'name': name,
            'points2d': points2d,
        })
        i += 2
    return headers, entries


def load_combined_transform(transform_dir=None, combined_path=None, S_path=None, R_path=None, T_path=None):
    """
    Load a 4x4 combined transform matrix M = T * R * S.
    Priority: combined_path -> transform_dir/combined_transform.txt -> (T,R,S) files.
    Returns (M, R_srt, s) where R_srt is the 3x3 rotation part and s is isotropic scale.
    """
    M = None
    if combined_path and os.path.exists(combined_path):
        M = np.loadtxt(combined_path)
    elif transform_dir:
        comb_try = os.path.join(transform_dir, 'combined_transform.txt')
        if os.path.exists(comb_try):
            M = np.loadtxt(comb_try)
        else:
            # Try separate S/R/T
            S_try = S_path or os.path.join(transform_dir, 'scale_matrix.txt')
            R_try = R_path or os.path.join(transform_dir, 'rotation_matrix.txt')
            T_try = T_path or os.path.join(transform_dir, 'translation_matrix.txt')
            if all(os.path.exists(p) for p in [S_try, R_try, T_try]):
                S = np.loadtxt(S_try)
                Rm = np.loadtxt(R_try)
                Tm = np.loadtxt(T_try)
                M = Tm @ Rm @ S
    elif all(p is not None and os.path.exists(p) for p in [S_path, R_path, T_path]):
        S = np.loadtxt(S_path)
        Rm = np.loadtxt(R_path)
        Tm = np.loadtxt(T_path)
        M = Tm @ Rm @ S

    if M is None:
        raise FileNotFoundError('Could not load transform. Provide --combined or --transform_dir (with combined or S/R/T).')

    # Extract rotation (normalized) and isotropic scale from upper-left 3x3
    A = M[:3, :3]
    # average column norms (robust for uniform scale)
    col_norms = [np.linalg.norm(A[:, j]) for j in range(3)]
    s = float(np.mean(col_norms)) if np.mean(col_norms) != 0 else 1.0
    R_srt = A / s if s != 0 else np.eye(3)
    # Fix potential reflection
    if np.linalg.det(R_srt) < 0:
        R_srt = -R_srt
        s = -s
    return M, R_srt, s


def apply_srt_to_entries(entries, M, R_srt):
    """
    For each entry, convert to world (C_world, R_c2w), apply SRT to get C_world', R_c2w',
    then convert back to R_w2c', t'. Returns updated entries and a parallel list of aligned
    world camera centers for debug outputs.
    """
    aligned_centers = []
    updated = []
    for e in entries:
        # cam2world: returns camera center C_world and R_c2w
        Cw, R_c2w = cam2world(e['qw'], e['qx'], e['qy'], e['qz'], e['tx'], e['ty'], e['tz'])
        Cw = np.asarray(Cw, dtype=np.float64)
        # Apply position transform using combined matrix M
        Cw_h = np.append(Cw, 1.0)
        Cw_p = (M @ Cw_h)[:3]
        # Update rotation in world: R_c2w' = R_srt @ R_c2w
        R_c2w_p = R_srt @ R_c2w
        # Convert back to COLMAP (world-to-camera)
        R_w2c_p = R_c2w_p.T
        # t' = -R_w2c' * C_world'
        t_p = -R_w2c_p @ Cw_p
        # Quaternion in COLMAP order (qw, qx, qy, qz). scipy returns [x,y,z,w]
        q_xyzw = R.from_matrix(R_w2c_p).as_quat()
        qx_p, qy_p, qz_p, qw_p = q_xyzw
        aligned_centers.append((e['name'], Cw_p))
        updated.append({
            'image_id': e['image_id'],
            'qw': float(qw_p), 'qx': float(qx_p), 'qy': float(qy_p), 'qz': float(qz_p),
            'tx': float(t_p[0]), 'ty': float(t_p[1]), 'tz': float(t_p[2]),
            'camera_id': e['camera_id'],
            'name': e['name'],
            'points2d': e['points2d'],
        })
    return updated, aligned_centers


def write_images_cam_txt(path, headers, entries):
    """
    Write a COLMAP-style images.txt with updated pose lines and preserved points2D lines.
    """
    with open(path, 'w') as f:
        # write standard header
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write(f'# Number of images: {len(entries)}\n')
        for e in entries:
            f.write(f"{e['image_id']} {e['qw']} {e['qx']} {e['qy']} {e['qz']} {e['tx']} {e['ty']} {e['tz']} {e['camera_id']} {e['name']}\n")
            f.write((e['points2d'] or '') + '\n')


def write_world_txt(path, centers):
    """
    Write a simple TXT listing aligned world camera centers: NAME Cx Cy Cz
    """
    with open(path, 'w') as f:
        f.write('# Aligned camera centers in world coordinates\n')
        f.write('# NAME Cx Cy Cz\n')
        for name, C in centers:
            f.write(f"{name} {C[0]} {C[1]} {C[2]}\n")


def write_world_ply(path, centers):
    """
    Write an ASCII PLY with vertices at the aligned world camera centers.
    """
    N = len(centers)
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {N}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for _, C in centers:
            f.write(f"{C[0]} {C[1]} {C[2]}\n")


def main():
    parser = argparse.ArgumentParser(description='Apply SRT to COLMAP images.txt: cam2world -> SRT -> outputs (world TXT/PLY, cam TXT).')
    parser.add_argument('--input_txt', required=True, help='Path to input images.txt')
    parser.add_argument('--output_txt', required=True, help='Path to output aligned COLMAP images.txt')
    parser.add_argument('--combined_transform', required=True, help='Path to combined 4x4 transform (T*R*S)')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_txt) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Load inputs
    print('Parsing images file...')
    headers, entries = parse_images_file(args.input_txt)
    print(f'  Found {len(entries)} images')

    print('Loading transform...')
    M, R_srt, s = load_combined_transform(
        transform_dir=None,
        combined_path=args.combined_transform,
        S_path=None, R_path=None, T_path=None,
    )
    print('  Loaded transform. Scale estimate:', s)

    # Apply SRT
    print('Applying SRT to poses...')
    updated_entries, aligned_centers = apply_srt_to_entries(entries, M, R_srt)

    # Outputs
    cam_txt = args.output_txt
    world_txt = os.path.join(out_dir, 'imagesAlignedworld.txt')
    world_ply = os.path.join(out_dir, 'imagesAlignedworld.ply')

    print('Writing debug world TXT...')
    write_world_txt(world_txt, aligned_centers)
    print('Writing world PLY...')
    write_world_ply(world_ply, aligned_centers)
    print('Writing aligned COLMAP images...')
    write_images_cam_txt(cam_txt, headers, updated_entries)

    print('Done.')


if __name__ == '__main__':
    main()
