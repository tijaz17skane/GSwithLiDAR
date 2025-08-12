import numpy as np
import re
import sys
import argparse
from scipy.spatial.transform import Rotation as R

def parse_images_txt(images_txt_path):
    cameras = []
    with open(images_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            # Only process lines where first column is integer IMAGE_ID
            try:
                image_id = int(parts[0])
            except ValueError:
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]
            cameras.append({
                'id': image_id,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                't': np.array([tx, ty, tz]),
                'name': name
            })
    return cameras

def camera_center(qw, qx, qy, qz, t):
    # COLMAP: C = -R^T * t
    # COLMAP stores quaternions as (qw, qx, qy, qz)
    # scipy expects (qx, qy, qz, qw)
    rot = R.from_quat([qx, qy, qz, qw])
    Rmat = rot.as_matrix()
    C = -Rmat.T @ t
    return C

def camera_dir(qw, qx, qy, qz):
    # Camera looks along Z axis in its local frame
    rot = R.from_quat([qx, qy, qz, qw])
    return rot.apply([0, 0, 1])

def write_ply_camera_centers(cameras, ply_path):
    verts = []
    colors = []
    for cam in cameras:
        C = cam['t']  # Use TX, TY, TZ directly
        verts.append(C)
        colors.append((255, 0, 0))  # Red for center
    verts = np.array(verts)
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex {}\n'.format(len(verts)))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for v, c in zip(verts, colors):
            f.write('{:.6f} {:.6f} {:.6f} {} {} {}\n'.format(*v, *c))


def write_ply_camera_centers_and_dirs(cameras, ply_path, dir_scale=0.5):
    verts = []
    colors = []
    for cam in cameras:
        C = cam['t']  # Use TX, TY, TZ directly
        dir_vec = camera_dir(cam['qw'], cam['qx'], cam['qy'], cam['qz'])
        tip = C + dir_scale * dir_vec
        verts.append(C)
        colors.append((255, 0, 0))  # Red for center
        verts.append(tip)
        colors.append((0, 0, 255))  # Blue for direction tip
    verts = np.array(verts)
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex {}\n'.format(len(verts)))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for v, c in zip(verts, colors):
            f.write('{:.6f} {:.6f} {:.6f} {} {} {}\n'.format(*v, *c))

def write_ply_camera_axes(cameras, ply_path, axis_scale=0.5):
    # For each camera, output 3 arrows: X (red), Y (green), Z (blue)
    verts = []
    colors = []
    for cam in cameras:
        C = cam['t']  # Use TX, TY, TZ directly
        rot = R.from_quat([cam['qx'], cam['qy'], cam['qz'], cam['qw']])
        axes = rot.apply(np.eye(3))  # 3x3, columns are X, Y, Z axes
        # X axis (red)
        verts.append(C)
        colors.append((255, 0, 0))
        verts.append(C + axis_scale * axes[0])
        colors.append((255, 0, 0))
        # Y axis (green)
        verts.append(C)
        colors.append((0, 255, 0))
        verts.append(C + axis_scale * axes[1])
        colors.append((0, 255, 0))
        # Z axis (blue)
        verts.append(C)
        colors.append((0, 0, 255))
        verts.append(C + axis_scale * axes[2])
        colors.append((0, 0, 255))
    verts = np.array(verts)
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex {}\n'.format(len(verts)))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for v, c in zip(verts, colors):
            f.write('{:.6f} {:.6f} {:.6f} {} {} {}\n'.format(*v, *c))

def write_ply_tx_ty_tz(cameras, ply_path):
    # For debugging: output TX,TY,TZ directly as points
    verts = [cam['t'] for cam in cameras]
    verts = np.array(verts)
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex {}\n'.format(len(verts)))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for v in verts:
            f.write('{:.6f} {:.6f} {:.6f} 0 255 0\n'.format(*v))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert COLMAP images.txt to camera centers PLY (for CloudCompare)')
    parser.add_argument('--images', required=True, help='Path to images.txt')
    parser.add_argument('--out', required=True, help='Output PLY file')
    parser.add_argument('--raw-txt', action='store_true', help='Output TX,TY,TZ directly (for debugging)')
    parser.add_argument('--with-orient', action='store_true', help='Also output orientation as direction vector (blue tip)')
    parser.add_argument('--with-axes', action='store_true', help='Also output camera axes (red=X, green=Y, blue=Z) as arrows')
    parser.add_argument('--dir-scale', type=float, default=0.5, help='Scale for direction vector (default: 0.5)')
    parser.add_argument('--axis-scale', type=float, default=0.5, help='Scale for axes arrows (default: 0.5)')
    args = parser.parse_args()
    cameras = parse_images_txt(args.images)
    if args.raw_txt:
        write_ply_tx_ty_tz(cameras, args.out)
    elif args.with_axes:
        write_ply_camera_axes(cameras, args.out, axis_scale=args.axis_scale)
    elif args.with_orient:
        write_ply_camera_centers_and_dirs(cameras, args.out, dir_scale=args.dir_scale)
    else:
        write_ply_camera_centers(cameras, args.out)
