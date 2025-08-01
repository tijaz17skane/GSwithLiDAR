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
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            image_id = int(parts[0])
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
    for cam in cameras:
        # Use the correct quaternion order and camera center formula
        C = camera_center(cam['qw'], cam['qx'], cam['qy'], cam['qz'], cam['t'])
        verts.append(C)
    verts = np.array(verts)
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex {}\n'.format(len(verts)))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for v in verts:
            f.write('{:.6f} {:.6f} {:.6f} 255 0 0\n'.format(*v))

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
    args = parser.parse_args()
    cameras = parse_images_txt(args.images)
    if args.raw_txt:
        write_ply_tx_ty_tz(cameras, args.out)
    else:
        write_ply_camera_centers(cameras, args.out)
