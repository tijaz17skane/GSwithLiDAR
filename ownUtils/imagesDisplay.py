import numpy as np
import argparse
import sys
import os
from collections import namedtuple

# Import from colmap_loader and graphics_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scene'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from colmap_loader import read_extrinsics_text, qvec2rotmat
from graphics_utils import getWorld2View2

# Define CameraInfo and getNerfppNorm locally to avoid import issues
CameraInfo = namedtuple('CameraInfo', ['uid', 'R', 'T', 'FovY', 'FovX', 'depth_params', 'image_path', 'image_name', 'depth_path', 'width', 'height', 'is_test'])

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def read_images_txt(file_path):
    """Read COLMAP images.txt using the existing loader."""
    cam_extrinsics = read_extrinsics_text(file_path)
    images = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        images.append({
            'image_id': extr.id,
            'qw': extr.qvec[0], 'qx': extr.qvec[1], 'qy': extr.qvec[2], 'qz': extr.qvec[3],
            'tx': extr.tvec[0], 'ty': extr.tvec[1], 'tz': extr.tvec[2],
            'camera_id': extr.camera_id,
            'name': extr.name
        })
    return images

def create_cam_infos(images):
    """Create CameraInfo objects for getNerfppNorm."""
    cam_infos = []
    for img in images:
        qvec = np.array([img['qw'], img['qx'], img['qy'], img['qz']])
        R_wc = qvec2rotmat(qvec)
        t_wc = np.array([img['tx'], img['ty'], img['tz']])
        R_cw = R_wc.T  # camera-to-world rotation
        cam_info = CameraInfo(
            uid=img['image_id'],
            R=R_cw,
            T=t_wc,
            FovY=0,  # dummy
            FovX=0,  # dummy
            depth_params=None,
            image_path='',
            image_name=img['name'],
            depth_path='',
            width=0,  # dummy
            height=0,  # dummy
            is_test=False
        )
        cam_infos.append(cam_info)
    return cam_infos

def write_images_txt(images, file_path):
    """Write images to images.txt format, skipping points2d."""
    with open(file_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        for img in images:
            f.write(f"{img['image_id']} {img['qw']:.6f} {img['qx']:.6f} {img['qy']:.6f} {img['qz']:.6f} {img['tx']:.6f} {img['ty']:.6f} {img['tz']:.6f} {img['camera_id']} {img['name']}\n")
            # Skip points2d, just write a placeholder or nothing
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Convert images.txt by centering the scene.')
    parser.add_argument('--input', required=True, help='Path to input images.txt')
    parser.add_argument('--output', required=True, help='Path to output imagesConverted.txt')
    
    args = parser.parse_args()
    
    # Read images
    images = read_images_txt(args.input)
    
    # Create CameraInfo objects
    cam_infos = create_cam_infos(images)
    
    # Compute normalization using the same logic as dataset_readers
    nerf_norm = getNerfppNorm(cam_infos)
    center = -nerf_norm['translate']
    
    print(f"Average camera center: {center}")
    
    # Adjust translations to center the scene
    for img in images:
        old_t = np.array([img['tx'], img['ty'], img['tz']])
        new_t = old_t - center
        img['tx'], img['ty'], img['tz'] = new_t
    
    # Write output
    write_images_txt(images, args.output)
    print(f"Converted images.txt written to {args.output}")

if __name__ == "__main__":
    main()