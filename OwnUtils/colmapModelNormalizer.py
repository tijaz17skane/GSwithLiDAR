import argparse
import os
import numpy as np
from cam_world_conversions import cam2world, world2cam

def parse_args():
    parser = argparse.ArgumentParser(description="Normalize COLMAP model translations.")
    parser.add_argument('--input_path', required=True, help='Input COLMAP model directory containing images.txt, points3D.txt, cameras.txt')
    parser.add_argument('--output_path', required=True, help='Output directory for normalized COLMAP model')
    return parser.parse_args()

def read_images_txt(path):
    images = []
    header = []
    with open(path, 'r') as f:
        lines = f.readlines()
    # Collect header lines (start with #)
    idx = 0
    while idx < len(lines) and lines[idx].startswith('#'):
        header.append(lines[idx])
        idx += 1
    # Now process pairs of lines (image entry + points2D)
    data_lines = lines[idx:]
    for i in range(0, len(data_lines), 2):
        if i + 1 >= len(data_lines):
            break
        line1 = data_lines[i]
        line2 = data_lines[i + 1] if i + 1 < len(data_lines) else '\n'
        
        parts = line1.strip().split()
        if len(parts) != 10:
            continue
        image = {
            'IMAGE_ID': int(parts[0]),
            'QW': float(parts[1]),
            'QX': float(parts[2]),
            'QY': float(parts[3]),
            'QZ': float(parts[4]),
            'TX': float(parts[5]),
            'TY': float(parts[6]),
            'TZ': float(parts[7]),
            'CAMERA_ID': int(parts[8]),
            'NAME': parts[9],
            'points2D_line': line2  # Store the second line as-is
        }
        images.append(image)
    return images, header

def read_points3D_txt(path):
    points = []
    header = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            header.append(line)
            continue
        parts = line.strip().split()
        if len(parts) < 7:
            header.append(line)
            continue
        # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
        point = {
            'POINT3D_ID': int(parts[0]),
            'X': float(parts[1]),
            'Y': float(parts[2]),
            'Z': float(parts[3]),
            'R': int(parts[4]),
            'G': int(parts[5]),
            'B': int(parts[6]),
            'ERROR': float(parts[7]) if len(parts) > 7 else 0.0,
            'rest': ' '.join(parts[8:]) if len(parts) > 8 else ''
        }
        points.append(point)
    return points, header

def write_images_txt(path, header, images):
    with open(path, 'w') as f:
        for line in header:
            f.write(line)
        for img in images:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            f.write(f"{img['IMAGE_ID']} {img['QW']:.8f} {img['QX']:.8f} {img['QY']:.8f} {img['QZ']:.8f} {img['TX']:.8f} {img['TY']:.8f} {img['TZ']:.8f} {img['CAMERA_ID']} {img['NAME']}\n")
            # Write the POINTS2D line
            f.write(img['points2D_line'])

def write_points3D_txt(path, header, points):
    with open(path, 'w') as f:
        for line in header:
            f.write(line)
        for pt in points:
            # POINT3D_ID X Y Z R G B ERROR TRACK[]
            rest = f" {pt['rest']}" if pt['rest'] else ""
            f.write(f"{pt['POINT3D_ID']} {pt['X']:.8f} {pt['Y']:.8f} {pt['Z']:.8f} {pt['R']} {pt['G']} {pt['B']} {pt['ERROR']:.4f}{rest}\n")

def write_cameras_txt(src_path, dst_path):
    with open(src_path, 'r') as fsrc, open(dst_path, 'w') as fdst:
        for line in fsrc:
            fdst.write(line)

def write_normalization_transform(path, offset):
    mat = np.eye(4)
    mat[:3, 3] = offset
    with open(path, 'w') as f:
        for row in mat:
            f.write(' '.join(f"{v:.8f}" for v in row) + '\n')

def write_camera_centers_ply(path, positions):
    """Write camera centers as a PLY point cloud."""
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(positions)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('end_header\n')
        for pos in positions:
            f.write(f'{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n')

def main():
    args = parse_args()
    images_path = os.path.join(args.input_path, 'images.txt')
    points_path = os.path.join(args.input_path, 'points3D.txt')
    cameras_path = os.path.join(args.input_path, 'cameras.txt')

    out_images_path = os.path.join(args.output_path, 'images.txt')
    out_points_path = os.path.join(args.output_path, 'points3D.txt')
    out_cameras_path = os.path.join(args.output_path, 'cameras.txt')
    out_norm_path = os.path.join(args.output_path, 'normalization_transform.txt')
    out_ply_path = os.path.join(args.output_path, 'camera_centers.ply')

    images, images_header = read_images_txt(images_path)
    points, points_header = read_points3D_txt(points_path)

    # Convert camera coordinates to world coordinates
    world_positions = []
    for img in images:
        pos_world, _ = cam2world(img['QW'], img['QX'], img['QY'], img['QZ'], img['TX'], img['TY'], img['TZ'])
        world_positions.append(pos_world)
    world_positions = np.array(world_positions)

    # Compute normalization offset (mean of world positions)
    offset = world_positions.mean(axis=0)

    # Normalize world positions
    norm_world_positions = world_positions - offset

    # Write normalized camera centers as PLY
    write_camera_centers_ply(out_ply_path, norm_world_positions)

    # Update images: convert normalized world positions back to camera coordinates
    for i, img in enumerate(images):
        norm_pos = norm_world_positions[i]
        t_cam, _ = world2cam(img['QW'], img['QX'], img['QY'], img['QZ'], norm_pos[0], norm_pos[1], norm_pos[2])
        img['TX'] = t_cam[0]
        img['TY'] = t_cam[1]
        img['TZ'] = t_cam[2]

    # Normalize points3D
    for pt in points:
        pt['X'] = pt['X'] - offset[0]
        pt['Y'] = pt['Y'] - offset[1]
        pt['Z'] = pt['Z'] - offset[2]

    # Write outputs
    os.makedirs(args.output_path, exist_ok=True)
    write_images_txt(out_images_path, images_header, images)
    write_points3D_txt(out_points_path, points_header, points)
    write_cameras_txt(cameras_path, out_cameras_path)
    write_normalization_transform(out_norm_path, offset)
    print(f"Normalized model written to {args.output_path}")

if __name__ == "__main__":
    main()