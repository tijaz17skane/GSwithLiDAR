import numpy as np
import argparse

def read_srt_txt(srt_file):
    """
    Read SRT from text file.
    
    Format:
    Scale: <value>
    Rotation:
    <row1>
    <row2>
    <row3>
    Translation:
    <x y z>
    """
    with open(srt_file, 'r') as f:
        lines = f.readlines()
    
    scale = None
    rotation = []
    translation = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Scale:'):
            scale = float(line.split(':')[1].strip())
        elif line.startswith('Rotation:'):
            i += 1
            for _ in range(3):
                row = list(map(float, lines[i].strip().split()))
                rotation.append(row)
                i += 1
            i -= 1  # adjust for loop
        elif line.startswith('Translation:'):
            translation = list(map(float, lines[i+1].strip().split()))
            break
        i += 1
    
    return scale, np.array(rotation), np.array(translation)

def read_points3d_txt(points_file):
    """
    Read points3D.txt and return list of points.
    
    Each point: {'id': int, 'x':float, 'y':float, 'z':float, 'r':int, 'g':int, 'b':int, 'error':float, 'tracks': list of (img_id, p2d_idx)}
    """
    points = []
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            error = float(parts[7])
            tracks = []
            for j in range(8, len(parts), 2):
                img_id = int(parts[j])
                p2d_idx = int(parts[j+1])
                tracks.append((img_id, p2d_idx))
            points.append({
                'id': point_id,
                'x': x, 'y': y, 'z': z,
                'r': r, 'g': g, 'b': b,
                'error': error,
                'tracks': tracks
            })
    return points

def apply_srt_to_point(x, y, z, scale, rotation, translation):
    """
    Apply inverse SRT to a point (since points are in world-to-camera coordinates).
    Inverse SRT: (1/S) * R^T * (point - t)
    """
    point = np.array([x, y, z])
    inv_scale = 1.0 / scale
    inv_rotation = rotation.T
    transformed = inv_scale * np.dot(inv_rotation, point - translation)
    return transformed[0], transformed[1], transformed[2]

def write_points3d_txt(points, output_file):
    """
    Write points to points3D.txt format.
    """
    with open(output_file, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        for p in points:
            track_str = ' '.join(f'{img_id} {p2d_idx}' for img_id, p2d_idx in p['tracks'])
            f.write(f"{p['id']} {p['x']:.6f} {p['y']:.6f} {p['z']:.6f} {p['r']} {p['g']} {p['b']} {p['error']:.6f} {track_str}\n")

def main():
    parser = argparse.ArgumentParser(description='Apply SRT to points3D.txt')
    parser.add_argument('--SRT_txt', required=True, help='Path to SRT text file')
    parser.add_argument('--inputPoints3D', required=True, help='Path to input points3D.txt')
    parser.add_argument('--outputPoints3D', required=True, help='Path to output points3D.txt')
    
    args = parser.parse_args()
    
    # Read SRT
    scale, rotation, translation = read_srt_txt(args.SRT_txt)
    print(f"Applying Scale: {scale}, Rotation:\n{rotation}, Translation: {translation}")
    
    # Read points
    points = read_points3d_txt(args.inputPoints3D)
    
    # Apply SRT to each point
    for p in points:
        p['x'], p['y'], p['z'] = apply_srt_to_point(p['x'], p['y'], p['z'], scale, rotation, translation)
    
    # Write output
    write_points3d_txt(points, args.outputPoints3D)
    print(f"Transformed points3D.txt written to {args.outputPoints3D}")

if __name__ == "__main__":
    main()