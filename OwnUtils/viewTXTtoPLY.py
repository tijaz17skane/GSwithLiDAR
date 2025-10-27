import argparse
import numpy as np

def read_points3d(path):
    points = []
    colors = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            points.append([x, y, z])
            colors.append([r, g, b])
    return np.array(points), np.array(colors)

def write_ply(path, points, colors):
    n = len(points)
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for pt, col in zip(points, colors):
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {col[0]} {col[1]} {col[2]}\n')

def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP points3D.txt to PLY format")
    parser.add_argument('--input', required=True, help='Input points3D.txt file')
    parser.add_argument('--output', required=True, help='Output PLY file')
    args = parser.parse_args()

    points, colors = read_points3d(args.input)
    write_ply(args.output, points, colors)
    print(f"Wrote {len(points)} points to {args.output}")

if __name__ == "__main__":
    main()
