import argparse
import numpy as np

def read_points3d(path):
    points = []
    header = []
    lines = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header.append(line)
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[1:4])
            points.append([x, y, z])
            lines.append(line)
    return np.array(points), header, lines

def write_points3d(path, header, lines, keep_mask):
    with open(path, 'w') as f:
        for line in header:
            f.write(line)
        for i, line in enumerate(lines):
            if keep_mask[i]:
                f.write(line)

def remove_outliers(points, factor):
    # Remove points that are farther than mean + factor * std from the centroid
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    mean = dists.mean()
    std = dists.std()
    keep_mask = dists < (mean + factor * std)
    return keep_mask

def main():
    parser = argparse.ArgumentParser(description="Remove outliers from point cloud txt file")
    parser.add_argument('--input', required=True, help='Input txt file')
    parser.add_argument('--output', required=True, help='Output txt file')
    parser.add_argument('--factor', type=float, default=2.0, help='Outlier removal strictness (higher = less strict)')
    args = parser.parse_args()

    points, header, lines = read_points3d(args.input)
    keep_mask = remove_outliers(points, args.factor)
    write_points3d(args.output, header, lines, keep_mask)
    print(f"Kept {keep_mask.sum()} of {len(points)} points (factor={args.factor}) in {args.output}")

if __name__ == "__main__":
    main()
