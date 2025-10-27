import argparse
import numpy as np

def read_points3d(path):
    points = []
    header = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header.append(line)
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            # We want X, Y, Z (columns 1, 2, 3)
            x, y, z = map(float, parts[1:4])
            points.append([x, y, z])
    return np.array(points), header

def write_points3d(path, header, original_lines, bbox_points, start_id):
    with open(path, 'w') as f:
        for line in header:
            f.write(line)
        for line in original_lines:
            if not line.startswith('#') and line.strip():
                f.write(line)
        for i, pt in enumerate(bbox_points):
            # Write bounding box points with white color and error 0.0, no track
            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR
            f.write(f"{start_id+i} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 255 255 255 0.0\n")

def generate_bbox_points(mins, maxs, offset, spacing):
    mins = mins - offset
    maxs = maxs + offset
    xs = np.arange(mins[0], maxs[0]+spacing, spacing)
    ys = np.arange(mins[1], maxs[1]+spacing, spacing)
    zs = np.arange(mins[2], maxs[2]+spacing, spacing)
    bbox_points = []
    # 6 faces
    for x in xs:
        for y in ys:
            bbox_points.append([x, y, mins[2]])
            bbox_points.append([x, y, maxs[2]])
    for x in xs:
        for z in zs:
            bbox_points.append([x, mins[1], z])
            bbox_points.append([x, maxs[1], z])
    for y in ys:
        for z in zs:
            bbox_points.append([mins[0], y, z])
            bbox_points.append([maxs[0], y, z])
    # Remove duplicates
    return np.unique(np.array(bbox_points), axis=0)

def main():
    parser = argparse.ArgumentParser(description="Add bounding box points to points3D.txt")
    parser.add_argument("--input", required=True, help="Input points3D.txt path")
    parser.add_argument("--output", required=True, help="Output txt path")
    parser.add_argument("--offset", type=float, default=1.0, help="Offset from min/max values")
    parser.add_argument("--spacing", type=float, default=1.0, help="Spacing between bounding box points")
    args = parser.parse_args()

    # Read original file and points
    with open(args.input, 'r') as f:
        original_lines = f.readlines()
    points, header = read_points3d(args.input)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    bbox_points = generate_bbox_points(mins, maxs, args.offset, args.spacing)
    # Find max POINT3D_ID
    last_id = 0
    for line in original_lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        try:
            pid = int(parts[0])
            last_id = max(last_id, pid)
        except Exception:
            continue

    write_points3d(args.output, header, original_lines, bbox_points, last_id+1)
    print(f"Added {len(bbox_points)} bounding box points to {args.output}")

if __name__ == "__main__":
    main()
