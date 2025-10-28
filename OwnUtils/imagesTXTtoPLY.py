import argparse

def read_images_txt(path):
    points = []
    meta = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            x, y, z = map(float, parts[5:8])
            points.append([x, y, z])
            meta.append(parts[:10])
    return points, meta

def write_ply(path, points):
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
        for pt in points:
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 255 255 255\n')

def main():
    parser = argparse.ArgumentParser(description="Convert images.txt camera centers to PLY (white color)")
    parser.add_argument('--input', required=True, help='Input images.txt file')
    parser.add_argument('--output', required=True, help='Output PLY file')
    args = parser.parse_args()

    points, meta = read_images_txt(args.input)
    write_ply(args.output, points)
    print(f"Wrote {len(points)} camera centers to {args.output} (white color)")

if __name__ == "__main__":
    main()
