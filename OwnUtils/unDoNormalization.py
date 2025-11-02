import argparse
import numpy as np

def load_normalization_transform(path):
    mat = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                mat.append([float(x) for x in line.strip().split()])
    mat = np.array(mat)
    assert mat.shape == (4, 4), "Normalization transform must be 4x4"
    translation = mat[:3, 3]
    return translation

def process_images(images_path, translation, output_path):
    with open(images_path, 'r') as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        if line.strip() == "" or line.startswith("#"):
            out_lines.append(line)
            continue
        parts = line.strip().split()
        if len(parts) < 10:
            out_lines.append(line)
            continue
        # TX, TY, TZ are at indices 5, 6, 7
        tx, ty, tz = map(float, parts[5:8])
        tx += translation[0]
        ty += translation[1]
        tz += translation[2]
        parts[5] = f"{tx:.4f}"
        parts[6] = f"{ty:.4f}"
        parts[7] = f"{tz:.4f}"
        out_lines.append(" ".join(parts) + "\n")
    with open(output_path, 'w') as f:
        f.writelines(out_lines)

def process_points3D(points_path, translation, output_path):
    with open(points_path, 'r') as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        if line.strip() == "" or line.startswith("#"):
            out_lines.append(line)
            continue
        parts = line.strip().split()
        if len(parts) < 7:
            out_lines.append(line)
            continue
        # X, Y, Z are at indices 1, 2, 3
        x, y, z = map(float, parts[1:4])
        x += translation[0]
        y += translation[1]
        z += translation[2]
        parts[1] = f"{x:.4f}"
        parts[2] = f"{y:.4f}"
        parts[3] = f"{z:.4f}"
        out_lines.append(" ".join(parts) + "\n")
    with open(output_path, 'w') as f:
        f.writelines(out_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization_transform", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--points3D", required=True)
    parser.add_argument("--output_images", default="images_normalized.txt")
    parser.add_argument("--output_points3D", default="points3D_normalized.txt")
    args = parser.parse_args()

    translation = load_normalization_transform(args.normalization_transform)
    process_images(args.images, translation, args.output_images)
    process_points3D(args.points3D, translation, args.output_points3D)
    print(f"Normalized files written to {args.output_images} and {args.output_points3D}")