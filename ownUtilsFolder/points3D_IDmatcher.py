import argparse
import numpy as np

def load_points(path):
    """
    Loads a points3D-like file into a dict:
    { id : (x, y, z, remaining_line) }
    """
    points = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue

            parts = line.split()
            pid = int(parts[0])
            x, y, z = map(float, parts[1:4])
            rest = parts[4:]  # RGB, error, tracks (if any)
            points[pid] = (np.array([x, y, z]), rest)

    return points


def create_spatial_index(points):
    """
    Creates a fast lookup by rounding (x,y,z).
    Useful when float precision differs slightly.
    """
    index = {}
    for pid, (xyz, rest) in points.items():
        key = tuple(np.round(xyz, 6))  # tolerance of 1e-6
        index[key] = (pid, xyz, rest)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points3D", required=True)           # reference file
    parser.add_argument("--points3DLidar", required=True)      # lidar file
    parser.add_argument("--points3Dmatched", required=True)    # output file
    args = parser.parse_args()

    # Load both files
    print("Loading points3D...")
    pts3D = load_points(args.points3D)
    print("Loading LiDAR...")
    ptsLidar = load_points(args.points3DLidar)

    # Build spatial index for fast matching
    print("Building spatial index...")
    pts3D_index = create_spatial_index(pts3D)

    matched = []
    missing = []

    for lid, (xyz_lidar, rest_lidar) in ptsLidar.items():
        key = tuple(np.round(xyz_lidar, 6))
        if key in pts3D_index:
            new_id, xyz_ref, rest_ref = pts3D_index[key]
            matched.append((new_id, xyz_ref, rest_lidar))
        else:
            missing.append((lid, xyz_lidar, rest_lidar))

    print(f"Matched: {len(matched)}")
    print(f"Missing: {len(missing)}")

    # Write output
    with open(args.points3Dmatched, "w") as f:
        f.write("# Matched LiDAR points merged with points3D IDs\n")

        for pid, xyz, rest in matched:
            x, y, z = xyz
            rest_str = " ".join(rest)
            f.write(f"{pid} {x:.8f} {y:.8f} {z:.8f} {rest_str}\n")

    print(f"Saved matched file: {args.points3Dmatched}")


if __name__ == "__main__":
    main()
