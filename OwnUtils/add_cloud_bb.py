import argparse
import laspy
import numpy as np


def compute_bounding_box(points: np.ndarray, offset: float):
    """
    Compute bounding box with offset.
    Returns min and max for x,y,z.
    """
    min_vals = points.min(axis=0) - offset
    max_vals = points.max(axis=0) + offset
    return min_vals, max_vals


def create_filled_bbox(min_vals, max_vals, spacing=1.0):
    """
    Create points filling the bounding box faces (not just edges).
    The result is a hollow cube with filled sides.

    Args:
        min_vals (array): [min_x, min_y, min_z]
        max_vals (array): [max_x, max_y, max_z]
        spacing (float): Spacing between points on surfaces.

    Returns:
        np.ndarray: Array of shape (N, 3) with bounding box surface points.
    """
    min_x, min_y, min_z = min_vals
    max_x, max_y, max_z = max_vals

    xs = np.arange(min_x, max_x + spacing, spacing)
    ys = np.arange(min_y, max_y + spacing, spacing)
    zs = np.arange(min_z, max_z + spacing, spacing)

    face_points = []

    # XY faces (bottom and top)
    X, Y = np.meshgrid(xs, ys)
    face_points.append(np.c_[X.ravel(), Y.ravel(), np.full(X.size, min_z)])
    face_points.append(np.c_[X.ravel(), Y.ravel(), np.full(X.size, max_z)])

    # XZ faces (front and back)
    X, Z = np.meshgrid(xs, zs)
    face_points.append(np.c_[X.ravel(), np.full(X.size, min_y), Z.ravel()])
    face_points.append(np.c_[X.ravel(), np.full(X.size, max_y), Z.ravel()])

    # YZ faces (left and right)
    Y, Z = np.meshgrid(ys, zs)
    face_points.append(np.c_[np.full(Y.size, min_x), Y.ravel(), Z.ravel()])
    face_points.append(np.c_[np.full(Y.size, max_x), Y.ravel(), Z.ravel()])

    return np.vstack(face_points)


def main():
    parser = argparse.ArgumentParser(description="Add a bounding box surface point cloud to a LAS scene")
    parser.add_argument("--input_las", type=str, required=True, help="Path to input .las file")
    parser.add_argument("--output_las", type=str, required=True, help="Path to output .las file with bounding box")
    parser.add_argument("--offset", type=float, default=0.0, help="Bounding box expansion offset")
    parser.add_argument("--spacing", type=float, default=0.5, help="Spacing between points on bounding box surfaces")
    args = parser.parse_args()

    # Load LAS file
    las = laspy.read(args.input_las)
    points = np.vstack((las.x, las.y, las.z)).T

    # Compute bounding box
    min_vals, max_vals = compute_bounding_box(points, args.offset)

    # Generate bounding box surface points
    bbox_points = create_filled_bbox(min_vals, max_vals, spacing=args.spacing)

    # Combine original + bbox points
    all_points = np.vstack((points, bbox_points))

    # Create new LAS file
    header = las.header.copy()
    new_las = laspy.LasData(header)

    new_las.x = all_points[:, 0]
    new_las.y = all_points[:, 1]
    new_las.z = all_points[:, 2]

    new_las.write(args.output_las)

    print(f"✅ Saved LAS with bounding box surfaces to {args.output_las}")


if __name__ == "__main__":
    main()
