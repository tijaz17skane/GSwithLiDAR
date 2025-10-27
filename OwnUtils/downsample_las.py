import argparse
import laspy
import numpy as np

def voxel_grid_downsample(points, voxel_size):
    # points: Nx3 numpy array
    coords = np.vstack((points.x, points.y, points.z)).T
    # Compute voxel indices
    voxel_indices = np.floor(coords / voxel_size).astype(np.int64)
    # Find unique voxels and the first point in each voxel
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return unique_indices

def random_downsample(points, factor, seed=42):
    n_points = len(points)
    n_keep = max(1, n_points // factor)
    rng = np.random.default_rng(seed)
    keep_indices = rng.choice(n_points, size=n_keep, replace=False)
    keep_indices.sort()
    return keep_indices

def downsample_las(input_path: str, output_path: str, voxel_size: float = None, factor: int = None, seed: int = 42):
    las = laspy.read(input_path)
    indices = np.arange(len(las.points))
    if voxel_size is not None:
        indices = voxel_grid_downsample(las.points, voxel_size)
    if factor is not None:
        indices = indices[random_downsample(las.points[indices], factor, seed)]
    downsampled_points = las.points[indices].copy()
    header = las.header
    las_out = laspy.LasData(header)
    las_out.points = downsampled_points
    las_out.write(output_path)
    print(f"✅ Downsampled LAS written to: {output_path}")
    print(f"Original points: {len(las.points)} → Downsampled points: {len(downsampled_points)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample a LAS file by voxel grid and/or random factor.")
    parser.add_argument("--input", required=True, help="Path to input .las file")
    parser.add_argument("--output", required=True, help="Path to output downsampled .las file")
    parser.add_argument("--voxel_size", type=float, help="Voxel size (in same units as LAS, e.g. meters)")
    parser.add_argument("--factor", type=int, help="Downsampling factor (2=half, 4=quarter, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    downsample_las(args.input, args.output, args.voxel_size, args.factor, args.seed)
