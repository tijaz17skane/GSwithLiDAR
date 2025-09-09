import argparse
import laspy
import numpy as np

def downsample_las(input_path: str, output_path: str, factor: int, seed: int = 42):
    # Read LAS file
    las = laspy.read(input_path)
    n_points = len(las.points)

    # Number of points to keep
    n_keep = max(1, n_points // factor)

    # Random indices (no replacement)
    rng = np.random.default_rng(seed)
    keep_indices = rng.choice(n_points, size=n_keep, replace=False)
    keep_indices.sort()  # sorted order keeps spatial structure

    # Extract downsampled points
    downsampled_points = las.points[keep_indices]

    # ✅ Force contiguous copy (fixes BufferError)
    downsampled_points = downsampled_points.copy()

    # Create new LAS with same header
    header = las.header
    las_out = laspy.LasData(header)
    las_out.points = downsampled_points

    # Write output file
    las_out.write(output_path)

    print(f"✅ Downsampled LAS written to: {output_path}")
    print(f"Original points: {n_points} → Downsampled points: {len(downsampled_points)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly downsample a LAS file by keeping 1/factor points.")
    parser.add_argument("--input", required=True, help="Path to input .las file")
    parser.add_argument("--output", required=True, help="Path to output downsampled .las file")
    parser.add_argument("--factor", type=int, required=True, help="Downsampling factor (2=half, 4=quarter, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    downsample_las(args.input, args.output, args.factor, args.seed)
