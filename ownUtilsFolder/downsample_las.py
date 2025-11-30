import argparse
import laspy
import numpy as np
import os
import glob

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

def downsample_las_from_dir(input_dir: str, output_dir: str, voxel_size: float = None, factor: int = None, seed: int = 42):
    """
    Find a .las file in input_dir, downsample it, and save to output_dir.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the input .las file
    output_dir : str
        Directory where the downsampled .las file will be saved
    voxel_size : float, optional
        Voxel size for voxel grid downsampling
    factor : int, optional
        Random downsampling factor
    seed : int
        Random seed for reproducibility
    """
    # Find .las files in input directory
    las_pattern = os.path.join(input_dir, "*.las")
    las_files = glob.glob(las_pattern)
    
    if not las_files:
        print(f"❌ Error: No .las files found in {input_dir}")
        return
    
    if len(las_files) > 1:
        print(f"⚠️  Warning: Multiple .las files found in {input_dir}:")
        for f in las_files:
            print(f"   - {os.path.basename(f)}")
        print(f"   Using the first one: {os.path.basename(las_files[0])}")
    
    input_path = las_files[0]
    input_filename = os.path.basename(input_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    name, ext = os.path.splitext(input_filename)
    suffix = ""
    if voxel_size is not None:
        suffix += f"_voxel{voxel_size}"
    if factor is not None:
        suffix += f"_factor{factor}"
    if not suffix:
        suffix = "_downsampled"
    
    output_filename = f"{name}{suffix}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📄 Input file: {input_filename}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Output file: {output_filename}")
    
    # Perform downsampling
    las = laspy.read(input_path)
    print(f"📊 Original points: {len(las.points)}")
    
    indices = np.arange(len(las.points))
    
    if voxel_size is not None:
        print(f"🔲 Applying voxel grid downsampling (voxel size: {voxel_size})...")
        indices = voxel_grid_downsample(las.points, voxel_size)
        print(f"   Points after voxel downsampling: {len(indices)}")
    
    if factor is not None:
        print(f"🎲 Applying random downsampling (factor: {factor})...")
        indices = indices[random_downsample(las.points[indices], factor, seed)]
        print(f"   Points after random downsampling: {len(indices)}")
    
    downsampled_points = las.points[indices].copy()
    header = las.header
    las_out = laspy.LasData(header)
    las_out.points = downsampled_points
    las_out.write(output_path)
    
    print(f"✅ Downsampled LAS written to: {output_path}")
    print(f"📊 Final result: {len(las.points)} → {len(downsampled_points)} points")
    
    reduction_percent = (1 - len(downsampled_points) / len(las.points)) * 100
    print(f"📉 Reduction: {reduction_percent:.1f}%")


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
    parser.add_argument("--input_dir", required=True, help="Directory containing input .las file")
    parser.add_argument("--out_dir", required=True, help="Directory where downsampled .las file will be saved")
    parser.add_argument("--voxel_size", type=float, help="Voxel size (in same units as LAS, e.g. meters)")
    parser.add_argument("--factor", type=int, default=64, help="Downsampling factor (2=half, 4=quarter, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.voxel_size is None and args.factor is None:
        print("❌ Error: At least one of --voxel_size or --factor must be specified")
        parser.print_help()
        exit(1)
    
    if args.voxel_size is not None and args.voxel_size <= 0:
        print("❌ Error: --voxel_size must be positive")
        exit(1)
    
    if args.factor is not None and args.factor < 2:
        print("❌ Error: --factor must be >= 2")
        exit(1)
    
    downsample_las_from_dir(args.input_dir, args.out_dir, args.voxel_size, args.factor, args.seed)
