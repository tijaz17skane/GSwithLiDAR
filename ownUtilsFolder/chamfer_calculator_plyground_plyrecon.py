import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree


def read_point_cloud(ply_path):
    """
    Read a point cloud from a PLY file.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        numpy array of points (N x 3)
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        raise ValueError(f"No points found in {ply_path}")
    return np.asarray(pcd.points)


def one_sided_chamfer_distance(points_src, points_tgt):
    """
    Calculate one-sided Chamfer distance from source to target.
    For each point in source, find the nearest point in target.
    
    Args:
        points_src: Source point cloud (N x 3)
        points_tgt: Target point cloud (M x 3)
        
    Returns:
        float: Mean squared distance from source to target
        numpy array: Individual squared distances for each source point
    """
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(points_tgt)
    
    # Find nearest neighbor distances for each source point
    distances, _ = tree.query(points_src, k=1)
    
    # Square the distances (Chamfer distance uses squared Euclidean distance)
    squared_distances = distances ** 2
    
    # Return mean and individual distances
    return np.mean(squared_distances), squared_distances


def symmetric_chamfer_distance(points_a, points_b):
    """
    Calculate symmetric (bidirectional) Chamfer distance.
    
    Args:
        points_a: First point cloud (N x 3)
        points_b: Second point cloud (M x 3)
        
    Returns:
        float: Symmetric Chamfer distance
        float: One-sided distance A -> B
        float: One-sided distance B -> A
    """
    cd_a_to_b, _ = one_sided_chamfer_distance(points_a, points_b)
    cd_b_to_a, _ = one_sided_chamfer_distance(points_b, points_a)
    
    # Symmetric Chamfer distance is the sum of both directions
    cd_symmetric = cd_a_to_b + cd_b_to_a
    
    return cd_symmetric, cd_a_to_b, cd_b_to_a


def calculate_statistics(distances):
    """
    Calculate statistical measures for distance array.
    
    Args:
        distances: Array of distances
        
    Returns:
        dict: Statistics including mean, median, std, min, max, percentiles
    """
    return {
        'mean': np.mean(distances),
        'median': np.median(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'p95': np.percentile(distances, 95),
        'p99': np.percentile(distances, 99),
        'rmse': np.sqrt(np.mean(distances))
    }


def print_results(cd_symmetric, cd_ground_to_const, cd_const_to_ground, 
                 stats_g2c, stats_c2g, num_ground, num_const):
    """
    Print formatted results.
    """
    print("\n" + "="*70)
    print("CHAMFER DISTANCE RESULTS")
    print("="*70)
    
    print(f"\nPoint Cloud Information:")
    print(f"  Ground truth points: {num_ground:,}")
    print(f"  Reconstruction points: {num_const:,}")
    
    print(f"\n{'One-Sided Chamfer Distances (Mean Squared Distance)':^70}")
    print("-"*70)
    print(f"  Ground → Reconstruction: {cd_ground_to_const:.6e}")
    print(f"  Reconstruction → Ground: {cd_const_to_ground:.6e}")
    
    print(f"\n{'Symmetric Chamfer Distance':^70}")
    print("-"*70)
    print(f"  Symmetric CD: {cd_symmetric:.6e}")
    print(f"  (Sum of both directions)")
    
    print(f"\n{'Detailed Statistics: Ground → Reconstruction':^70}")
    print("-"*70)
    print(f"  Mean Squared Distance:  {stats_g2c['mean']:.6e}")
    print(f"  RMSE:                   {stats_g2c['rmse']:.6e}")
    print(f"  Median:                 {stats_g2c['median']:.6e}")
    print(f"  Std Dev:                {stats_g2c['std']:.6e}")
    print(f"  Min:                    {stats_g2c['min']:.6e}")
    print(f"  Max:                    {stats_g2c['max']:.6e}")
    print(f"  95th Percentile:        {stats_g2c['p95']:.6e}")
    print(f"  99th Percentile:        {stats_g2c['p99']:.6e}")
    
    print(f"\n{'Detailed Statistics: Reconstruction → Ground':^70}")
    print("-"*70)
    print(f"  Mean Squared Distance:  {stats_c2g['mean']:.6e}")
    print(f"  RMSE:                   {stats_c2g['rmse']:.6e}")
    print(f"  Median:                 {stats_c2g['median']:.6e}")
    print(f"  Std Dev:                {stats_c2g['std']:.6e}")
    print(f"  Min:                    {stats_c2g['min']:.6e}")
    print(f"  Max:                    {stats_c2g['max']:.6e}")
    print(f"  95th Percentile:        {stats_c2g['p95']:.6e}")
    print(f"  99th Percentile:        {stats_c2g['p99']:.6e}")
    
    print("\n" + "="*70)


def save_results_to_file(output_path, cd_symmetric, cd_ground_to_const, 
                        cd_const_to_ground, stats_g2c, stats_c2g, 
                        num_ground, num_const):
    """
    Save results to a text file.
    """
    with open(output_path, 'w') as f:
        f.write("CHAMFER DISTANCE RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Point Cloud Information:\n")
        f.write(f"  Ground truth points: {num_ground:,}\n")
        f.write(f"  Reconstruction points: {num_const:,}\n\n")
        
        f.write(f"One-Sided Chamfer Distances (Mean Squared Distance):\n")
        f.write(f"  Ground → Reconstruction: {cd_ground_to_const:.6e}\n")
        f.write(f"  Reconstruction → Ground: {cd_const_to_ground:.6e}\n\n")
        
        f.write(f"Symmetric Chamfer Distance:\n")
        f.write(f"  Symmetric CD: {cd_symmetric:.6e}\n\n")
        
        f.write(f"Detailed Statistics (Ground → Reconstruction):\n")
        for key, value in stats_g2c.items():
            f.write(f"  {key}: {value:.6e}\n")
        
        f.write(f"\nDetailed Statistics (Reconstruction → Ground):\n")
        for key, value in stats_c2g.items():
            f.write(f"  {key}: {value:.6e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate one-sided and symmetric Chamfer distances between two point clouds'
    )
    parser.add_argument('--ply_ground', required=True, 
                       help='Path to ground truth point cloud (PLY format)')
    parser.add_argument('--ply_const', required=True, 
                       help='Path to reconstructed/compared point cloud (PLY format)')
    parser.add_argument('--output', default=None,
                       help='Optional: Path to save results as text file')
    parser.add_argument('--subsample', type=int, default=None,
                       help='Optional: Subsample point clouds to N points for faster computation')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.ply_ground).exists():
        raise FileNotFoundError(f"Ground truth PLY not found: {args.ply_ground}")
    if not Path(args.ply_const).exists():
        raise FileNotFoundError(f"Reconstruction PLY not found: {args.ply_const}")
    
    print("Loading point clouds...")
    print(f"  Ground truth: {args.ply_ground}")
    points_ground = read_point_cloud(args.ply_ground)
    print(f"    Loaded {len(points_ground):,} points")
    
    print(f"  Reconstruction: {args.ply_const}")
    points_const = read_point_cloud(args.ply_const)
    print(f"    Loaded {len(points_const):,} points")
    
    # Subsample if requested
    if args.subsample is not None:
        print(f"\nSubsampling to {args.subsample:,} points...")
        if len(points_ground) > args.subsample:
            indices = np.random.choice(len(points_ground), args.subsample, replace=False)
            points_ground = points_ground[indices]
            print(f"  Ground truth subsampled to {len(points_ground):,} points")
        
        if len(points_const) > args.subsample:
            indices = np.random.choice(len(points_const), args.subsample, replace=False)
            points_const = points_const[indices]
            print(f"  Reconstruction subsampled to {len(points_const):,} points")
    
    print("\nCalculating Chamfer distances...")
    
    # Calculate one-sided distances with detailed statistics
    print("  Computing Ground → Reconstruction...")
    cd_ground_to_const, dists_g2c = one_sided_chamfer_distance(points_ground, points_const)
    stats_g2c = calculate_statistics(dists_g2c)
    
    print("  Computing Reconstruction → Ground...")
    cd_const_to_ground, dists_c2g = one_sided_chamfer_distance(points_const, points_ground)
    stats_c2g = calculate_statistics(dists_c2g)
    
    # Calculate symmetric distance
    cd_symmetric = cd_ground_to_const + cd_const_to_ground
    
    # Print results
    print_results(cd_symmetric, cd_ground_to_const, cd_const_to_ground,
                 stats_g2c, stats_c2g, len(points_ground), len(points_const))
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        save_results_to_file(output_path, cd_symmetric, cd_ground_to_const,
                           cd_const_to_ground, stats_g2c, stats_c2g,
                           len(points_ground), len(points_const))
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()