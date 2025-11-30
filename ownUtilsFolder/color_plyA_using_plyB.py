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
        open3d.geometry.PointCloud object
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        raise ValueError(f"No points found in {ply_path}")
    return pcd


def remove_colors(pcd):
    """
    Remove colors from a point cloud.
    
    Args:
        pcd: Point cloud to remove colors from
        
    Returns:
        open3d.geometry.PointCloud: Point cloud without colors
    """
    pcd_no_color = o3d.geometry.PointCloud()
    pcd_no_color.points = pcd.points
    
    # Copy normals if they exist
    if pcd.has_normals():
        pcd_no_color.normals = pcd.normals
    
    return pcd_no_color


def transfer_colors_kdtree(pcd_source, pcd_target):
    """
    Transfer colors from source point cloud to target point cloud
    using nearest neighbor search with KD-tree.
    
    Args:
        pcd_source: Source point cloud with colors
        pcd_target: Target point cloud to receive colors
        
    Returns:
        open3d.geometry.PointCloud: Target point cloud with transferred colors
    """
    # Extract points and colors from source
    source_points = np.asarray(pcd_source.points)
    source_colors = np.asarray(pcd_source.colors)
    
    # Extract points from target
    target_points = np.asarray(pcd_target.points)
    
    print("Building KD-tree from source point cloud...")
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(source_points)
    
    print("Finding nearest neighbors for color transfer...")
    # Find nearest neighbor in source for each target point
    distances, indices = tree.query(target_points, k=1)
    
    # Transfer colors based on nearest neighbors
    transferred_colors = source_colors[indices]
    
    # Create new point cloud with transferred colors
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(target_points)
    pcd_colored.colors = o3d.utility.Vector3dVector(transferred_colors)
    
    # Copy normals if they exist in target
    if pcd_target.has_normals():
        pcd_colored.normals = pcd_target.normals
    
    # Calculate statistics
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    median_distance = np.median(distances)
    
    print(f"\nColor transfer statistics:")
    print(f"  Mean nearest neighbor distance: {mean_distance:.6f}")
    print(f"  Median nearest neighbor distance: {median_distance:.6f}")
    print(f"  Max nearest neighbor distance: {max_distance:.6f}")
    
    return pcd_colored


def main():
    parser = argparse.ArgumentParser(
        description='Transfer colors from one point cloud to another using KD-tree nearest neighbor search'
    )
    parser.add_argument('--plyA', required=True,
                       help='Path to source PLY file (with colors)')
    parser.add_argument('--plyB', required=True,
                       help='Path to target PLY file (to be colored)')
    parser.add_argument('--output', default=None,
                       help='Output path for colored PLY (default: plyB_colored.ply in same directory as plyB)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the result after color transfer')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.plyA).exists():
        raise FileNotFoundError(f"Source PLY not found: {args.plyA}")
    if not Path(args.plyB).exists():
        raise FileNotFoundError(f"Target PLY not found: {args.plyB}")
    
    print("="*70)
    print("COLOR TRANSFER USING KD-TREE")
    print("="*70)
    
    # Load point clouds
    print(f"\nLoading source point cloud (with colors): {args.plyA}")
    pcd_source = read_point_cloud(args.plyA)
    print(f"  Points: {len(pcd_source.points):,}")
    print(f"  Has colors: {pcd_source.has_colors()}")
    
    if not pcd_source.has_colors():
        raise ValueError(f"Source point cloud has no colors: {args.plyA}")
    
    print(f"\nLoading target point cloud: {args.plyB}")
    pcd_target = read_point_cloud(args.plyB)
    print(f"  Points: {len(pcd_target.points):,}")
    print(f"  Has colors (before): {pcd_target.has_colors()}")
    
    # Remove existing colors from target
    if pcd_target.has_colors():
        print("  Removing existing colors from target...")
        pcd_target = remove_colors(pcd_target)
        print(f"  Has colors (after removal): {pcd_target.has_colors()}")
    
    # Transfer colors
    print("\n" + "-"*70)
    pcd_colored = transfer_colors_kdtree(pcd_source, pcd_target)
    print("-"*70)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        plyB_path = Path(args.plyB)
        output_path = plyB_path.parent / f"{plyB_path.stem}_colored.ply"
    
    # Save colored point cloud
    print(f"\nSaving colored point cloud to: {output_path}")
    o3d.io.write_point_cloud(str(output_path), pcd_colored)
    print("Done!")
    
    # Visualize if requested
    if args.visualize:
        print("\nLaunching visualization...")
        print("  - Use mouse to rotate/zoom")
        print("  - Press 'q' or close window to exit")
        o3d.visualization.draw_geometries(
            [pcd_colored],
            window_name="Colored Point Cloud",
            width=1024,
            height=768
        )
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()