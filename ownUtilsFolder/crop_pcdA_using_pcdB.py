import argparse
import os
import numpy as np
import open3d as o3d
from pathlib import Path


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


def get_bounding_box(pcd):
    """
    Calculate the axis-aligned bounding box of a point cloud.
    
    Args:
        pcd: open3d.geometry.PointCloud object
        
    Returns:
        tuple: (min_bound, max_bound) as numpy arrays
    """
    points = np.asarray(pcd.points)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound


def crop_point_cloud(pcd, min_bound, max_bound):
    """
    Crop a point cloud using a bounding box.
    
    Args:
        pcd: open3d.geometry.PointCloud object to crop
        min_bound: Minimum coordinates [x, y, z]
        max_bound: Maximum coordinates [x, y, z]
        
    Returns:
        open3d.geometry.PointCloud: Cropped point cloud
    """
    points = np.asarray(pcd.points)
    
    # Create boolean mask for points within bounding box
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    
    # Filter points
    cropped_points = points[mask]
    
    # Create new point cloud with cropped points
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
    
    # Copy colors if they exist
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    
    # Copy normals if they exist
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cropped_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
    
    return cropped_pcd


def main():
    parser = argparse.ArgumentParser(
        description='Crop point cloud A using bounding box from point cloud B'
    )
    parser.add_argument('--plyA', required=True, help='Path to point cloud A (to be cropped)')
    parser.add_argument('--plyB', required=True, help='Path to point cloud B (defines bounding box)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.plyA):
        raise FileNotFoundError(f"Point cloud A not found: {args.plyA}")
    if not os.path.exists(args.plyB):
        raise FileNotFoundError(f"Point cloud B not found: {args.plyB}")
    
    print(f"Reading point cloud A: {args.plyA}")
    pcd_a = read_point_cloud(args.plyA)
    print(f"  - Points in A: {len(pcd_a.points)}")
    
    print(f"Reading point cloud B: {args.plyB}")
    pcd_b = read_point_cloud(args.plyB)
    print(f"  - Points in B: {len(pcd_b.points)}")
    
    # Get bounding box from point cloud B
    min_bound, max_bound = get_bounding_box(pcd_b)
    print(f"\nBounding box from point cloud B:")
    print(f"  - Min: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}]")
    print(f"  - Max: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}]")
    
    # Crop point cloud A
    print("\nCropping point cloud A...")
    cropped_pcd = crop_point_cloud(pcd_a, min_bound, max_bound)
    print(f"  - Points remaining after crop: {len(cropped_pcd.points)}")
    print(f"  - Points removed: {len(pcd_a.points) - len(cropped_pcd.points)}")
    
    # Generate output path in the same directory as point cloud A
    ply_a_path = Path(args.plyA)
    output_path = ply_a_path.parent / f"{ply_a_path.stem}_cropped.ply"
    
    # Save cropped point cloud
    print(f"\nSaving cropped point cloud to: {output_path}")
    o3d.io.write_point_cloud(str(output_path), cropped_pcd)
    print("Done!")


if __name__ == "__main__":
    main()