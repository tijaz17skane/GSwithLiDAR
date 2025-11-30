import open3d as o3d
import numpy as np
import argparse
from scipy.spatial import cKDTree
from tqdm import tqdm

def transfer_colors_nearest_neighbor(source_pcd, target_pcd):
    """
    Transfer colors from source point cloud to target point cloud
    using nearest neighbor matching.
    """
    print("\nTransferring colors from Point Cloud A to Point Cloud B...")
    
    # Get points and colors
    source_points = np.asarray(source_pcd.points)
    source_colors = np.asarray(source_pcd.colors)
    target_points = np.asarray(target_pcd.points)
    
    # Build KDTree for efficient nearest neighbor search
    print("Building KDTree...")
    tree = cKDTree(source_points)
    
    # Find nearest neighbors
    print("Finding nearest neighbors...")
    distances, indices = tree.query(target_points, k=1, workers=-1)
    
    # Transfer colors
    print("Transferring colors...")
    transferred_colors = source_colors[indices]
    
    # Create new point cloud with transferred colors
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = target_pcd.points
    result_pcd.colors = o3d.utility.Vector3dVector(transferred_colors)
    
    # Copy normals if they exist
    if target_pcd.has_normals():
        result_pcd.normals = target_pcd.normals
    
    print(f"Average nearest neighbor distance: {np.mean(distances):.6f}")
    print(f"Max nearest neighbor distance: {np.max(distances):.6f}")
    
    return result_pcd, distances

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transfer colors from one point cloud to another')
    parser.add_argument('--plyA', type=str, 
                        default='/mnt/data/tijaz/data2/coloringFromScratch/3dgs_to_pc.ply',
                        help='Path to the source PLY file (with colors)')
    parser.add_argument('--plyB', type=str, 
                        default='/mnt/data/tijaz/data2/coloringFromScratch/points3D.ply',
                        help='Path to the target PLY file')
    parser.add_argument('--output', type=str,
                        default='/mnt/data/tijaz/data2/coloringFromScratch/points3D_colored.ply',
                        help='Path to save the output PLY file (default: pointcloud_B_colored.ply)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the results')
    
    args = parser.parse_args()
    
    # Load the two point cloud files
    print(f"Loading {args.plyA}...")
    plyA = o3d.io.read_point_cloud(args.plyA)
    
    print(f"Loading {args.plyB}...")
    plyB = o3d.io.read_point_cloud(args.plyB)
    
    # Print basic information about the point clouds
    print("\nPoint Cloud A (Source):")
    print(f"  Number of points: {len(plyA.points)}")
    print(f"  Has colors: {plyA.has_colors()}")
    print(f"  Has normals: {plyA.has_normals()}")
    
    print("\nPoint Cloud B (Target):")
    print(f"  Number of points: {len(plyB.points)}")
    print(f"  Has colors: {plyB.has_colors()}")
    print(f"  Has normals: {plyB.has_normals()}")
    
    # Access point coordinates as numpy arrays
    points_A = np.asarray(plyA.points)
    points_B = np.asarray(plyB.points)
    
    print(f"\nPoint Cloud A shape: {points_A.shape}")
    print(f"Point Cloud B shape: {points_B.shape}")
    
    # Check if source has colors
    if not plyA.has_colors():
        print("\nError: Point Cloud A does not have colors to transfer!")
        return
    
    colors_A = np.asarray(plyA.colors)
    print(f"Colors A shape: {colors_A.shape}")
    
    if plyB.has_colors():
        colors_B = np.asarray(plyB.colors)
        print(f"Colors B shape (original): {colors_B.shape}")
    
    # Transfer colors
    result_pcd, distances = transfer_colors_nearest_neighbor(plyA, plyB)
    
    # Save the result
    print(f"\nSaving result to {args.output}...")
    o3d.io.write_point_cloud(args.output, result_pcd)
    print("Done!")
    
    # Visualize if requested
    if args.visualize:
        print("\nVisualizing results...")
        print("Original Point Cloud B:")
        o3d.visualization.draw_geometries([plyB], window_name="Original Point Cloud B")
        
        print("Point Cloud B with transferred colors:")
        o3d.visualization.draw_geometries([result_pcd], window_name="Point Cloud B with Transferred Colors")
        
        print("Comparison (A: source, B: result):")
        # Color source cloud slightly differently for comparison
        plyA_vis = o3d.geometry.PointCloud(plyA)
        o3d.visualization.draw_geometries([plyA_vis, result_pcd], 
                                         window_name="Source (A) and Result (B)")

if __name__ == "__main__":
    main()