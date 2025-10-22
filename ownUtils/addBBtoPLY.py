import numpy as np
import open3d as o3d
import argparse
import os
import struct

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove outliers using statistical outlier removal"""
    # Create numpy array from point cloud
    points = np.asarray(pcd.points)
    
    # Calculate mean distance to k nearest neighbors for each point
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mean_distances = []
    
    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], nb_neighbors)
        distances = np.linalg.norm(points[idx[1:]] - points[i], axis=1)
        mean_distances.append(np.mean(distances))
    
    mean_distances = np.array(mean_distances)
    
    # Calculate threshold for outlier removal
    mu = np.mean(mean_distances)
    sigma = np.std(mean_distances)
    threshold = mu + std_ratio * sigma
    
    # Filter points
    mask = mean_distances < threshold
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    
    return filtered_pcd

def read_ply(file_path):
    """Read PLY file and return points"""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def write_ply(points, colors, file_path):
    """Write points and colors to PLY file"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file_path, pcd)

def write_colmap_points3D_txt(points, colors, file_path):
    """Write points in COLMAP's points3D.txt format
    Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    with open(file_path, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        
        for i, (point, color) in enumerate(zip(points, colors)):
            # Convert colors from float [0-1] to int [0-255]
            r, g, b = [int(c * 255) for c in color]
            # Each point is seen in at least 2 views (using dummy image IDs 0 and 1)
            track_info = f"0 {i} 1 {i}"  # Two views with point2D_idx same as point3D_id
            f.write(f'{i} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b} 1.0 {track_info}\n')

def write_colmap_points3D(points, colors, file_path):
    """Write points in COLMAP's points3D.bin format
    Format specification:
    - 8 bytes: Number of points (uint64_t)
    For each point:
        - 8 bytes: Point ID (uint64_t)
        - 24 bytes: XYZ (3 * double)
        - 3 bytes: RGB (3 * uint8_t)
        - 8 bytes: Error (double)
        - 8 bytes: Track length (uint64_t)
        For each track element:
            - 8 bytes: Image ID (uint64_t)
            - 8 bytes: Point2D ID (uint64_t)
    """
    with open(file_path, 'wb') as f:
        # Write number of points
        f.write(struct.pack('Q', len(points)))
        
        # For each point
        for i, (point, color) in enumerate(zip(points, colors)):
            # Point ID, XYZ coordinates, and RGB colors
            f.write(struct.pack('Q', i))  # Point ID
            f.write(struct.pack('3d', *point))  # XYZ
            f.write(struct.pack('3B',  # RGB
                int(color[0] * 255), 
                int(color[1] * 255), 
                int(color[2] * 255)))
            
            # Error (using small default value)
            f.write(struct.pack('d', 1e-4))
            
            # Track length (minimal track with 2 observations)
            track_length = 2
            f.write(struct.pack('Q', track_length))
            
            # Track elements (using placeholder values)
            # We use image IDs 0 and 1 with point2D IDs equal to the point3D ID
            # This ensures the point appears in at least 2 images (minimum for COLMAP)
            for image_id in range(2):
                f.write(struct.pack('Q', image_id))  # Image ID
                f.write(struct.pack('Q', i))  # Point2D ID

def create_bounding_box_points(points, offset, spacing):
    """Create points forming a bounding box with given offset and spacing"""
    # Get min and max coordinates
    min_coords = np.min(points, axis=0) - offset
    max_coords = np.max(points, axis=0) + offset
    
    box_points = []
    
    # Create points for each face of the box
    for dim in range(3):  # For each dimension (x, y, z)
        # Get the other two dimensions
        dims = list(range(3))
        dims.remove(dim)
        
        # For each face perpendicular to the current dimension
        for val in [min_coords[dim], max_coords[dim]]:
            # Calculate points in a grid on this face
            x_range = np.arange(min_coords[dims[0]], max_coords[dims[0]] + spacing, spacing)
            y_range = np.arange(min_coords[dims[1]], max_coords[dims[1]] + spacing, spacing)
            
            # Create grid of points on this face
            for x in x_range:
                for y in y_range:
                    point = np.zeros(3)
                    point[dim] = val  # Fixed dimension
                    point[dims[0]] = x
                    point[dims[1]] = y
                    box_points.append(point)
    
    return np.array(box_points)

def main():
    parser = argparse.ArgumentParser(description='Add bounding box to PLY file')
    parser.add_argument('--input', '-i', required=True, help='Input PLY file path')
    parser.add_argument('--output', '-o', required=True, help='Output PLY file path')
    parser.add_argument('--offset', type=float, default=0.1, help='Offset from min/max coordinates')
    parser.add_argument('--spacing', type=float, default=0.05, help='Spacing between points in bounding box')
    parser.add_argument('--colmap', action='store_true', help='Also output in COLMAP points3D.bin format')
    parser.add_argument('--colmap_output', type=str, help='Path for the COLMAP points3D.bin output file')
    parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers before creating bounding box')
    parser.add_argument('--outlier_std_ratio', type=float, default=2.0, help='Standard deviation ratio for outlier removal (higher=less strict)')
    parser.add_argument('--outlier_nb_neighbors', type=int, default=20, help='Number of neighbors for outlier removal')
    parser.add_argument('--box_color', type=str, default='1,1,0', help='RGB color for bounding box points (default: yellow)')

    args = parser.parse_args()
    
    # Read input PLY file
    pcd = read_ply(args.input)
    
    # Remove outliers if requested
    if args.remove_outliers:
        print("Removing outliers...")
        pcd = remove_outliers(pcd, args.outlier_nb_neighbors, args.outlier_std_ratio)
        print(f"Removed {len(np.asarray(read_ply(args.input).points)) - len(np.asarray(pcd.points))} outlier points")
    
    # Convert to numpy array
    points = np.asarray(pcd.points)
    
    # Create bounding box points
    box_points = create_bounding_box_points(points, args.offset, args.spacing)
    
    # Combine original and box points
    all_points = np.vstack([points, box_points])
    
    # Create colors array
    box_color = np.array([float(x) for x in args.box_color.split(',')])
    original_colors = np.zeros((len(points), 3))  # Black for original points
    box_colors = np.tile(box_color, (len(box_points), 1))
    all_colors = np.vstack([original_colors, box_colors])
    
    # Write combined PLY file
    write_ply(all_points, all_colors, args.output)
    
    # Write COLMAP format if requested
    if args.colmap:
        # Write binary format
        colmap_output = args.colmap_output if args.colmap_output else os.path.join(os.path.dirname(args.output), 'points3D.bin')
        write_colmap_points3D(all_points, all_colors, colmap_output)
        print(f"COLMAP points3D.bin written to: {colmap_output}")
        
        # Write text format
        colmap_txt_output = os.path.splitext(colmap_output)[0] + '.txt'
        write_colmap_points3D_txt(all_points, all_colors, colmap_txt_output)
        print(f"COLMAP points3D.txt written to: {colmap_txt_output}")
    
    print(f"Added bounding box to PLY file. Output written to: {args.output}")
    print(f"Total points: {len(all_points)}")
    print(f"Original points: {len(points)}")
    print(f"Bounding box points: {len(box_points)}")

if __name__ == "__main__":
    main()
