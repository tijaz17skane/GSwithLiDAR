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

def read_colmap_bin(bin_file):
    """Read COLMAP points3D.bin file and return points and colors."""
    points = []
    colors = []
    with open(bin_file, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]  # Read number of points
        for _ in range(num_points):
            point_id = struct.unpack('Q', f.read(8))[0]  # Point ID
            x, y, z = struct.unpack('3d', f.read(24))  # XYZ coordinates
            r, g, b = struct.unpack('3B', f.read(3))  # RGB colors
            f.read(1)  # Padding byte
            error = struct.unpack('d', f.read(8))[0]  # Error
            track_length = struct.unpack('Q', f.read(8))[0]  # Track length
            for _ in range(track_length):
                f.read(16)  # Skip track elements
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(points), np.array(colors)

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

def read_colmap_txt(txt_file):
    """Read COLMAP points3D.txt file and return points and colors."""
    points = []
    colors = []
    with open(txt_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip comments
            parts = line.strip().split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(points), np.array(colors)

def write_colmap_txt(points, colors, txt_file):
    """Write points and colors to COLMAP points3D.txt file."""
    with open(txt_file, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        for i, (point, color) in enumerate(zip(points, colors)):
            r, g, b = [int(c * 255) for c in color]
            track_info = f"0 {i} 1 {i}"  # Dummy track info
            f.write(f'{i} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b} 1.0 {track_info}\n')

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
    parser = argparse.ArgumentParser(description='Add bounding box to COLMAP points3D.bin or points3D.txt file')
    parser.add_argument('--input', '-i', required=True, help='Input COLMAP points3D.bin or points3D.txt file path')
    parser.add_argument('--output', '-o', required=True, help='Output COLMAP points3D.bin or points3D.txt file path')
    parser.add_argument('--offset', type=float, default=0.1, help='Offset from min/max coordinates')
    parser.add_argument('--spacing', type=float, default=0.05, help='Spacing between points in bounding box')
    parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers before creating bounding box')
    parser.add_argument('--outlier_std_ratio', type=float, default=2.0, help='Standard deviation ratio for outlier removal (higher=less strict)')
    parser.add_argument('--outlier_nb_neighbors', type=int, default=20, help='Number of neighbors for outlier removal')
    parser.add_argument('--box_color', type=str, default='1,1,0', help='RGB color for bounding box points (default: yellow)')

    args = parser.parse_args()

    # Determine file type from extension
    file_ext = os.path.splitext(args.input)[1].lower()
    if file_ext == '.bin':
        # Read input COLMAP .bin file
        points, colors = read_colmap_bin(args.input)

        # Remove outliers if requested
        if args.remove_outliers:
            print("Removing outliers...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            filtered_pcd = remove_outliers(pcd, args.outlier_nb_neighbors, args.outlier_std_ratio)
            points = np.asarray(filtered_pcd.points)
            colors = colors[:len(points)]  # Adjust colors to match filtered points
            print(f"Removed {len(colors) - len(points)} outlier points")

        # Create bounding box points
        box_points = create_bounding_box_points(points, args.offset, args.spacing)

        # Combine original and box points
        all_points = np.vstack([points, box_points])

        # Create colors array
        box_color = np.array([float(x) for x in args.box_color.split(',')])
        box_colors = np.tile(box_color, (len(box_points), 1))
        all_colors = np.vstack([colors, box_colors])

        # Write combined COLMAP .bin file
        write_colmap_points3D(all_points, all_colors, args.output)
        print(f"Added bounding box to COLMAP points3D.bin file. Output written to: {args.output}")
        print(f"Total points: {len(all_points)}")
        print(f"Original points: {len(points)}")
        print(f"Bounding box points: {len(box_points)}")
    
    elif file_ext == '.txt':
        # Read input COLMAP .txt file
        points, colors = read_colmap_txt(args.input)

        # Remove outliers if requested
        if args.remove_outliers:
            print("Removing outliers...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            filtered_pcd = remove_outliers(pcd, args.outlier_nb_neighbors, args.outlier_std_ratio)
            points = np.asarray(filtered_pcd.points)
            colors = colors[:len(points)]  # Adjust colors to match filtered points
            print(f"Removed {len(colors) - len(points)} outlier points")

        # Create bounding box points
        box_points = create_bounding_box_points(points, args.offset, args.spacing)

        # Combine original and box points
        all_points = np.vstack([points, box_points])

        # Create colors array
        box_color = np.array([float(x) for x in args.box_color.split(',')])
        box_colors = np.tile(box_color, (len(box_points), 1))
        all_colors = np.vstack([colors, box_colors])

        # Write combined COLMAP .txt file
        write_colmap_txt(all_points, all_colors, args.output)
        print(f"Added bounding box to COLMAP points3D.txt file. Output written to: {args.output}")
        print(f"Total points: {len(all_points)}")
        print(f"Original points: {len(points)}")
        print(f"Bounding box points: {len(box_points)}")
    else:
        print("Unsupported file format. Please provide a .bin or .txt file.")

if __name__ == "__main__":
    main()
