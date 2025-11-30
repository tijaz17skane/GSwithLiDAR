#!/usr/bin/env python3
"""
Point Cloud Merger for COLMAP and LiDAR data

This script merges two point clouds:
1. COLMAP point cloud (points3DColmap.txt)
2. LiDAR point cloud (points3DLidarDS64Mapped2colmap.txt)

Outputs:
1. points3Dcombined.txt - All points from both inputs
2. points3DcombinedCropped.txt - All LiDAR points + COLMAP points within LiDAR bounding box
"""

import numpy as np
import os
import argparse


def parse_colmap_points3D(filepath):
    """
    Parse COLMAP points3D.txt file.
    
    Format:
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    
    Returns:
    --------
    points : np.ndarray (N, 3)
        XYZ coordinates
    colors : np.ndarray (N, 3)
        RGB colors
    errors : np.ndarray (N,)
        Reconstruction errors
    point_ids : list
        POINT3D_ID for each point
    tracks : list
        Track information for each point
    header_lines : list
        Comment/header lines from file
    """
    points = []
    colors = []
    errors = []
    point_ids = []
    tracks = []
    header_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            
            # Store header/comment lines
            if line_stripped.startswith('#') or not line_stripped:
                header_lines.append(line)
                continue
            
            parts = line_stripped.split()
            if len(parts) < 8:
                continue
            
            # Parse: POINT3D_ID X Y Z R G B ERROR [TRACK...]
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            error = float(parts[7])
            
            # Track data: remaining elements (IMAGE_ID, POINT2D_IDX pairs)
            track = parts[8:] if len(parts) > 8 else []
            
            point_ids.append(point_id)
            points.append([x, y, z])
            colors.append([r, g, b])
            errors.append(error)
            tracks.append(track)
    
    return (np.array(points), np.array(colors), np.array(errors), 
            point_ids, tracks, header_lines)


def compute_bounding_box(points):
    """
    Compute axis-aligned bounding box for a point cloud.
    
    Returns:
    --------
    min_bound : np.ndarray (3,)
        Minimum coordinates (x_min, y_min, z_min)
    max_bound : np.ndarray (3,)
        Maximum coordinates (x_max, y_max, z_max)
    """
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound


def filter_points_by_bbox(points, colors, errors, point_ids, tracks, min_bound, max_bound):
    """
    Filter points to keep only those within the given bounding box.
    
    Returns filtered versions of all input arrays/lists.
    """
    # Check which points are inside the bounding box
    inside_x = (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0])
    inside_y = (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1])
    inside_z = (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
    inside_mask = inside_x & inside_y & inside_z
    
    # Filter all data
    filtered_points = points[inside_mask]
    filtered_colors = colors[inside_mask]
    filtered_errors = errors[inside_mask]
    filtered_point_ids = [point_ids[i] for i in range(len(point_ids)) if inside_mask[i]]
    filtered_tracks = [tracks[i] for i in range(len(tracks)) if inside_mask[i]]
    
    return filtered_points, filtered_colors, filtered_errors, filtered_point_ids, filtered_tracks


def write_colmap_points3D(filepath, points, colors, errors, point_ids, tracks, header_lines=None):
    """
    Write points to COLMAP points3D.txt format.
    """
    with open(filepath, 'w') as f:
        # Write header
        if header_lines:
            for line in header_lines:
                f.write(line)
        else:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(points)}\n")
        
        # Write point data
        for i in range(len(points)):
            point_id = point_ids[i]
            x, y, z = points[i]
            r, g, b = colors[i]
            error = errors[i]
            track = tracks[i] if i < len(tracks) else []
            
            # Format: POINT3D_ID X Y Z R G B ERROR [TRACK...]
            track_str = ' '.join(map(str, track)) if track else ''
            f.write(f"{point_id} {x:.8f} {y:.8f} {z:.8f} {r} {g} {b} {error:.8f}")
            if track_str:
                f.write(f" {track_str}")
            f.write("\n")


def write_ply(filepath, points, colors):
    """
    Write points to PLY format for visualization.
    
    Parameters:
    -----------
    filepath : str
        Output PLY file path
    points : np.ndarray (N, 3)
        XYZ coordinates
    colors : np.ndarray (N, 3)
        RGB colors (0-255)
    """
    with open(filepath, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def merge_point_clouds(points1, colors1, errors1, ids1, tracks1,
                        points2, colors2, errors2, ids2, tracks2):
    """
    Merge two point clouds by concatenating all data.
    Keep COLMAP (points1) IDs unchanged, renumber LiDAR (points2) IDs to start after max COLMAP ID.
    """
    # Keep COLMAP IDs unchanged
    # Renumber LiDAR IDs to start after the maximum COLMAP ID
    if len(ids1) > 0:
        max_colmap_id = max(ids1)
        # Renumber LiDAR points starting from max_colmap_id + 1
        lidar_ids = list(range(max_colmap_id + 1, max_colmap_id + 1 + len(points2)))
    else:
        # If no COLMAP points, just use sequential IDs for LiDAR
        lidar_ids = list(range(1, len(points2) + 1))
    
    merged_ids = list(ids1) + lidar_ids
    merged_points = np.vstack([points1, points2])
    merged_colors = np.vstack([colors1, colors2])
    merged_errors = np.concatenate([errors1, errors2])
    merged_tracks = tracks1 + tracks2
    
    return merged_points, merged_colors, merged_errors, merged_ids, merged_tracks


def main():
    parser = argparse.ArgumentParser(
        description='Merge COLMAP and LiDAR point clouds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    OUTPUTS (named):
      colmap_as_is.txt / colmap_as_is.ply
          - COLMAP points exported unchanged
      lidar_as_is.txt / lidar_as_is.ply
          - LiDAR points exported unchanged
      colmap_bounded_by_lidar.txt / colmap_bounded_by_lidar.ply
          - COLMAP points cropped to the LiDAR bounding box (no LiDAR points)
      combined_full.txt / combined_full.ply
          - Combined COLMAP + LiDAR point cloud
      combined_cropped_bounded_by_lidar.txt / combined_cropped_bounded_by_lidar.ply
          - Combined COLMAP+LiDAR, but COLMAP points cropped by LiDAR bounding box
            """
    )
    
    parser.add_argument('--colmap', type=str, 
                       default='/mnt/data/tijaz/data/Attempt3/colmapCompleteOutput/points3DColmap.txt',
                       help='Path to COLMAP points3D.txt file')
    parser.add_argument('--lidar', type=str,
                       default='/mnt/data/tijaz/data/Attempt3/colmapCompleteOutput/points3DLidarDS64Mapped2colmap.txt',
                       help='Path to LiDAR points3D.txt file')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/data/tijaz/data/Attempt3/colmapCompleteOutput',
                       help='Output directory for merged point clouds')
    parser.add_argument('--output_bounded_colmap', action='store_true',
                       help='Also output COLMAP points bounded by LiDAR bbox (no LiDAR points)')
    parser.add_argument('--bbox_margin', type=float, default=0.0,
                       help='Margin to expand (+) or shrink (-) the LiDAR bounding box (in meters). Default: 0.0')
    # Flags to selectively export specific outputs (when any is set, only the requested outputs are written)
    parser.add_argument('--out_colmap', action='store_true', help='Export COLMAP points only (TXT + PLY). File: colmap_as_is.*')
    parser.add_argument('--out_lidar', action='store_true', help='Export LiDAR points only (TXT + PLY). File: lidar_as_is.*')
    parser.add_argument('--out_colmap_bounded', action='store_true', help='Export COLMAP points bounded by LiDAR bbox (TXT + PLY). File: colmap_bounded_by_lidar.*')
    parser.add_argument('--out_combined', action='store_true', help='Export combined COLMAP+LiDAR point cloud (TXT + PLY). File: combined_full.*')
    parser.add_argument('--out_combined_bounded', action='store_true', help='Export combined point cloud but with COLMAP bounded by LiDAR bbox (TXT + PLY). File: combined_cropped_bounded_by_lidar.*')
    
    args = parser.parse_args()
    
    print("="*70)
    print("POINT CLOUD MERGER (COLMAP + LiDAR)")
    print("="*70)
    
    # Load COLMAP point cloud
    print(f"\nLoading COLMAP point cloud from: {args.colmap}")
    if not os.path.exists(args.colmap):
        print(f"ERROR: File not found: {args.colmap}")
        return
    
    colmap_data = parse_colmap_points3D(args.colmap)
    points_colmap, colors_colmap, errors_colmap, ids_colmap, tracks_colmap, header_colmap = colmap_data
    print(f"  ✓ Loaded {len(points_colmap)} points from COLMAP")
    
    # Load LiDAR point cloud
    print(f"\nLoading LiDAR point cloud from: {args.lidar}")
    if not os.path.exists(args.lidar):
        print(f"ERROR: File not found: {args.lidar}")
        return
    
    lidar_data = parse_colmap_points3D(args.lidar)
    points_lidar, colors_lidar, errors_lidar, ids_lidar, tracks_lidar, header_lidar = lidar_data
    print(f"  ✓ Loaded {len(points_lidar)} points from LiDAR")
    
    # Compute LiDAR bounding box
    print("\nComputing LiDAR bounding box...")
    min_bound, max_bound = compute_bounding_box(points_lidar)
    
    # Apply margin to bounding box
    if args.bbox_margin != 0.0:
        print(f"  Applying bounding box margin: {args.bbox_margin:+.4f} meters")
        min_bound = min_bound - args.bbox_margin
        max_bound = max_bound + args.bbox_margin
        print(f"  (Positive margin expands bbox, negative margin shrinks it)")
    
    print(f"  Min bounds: [{min_bound[0]:.4f}, {min_bound[1]:.4f}, {min_bound[2]:.4f}]")
    print(f"  Max bounds: [{max_bound[0]:.4f}, {max_bound[1]:.4f}, {max_bound[2]:.4f}]")
    bbox_size = max_bound - min_bound
    print(f"  Bounding box size: [{bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}]")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Decide whether user requested specific outputs. If any of the selective flags
    # are set, only write those. Otherwise keep the original behavior (write combined
    # and cropped combined, and optional bounded colmap when --output_bounded_colmap is set).
    any_out_flags = (
        args.out_colmap or args.out_lidar or args.out_colmap_bounded
        or args.out_combined or args.out_combined_bounded
    )

    # Prepare filtered and merged clouds (these are cheap operations and reused below)
    filtered_colmap = filter_points_by_bbox(
        points_colmap, colors_colmap, errors_colmap, ids_colmap, tracks_colmap,
        min_bound, max_bound
    )
    points_colmap_filtered, colors_colmap_filtered, errors_colmap_filtered, ids_colmap_filtered, tracks_colmap_filtered = filtered_colmap

    combined_points, combined_colors, combined_errors, combined_ids, combined_tracks = merge_point_clouds(
        points_colmap, colors_colmap, errors_colmap, ids_colmap, tracks_colmap,
        points_lidar, colors_lidar, errors_lidar, ids_lidar, tracks_lidar
    )

    cropped_points, cropped_colors, cropped_errors, cropped_ids, cropped_tracks = merge_point_clouds(
        points_colmap_filtered, colors_colmap_filtered, errors_colmap_filtered,
        ids_colmap_filtered, tracks_colmap_filtered,
        points_lidar, colors_lidar, errors_lidar, ids_lidar, tracks_lidar
    )

    written = []

    if any_out_flags:
        # Only write requested outputs
        if args.out_colmap:
            out_colmap_txt = os.path.join(args.output_dir, 'colmap_as_is.txt')
            out_colmap_ply = os.path.join(args.output_dir, 'colmap_as_is.ply')
            print(f"Writing COLMAP-only TXT to: {out_colmap_txt}")
            write_colmap_points3D(out_colmap_txt, points_colmap, colors_colmap, errors_colmap, ids_colmap, tracks_colmap, header_colmap)
            print(f"Writing COLMAP-only PLY to: {out_colmap_ply}")
            write_ply(out_colmap_ply, points_colmap, colors_colmap)
            written.append((out_colmap_txt, len(points_colmap)))
            written.append((out_colmap_ply, len(points_colmap)))

        if args.out_lidar:
            out_lidar_txt = os.path.join(args.output_dir, 'lidar_as_is.txt')
            out_lidar_ply = os.path.join(args.output_dir, 'lidar_as_is.ply')
            print(f"Writing LiDAR-only TXT to: {out_lidar_txt}")
            write_colmap_points3D(out_lidar_txt, points_lidar, colors_lidar, errors_lidar, ids_lidar, tracks_lidar, header_lidar)
            print(f"Writing LiDAR-only PLY to: {out_lidar_ply}")
            write_ply(out_lidar_ply, points_lidar, colors_lidar)
            written.append((out_lidar_txt, len(points_lidar)))
            written.append((out_lidar_ply, len(points_lidar)))

        if args.out_colmap_bounded or args.output_bounded_colmap:
            out_bounded_txt = os.path.join(args.output_dir, 'colmap_bounded_by_lidar.txt')
            out_bounded_ply = os.path.join(args.output_dir, 'colmap_bounded_by_lidar.ply')
            print(f"Writing bounded COLMAP TXT to: {out_bounded_txt}")
            write_colmap_points3D(out_bounded_txt, points_colmap_filtered, colors_colmap_filtered, errors_colmap_filtered, ids_colmap_filtered, tracks_colmap_filtered, header_colmap)
            print(f"Writing bounded COLMAP PLY to: {out_bounded_ply}")
            write_ply(out_bounded_ply, points_colmap_filtered, colors_colmap_filtered)
            written.append((out_bounded_txt, len(points_colmap_filtered)))
            written.append((out_bounded_ply, len(points_colmap_filtered)))

        if args.out_combined:
            out_combined = os.path.join(args.output_dir, 'combined_full.txt')
            out_combined_ply = os.path.join(args.output_dir, 'combined_full.ply')
            print(f"Writing combined TXT to: {out_combined}")
            write_colmap_points3D(out_combined, combined_points, combined_colors, combined_errors, combined_ids, combined_tracks, header_colmap)
            print(f"Writing combined PLY to: {out_combined_ply}")
            write_ply(out_combined_ply, combined_points, combined_colors)
            written.append((out_combined, len(combined_points)))
            written.append((out_combined_ply, len(combined_points)))

        if args.out_combined_bounded:
            out_cropped = os.path.join(args.output_dir, 'combined_cropped_bounded_by_lidar.txt')
            out_cropped_ply = os.path.join(args.output_dir, 'combined_cropped_bounded_by_lidar.ply')
            print(f"Writing combined-bounded TXT to: {out_cropped}")
            write_colmap_points3D(out_cropped, cropped_points, cropped_colors, cropped_errors, cropped_ids, cropped_tracks, header_colmap)
            print(f"Writing combined-bounded PLY to: {out_cropped_ply}")
            write_ply(out_cropped_ply, cropped_points, cropped_colors)
            written.append((out_cropped, len(cropped_points)))
            written.append((out_cropped_ply, len(cropped_points)))

    else:
        # Original behavior when no selective flags were provided: write combined, cropped combined,
        # and bounded colmap if requested via --output_bounded_colmap
        print("\n" + "-"*70)
        print("OUTPUT 1: Combined point cloud (all points)")
        print("-"*70)
        output_combined = os.path.join(args.output_dir, 'points3Dcombined.txt')
        print(f"Writing combined point cloud to: {output_combined}")
        write_colmap_points3D(output_combined, combined_points, combined_colors, combined_errors, combined_ids, combined_tracks, header_colmap)
        output_combined_ply = os.path.join(args.output_dir, 'points3Dcombined.ply')
        print(f"Writing combined point cloud (PLY) to: {output_combined_ply}")
        write_ply(output_combined_ply, combined_points, combined_colors)
        written.append((output_combined, len(combined_points)))
        written.append((output_combined_ply, len(combined_points)))

        print("\n" + "-"*70)
        print("OUTPUT 2: Cropped combined point cloud (COLMAP within LiDAR bbox + all LiDAR)")
        print("-"*70)
        print(f"  ✓ Kept {len(points_colmap_filtered)}/{len(points_colmap)} COLMAP points within LiDAR bbox")
        output_cropped = os.path.join(args.output_dir, 'points3DcombinedCropped.txt')
        print(f"\nWriting cropped combined point cloud to: {output_cropped}")
        write_colmap_points3D(output_cropped, cropped_points, cropped_colors, cropped_errors, cropped_ids, cropped_tracks, header_colmap)
        output_cropped_ply = os.path.join(args.output_dir, 'points3DcombinedCropped.ply')
        print(f"Writing cropped combined point cloud (PLY) to: {output_cropped_ply}")
        write_ply(output_cropped_ply, cropped_points, cropped_colors)
        written.append((output_cropped, len(cropped_points)))
        written.append((output_cropped_ply, len(cropped_points)))

        if args.output_bounded_colmap:
            out_bounded_txt = os.path.join(args.output_dir, 'points3DColmapBounded.txt')
            out_bounded_ply = os.path.join(args.output_dir, 'points3DColmapBounded.ply')
            print("\n" + "-"*70)
            print("OUTPUT 3: Bounded COLMAP only (COLMAP points within LiDAR bbox, no LiDAR)")
            print("-"*70)
            print(f"Writing bounded COLMAP point cloud to: {out_bounded_txt}")
            write_colmap_points3D(out_bounded_txt, points_colmap_filtered, colors_colmap_filtered, errors_colmap_filtered, ids_colmap_filtered, tracks_colmap_filtered, header_colmap)
            print(f"Writing bounded COLMAP point cloud (PLY) to: {out_bounded_ply}")
            write_ply(out_bounded_ply, points_colmap_filtered, colors_colmap_filtered)
            written.append((out_bounded_txt, len(points_colmap_filtered)))
            written.append((out_bounded_ply, len(points_colmap_filtered)))

    # Print summary
    print("\n" + "="*70)
    print("✓ POINT CLOUD EXPORT COMPLETED!")
    print("="*70)
    print(f"\nOutputs saved to: {args.output_dir}")
    for fname, count in written:
        base = os.path.basename(fname)
        print(f"  - {base} ({count} points)")
    print()


if __name__ == "__main__":
    main()
