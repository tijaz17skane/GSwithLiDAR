#!/usr/bin/env python3
"""
Add a spherical scaffold of points to a COLMAP points3D.txt file.

Usage:
  # Use percentage (default 10% of original points)
  python add_Scaffold_to_points3D.py --input points3D.txt --output points3D_with_scaffold.txt --offset 0.5 --percentage 10
  
  # Use explicit count
  python add_Scaffold_to_points3D.py --input points3D.txt --output points3D_with_scaffold.txt --offset 0.5 --num-points 2000
  
  # Also output as PLY
  python add_Scaffold_to_points3D.py --input points3D.txt --output points3D_with_scaffold.txt --output-ply output.ply --offset 0.5

This script:
 - Parses a COLMAP points3D.txt file
 - Computes the point-cloud centroid and maximal radius
 - Generates N points on a sphere of radius (max_radius + offset) using a Fibonacci sphere
   - N can be specified explicitly with --num-points, OR
   - N can be calculated as a percentage of original points with --percentage (default 10%)
 - Appends the scaffold points to the points3D list and writes an updated points3D.txt
 - Optionally outputs as PLY file for visualization with --output-ply

The output preserves header lines and updates the "# Number of points:" header when present.
"""

import argparse
import math
import numpy as np
import os
import sys


def parse_points3d_txt(path):
    headers = []
    points = []  # list of (id, x,y,z, r,g,b, error, track)
    max_id = -1

    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                headers.append(s)
                continue

            parts = s.split()
            # COLMAP points3D.txt has at least 8 columns: id x y z r g b error [track...]
            if len(parts) < 8:
                # skip malformed
                continue

            try:
                pid = int(parts[0])
                x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                r = int(parts[4]); g = int(parts[5]); b = int(parts[6])
                err = float(parts[7])
                track = ' '.join(parts[8:]) if len(parts) > 8 else ''

                points.append({
                    'id': pid,
                    'xyz': np.array([x, y, z], dtype=float),
                    'color': (r, g, b),
                    'error': err,
                    'track': track,
                })

                if pid > max_id:
                    max_id = pid

            except Exception:
                # ignore malformed lines
                continue

    return headers, points, max_id


def write_points3d_txt(path, headers, points):
    # Update header for number of points if present
    total_points = len(points)
    new_headers = []
    updated = False
    for h in headers:
        if h.startswith('#') and 'Number of points:' in h:
            new_headers.append(f"# Number of points: {total_points}")
            updated = True
        else:
            new_headers.append(h)

    if not updated:
        new_headers.append(f"# Number of points: {total_points}")

    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(path, 'w') as f:
        for h in new_headers:
            f.write(h + '\n')

        for p in points:
            pid = p['id']
            x, y, z = p['xyz']
            r, g, b = p['color']
            err = p['error']
            track = p['track']

            line_parts = [
                str(pid),
                f"{x:.8f}", f"{y:.8f}", f"{z:.8f}",
                str(r), str(g), str(b),
                f"{err:.8f}",
            ]
            if track:
                line_parts.append(track)

            f.write(' '.join(line_parts) + '\n')


def write_ply(path, points, include_color=True):
    """Write an ASCII PLY file with vertex positions and optional RGB colors.

    path: output ply path
    points: list of dicts with keys 'xyz' and 'color'
    """
    vertex_count = len(points)
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Header
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {vertex_count}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if include_color:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')

        for p in points:
            x, y, z = p['xyz']
            if include_color:
                r, g, b = p.get('color', (255, 255, 255))
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def fibonacci_sphere(samples):
    # Returns unit vectors positions on sphere using Fibonacci sphere algorithm
    rnd = 1.0
    points = []
    offset = 2.0 / samples
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(max(0.0, 1 - y * y))
        phi = ((i + rnd) % samples) * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points.append(np.array([x, y, z], dtype=float))

    return np.array(points)


def add_scaffold(headers, points, max_id, offset=0.5, num_points=2000, color=(255, 0, 0), error=0.0, verbose=False):
    # compute centroid and max distance
    xyz = np.stack([p['xyz'] for p in points], axis=0)
    centroid = np.mean(xyz, axis=0)
    dists = np.linalg.norm(xyz - centroid[None, :], axis=1)
    max_dist = float(np.max(dists))
    radius = max_dist + offset

    if verbose:
        print(f"Existing points: {len(points)}")
        print(f"Centroid: {centroid}")
        print(f"Max dist to centroid: {max_dist:.6f}")
        print(f"Scaffold radius: {radius:.6f} (offset {offset})")

    directions = fibonacci_sphere(num_points)
    scaffold_points = []
    next_id = max_id + 1
    for i in range(num_points):
        dir_vec = directions[i]
        pos = centroid + dir_vec * radius
        scaffold_points.append({
            'id': next_id + i,
            'xyz': pos,
            'color': color,
            'error': float(error),
            'track': ''
        })

    # Return combined list (original preserved order, scaffold appended)
    return points + scaffold_points


def parse_color(s):
    parts = s.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('Color must be R,G,B')
    return tuple(int(x) for x in parts)


def main():
    parser = argparse.ArgumentParser(description='Add spherical scaffold points to a COLMAP points3D.txt')
    parser.add_argument('--input', '-i', required=True, help='Input COLMAP points3D.txt')
    parser.add_argument('--output', '-o', required=True, help='Output points3D.txt with scaffold')
    parser.add_argument('--offset', type=float, default=5.0, help='Distance beyond max point radius to place scaffold (default 0.5)')
    parser.add_argument('--num-points', type=int, default=None, help='Number of scaffold points to generate (overrides --percentage)')
    parser.add_argument('--percentage', type=float, default=10.0, help='Percentage of original points to use for scaffold (default 10.0). Ignored if --num-points is specified.')
    parser.add_argument('--color', type=parse_color, default='255,0,0', help='RGB color for scaffold points as R,G,B (default 255,0,0)')
    parser.add_argument('--error', type=float, default=0.0, help='Error value to assign to scaffold points (default 0.0)')
    parser.add_argument('--output-ply', type=str, default=None, help='Optional: Output PLY file path. If not specified, no PLY will be generated.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file if exists')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}")
        sys.exit(1)

    if os.path.exists(args.output) and not args.overwrite:
        print(f"Error: output exists (use --overwrite to replace): {args.output}")
        sys.exit(1)

    headers, points, max_id = parse_points3d_txt(args.input)
    if len(points) == 0:
        print('No points found in input. Aborting.')
        sys.exit(1)

    # Calculate number of scaffold points based on percentage or explicit count
    if args.num_points is not None:
        num_scaffold_points = args.num_points
        if args.verbose:
            print(f"Using explicit count: {num_scaffold_points} scaffold points")
    else:
        num_scaffold_points = int(len(points) * args.percentage / 100.0)
        if args.verbose:
            print(f"Using {args.percentage}% of {len(points)} original points = {num_scaffold_points} scaffold points")
    
    # Ensure at least 1 scaffold point
    if num_scaffold_points < 1:
        num_scaffold_points = 1
        if args.verbose:
            print(f"Warning: Calculated scaffold points < 1, using minimum of 1")

    # ensure color is tuple
    if isinstance(args.color, str):
        try:
            color = parse_color(args.color)
        except Exception:
            color = (255, 0, 0)
    else:
        color = args.color

    if args.verbose:
        print(f"Loaded {len(points)} points, max existing id {max_id}")

    combined = add_scaffold(headers, points, max_id, offset=args.offset, num_points=num_scaffold_points, color=color, error=args.error, verbose=args.verbose)

    write_points3d_txt(args.output, headers, combined)
    print(f"✓ Wrote COLMAP points3D.txt to: {args.output}")

    # Optionally write PLY file
    if args.output_ply:
        try:
            write_ply(args.output_ply, combined, include_color=True)
            print(f"✓ Wrote PLY file to: {args.output_ply}")
        except Exception as e:
            print(f"❌ Warning: Failed to write PLY file: {e}")

    print(f"\n📊 Summary:")
    print(f"   Original points:      {len(points)}")
    print(f"   Scaffold points added: {num_scaffold_points}")
    print(f"   Total points:         {len(combined)}")


if __name__ == '__main__':
    main()
