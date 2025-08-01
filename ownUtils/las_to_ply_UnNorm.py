#!/usr/bin/env python3
"""
Convert LAS/LAZ files to PLY format (no normalization, keep original RGB)
"""

import laspy
import numpy as np
import argparse
import os
from plyfile import PlyData, PlyElement
import sys

def las_to_ply_unnorm(las_path, ply_path, color_scale=1.0, intensity_to_rgb=True):
    """
    Convert LAS/LAZ file to PLY format
    Use original XYZ coordinates (no normalization)
    Keep RGB values if present
    """
    print(f"Reading LAS/LAZ file: {las_path}")
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).transpose()

    # Use RGB if available, else fallback to intensity or white
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        rgb = np.vstack((las.red, las.green, las.blue)).transpose()
        rgb = rgb.astype(np.uint8)
        print("Using RGB data from LAS file")
    elif intensity_to_rgb and hasattr(las, 'intensity'):
        intensity = las.intensity
        intensity_norm = ((intensity - intensity.min()) / (intensity.max() - intensity.min()) * 255).astype(np.uint8)
        rgb = np.column_stack([intensity_norm, intensity_norm, intensity_norm])
        print("Converting intensity to RGB")
    else:
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)
        print("No color data found, using white")

    if color_scale != 1.0:
        rgb = np.clip(rgb * color_scale, 0, 255).astype(np.uint8)

    vertex_data = np.zeros(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['nx'] = 0.0
    vertex_data['ny'] = 0.0
    vertex_data['nz'] = 0.0
    vertex_data['red'] = rgb[:, 0]
    vertex_data['green'] = rgb[:, 1]
    vertex_data['blue'] = rgb[:, 2]

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

    print(f"Converted {len(xyz)} points to PLY format (no normalization)")
    print(f"Output saved to: {ply_path}")
    print(f"Point cloud bounds:")
    print(f"  X: {xyz[:, 0].min():.3f} to {xyz[:, 0].max():.3f}")
    print(f"  Y: {xyz[:, 1].min():.3f} to {xyz[:, 1].max():.3f}")
    print(f"  Z: {xyz[:, 2].min():.3f} to {xyz[:, 2].max():.3f}")

def main():
    parser = argparse.ArgumentParser(description="Convert LAS/LAZ files to PLY format (no normalization)")
    parser.add_argument("input", help="Input LAS/LAZ file path")
    parser.add_argument("output", help="Output PLY file path")
    parser.add_argument("--color-scale", type=float, default=1.0,
                       help="Scale factor for RGB values (default: 1.0)")
    parser.add_argument("--no-intensity-rgb", action="store_true",
                       help="Don't convert intensity to RGB if no color data")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)

    try:
        las_to_ply_unnorm(args.input, args.output, args.color_scale, not args.no_intensity_rgb)
    except Exception as e:
        print(f"Error converting file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
