#!/usr/bin/env python3
"""
Visualize FHF dataset: Convert LAS to PLY and export camera centers from meta.json as PLY points.
"""
import laspy
import numpy as np
import argparse
import os
import json
from plyfile import PlyData, PlyElement

def las_to_ply_points(las_path, ply_path, normalize=False, min_xyz=None):
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    print("LAS points BEFORE normalization:")
    print(f"  X: {xyz[:,0].min():.6f} to {xyz[:,0].max():.6f}")
    print(f"  Y: {xyz[:,1].min():.6f} to {xyz[:,1].max():.6f}")
    print(f"  Z: {xyz[:,2].min():.6f} to {xyz[:,2].max():.6f}")
    if normalize and min_xyz is not None:
        xyz = xyz - min_xyz
        print("LAS points AFTER normalization:")
        print(f"  X: {xyz[:,0].min():.6f} to {xyz[:,0].max():.6f}")
        print(f"  Y: {xyz[:,1].min():.6f} to {xyz[:,1].max():.6f}")
        print(f"  Z: {xyz[:,2].min():.6f} to {xyz[:,2].max():.6f}")
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        rgb = np.vstack((las.red, las.green, las.blue)).transpose()
        rgb = rgb.astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)
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
    PlyData([PlyElement.describe(vertex_data, 'vertex')]).write(ply_path)
    print(f"LAS points written to {ply_path} ({len(xyz)} points)")
    return xyz

def meta_to_camera_centers(meta_path, ply_path, normalize=False, min_xyz=None):
    with open(meta_path) as f:
        meta = json.load(f)
    images = meta.get('images', [])
    if not images and 'spherical_images' in meta:
        images = meta['spherical_images']
    centers = []
    for img in images:
        pose = img.get('pose', {})
        t = pose.get('translation', None)
        if t and len(t) == 3:
            centers.append(t)
    centers = np.array(centers, dtype=np.float32)
    if centers.shape[0] > 0:
        print("Camera centers BEFORE normalization:")
        print(f"  X: {centers[:,0].min():.6f} to {centers[:,0].max():.6f}")
        print(f"  Y: {centers[:,1].min():.6f} to {centers[:,1].max():.6f}")
        print(f"  Z: {centers[:,2].min():.6f} to {centers[:,2].max():.6f}")
    if normalize and min_xyz is not None:
        centers = centers - min_xyz
        if centers.shape[0] > 0:
            print("Camera centers AFTER normalization:")
            print(f"  X: {centers[:,0].min():.6f} to {centers[:,0].max():.6f}")
            print(f"  Y: {centers[:,1].min():.6f} to {centers[:,1].max():.6f}")
            print(f"  Z: {centers[:,2].min():.6f} to {centers[:,2].max():.6f}")
    vertex_data = np.zeros(len(centers), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = centers[:, 0]
    vertex_data['y'] = centers[:, 1]
    vertex_data['z'] = centers[:, 2]
    vertex_data['nx'] = 0.0
    vertex_data['ny'] = 0.0
    vertex_data['nz'] = 0.0
    vertex_data['red'] = 255
    vertex_data['green'] = 0
    vertex_data['blue'] = 0
    PlyData([PlyElement.describe(vertex_data, 'vertex')]).write(ply_path)
    print(f"Camera centers written to {ply_path} ({len(centers)} cameras)")

def main():
    parser = argparse.ArgumentParser(description="Visualize FHF: LAS to PLY and camera centers from meta.json to PLY")
    parser.add_argument('--las', required=True, help='Input LAS/LAZ file')
    parser.add_argument('--meta', required=True, help='Input meta.json file')
    parser.add_argument('--out-las-ply', required=True, help='Output PLY for LAS points')
    parser.add_argument('--out-cam-ply', required=True, help='Output PLY for camera centers')
    parser.add_argument('--normalize', action='store_true', help='Normalize both LAS and camera centers by subtracting min(x), min(y), min(z) from LAS')
    args = parser.parse_args()
    min_xyz = None
    if args.normalize:
        las = laspy.read(args.las)
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        min_xyz = xyz.min(axis=0)
        print(f"Normalization enabled. min_xyz: {min_xyz}")
    las_to_ply_points(args.las, args.out_las_ply, normalize=args.normalize, min_xyz=min_xyz)
    meta_to_camera_centers(args.meta, args.out_cam_ply, normalize=args.normalize, min_xyz=min_xyz)

if __name__ == '__main__':
    main()
