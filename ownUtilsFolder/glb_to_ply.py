#!/usr/bin/env python3

from pygltflib import GLTF2
import argparse
import numpy as np
from pathlib import Path
import struct

def extract_buffer_data(gltf, accessor_idx):
    """Extract data from a GLTF accessor"""
    accessor = gltf.accessors[accessor_idx]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    
    # Get the binary data
    data = gltf.binary_blob()
    
    # Calculate byte offset
    byte_offset = buffer_view.byteOffset or 0
    if accessor.byteOffset:
        byte_offset += accessor.byteOffset
    
    # Determine component size and format
    component_type_map = {
        5120: ('b', 1),   # BYTE
        5121: ('B', 1),   # UNSIGNED_BYTE
        5122: ('h', 2),   # SHORT
        5123: ('H', 2),   # UNSIGNED_SHORT
        5125: ('I', 4),   # UNSIGNED_INT
        5126: ('f', 4),   # FLOAT
    }
    
    fmt_char, component_size = component_type_map[accessor.componentType]
    
    # Determine number of components per element
    type_component_count = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16,
    }
    
    components_per_element = type_component_count[accessor.type]
    element_size = component_size * components_per_element
    
    # Use stride if available
    stride = buffer_view.byteStride if buffer_view.byteStride else element_size
    
    # Extract data
    result = []
    for i in range(accessor.count):
        element_offset = byte_offset + i * stride
        element_data = struct.unpack_from(
            f'<{components_per_element}{fmt_char}',
            data,
            element_offset
        )
        result.append(element_data)
    
    return np.array(result)

def write_ply_binary(positions, colors, output_path):
    """Write point cloud to binary PLY file"""
    num_points = len(positions)
    
    print(f"Writing {num_points} points to binary PLY...")
    
    with open(output_path, 'wb') as f:
        # Write ASCII header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode('ascii'))
        
        # Write binary data
        for i in range(num_points):
            x, y, z = positions[i]
            r, g, b = colors[i]
            f.write(struct.pack('<fffBBB', x, y, z, int(r), int(g), int(b)))
            
            if (i + 1) % 100000 == 0:
                print(f"  Written {i + 1}/{num_points} points...")

def generate_depth_images(positions, colors, output_dir, num_views=8):
    """Generate depth images from multiple viewpoints"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping depth image generation.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {num_views} depth views...")
    
    # Calculate scene bounds
    min_bounds = np.min(positions, axis=0)
    max_bounds = np.max(positions, axis=0)
    center = (min_bounds + max_bounds) / 2
    extent = np.max(max_bounds - min_bounds)
    
    print(f"Scene center: {center}")
    print(f"Scene extent: {extent:.2f}")
    
    # Generate views in a circle around the scene
    for view_idx in range(num_views):
        angle = 2 * np.pi * view_idx / num_views
        
        # Camera position on a circle
        radius = extent * 1.5
        cam_x = center[0] + radius * np.cos(angle)
        cam_y = center[1] + radius * np.sin(angle)
        cam_z = center[2]
        
        # Transform points to camera space
        cam_pos = np.array([cam_x, cam_y, cam_z])
        relative_pos = positions - cam_pos
        
        # Calculate depth (distance from camera)
        depths = np.linalg.norm(relative_pos, axis=1)
        
        # Project to 2D (rotate to align with camera view)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_proj = relative_pos[:, 0] * cos_a + relative_pos[:, 1] * sin_a
        z_proj = relative_pos[:, 2]
        
        # Create depth image with scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Depth visualization
        scatter1 = ax1.scatter(x_proj, z_proj, c=depths, cmap='viridis', 
                               s=0.1, alpha=0.8)
        ax1.set_xlim(-extent/2, extent/2)
        ax1.set_ylim(-extent/2, extent/2)
        ax1.set_aspect('equal')
        ax1.set_title(f'Depth Map - View {view_idx + 1} (Angle: {np.degrees(angle):.1f}°)')
        ax1.set_xlabel('X (projected)')
        ax1.set_ylabel('Z')
        plt.colorbar(scatter1, ax=ax1, label='Depth (distance from camera)')
        
        # Color visualization (RGB)
        scatter2 = ax2.scatter(x_proj, z_proj, c=colors/255.0, 
                               s=0.1, alpha=0.8)
        ax2.set_xlim(-extent/2, extent/2)
        ax2.set_ylim(-extent/2, extent/2)
        ax2.set_aspect('equal')
        ax2.set_title(f'RGB View {view_idx + 1}')
        ax2.set_xlabel('X (projected)')
        ax2.set_ylabel('Z')
        
        output_path = output_dir / f"depth_view_{view_idx:03d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated: {output_path.name}")
    
    # Generate a top-down view
    print("\nGenerating top-down view...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top-down depth (using Z coordinate as depth)
    scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], 
                          c=positions[:, 2], cmap='terrain', s=0.1, alpha=0.8)
    ax1.set_aspect('equal')
    ax1.set_title('Top-Down View - Elevation')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(scatter1, ax=ax1, label='Z (elevation)')
    
    # Top-down RGB
    scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], 
                          c=colors/255.0, s=0.1, alpha=0.8)
    ax2.set_aspect('equal')
    ax2.set_title('Top-Down View - RGB')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    output_path = output_dir / "depth_view_topdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: {output_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Extract PLY and depth images from GLB")
    parser.add_argument("--input", "-i", required=True, help="Input .glb file")
    parser.add_argument("--output-ply", default=None, help="Output PLY file")
    parser.add_argument("--output-depth-dir", default=None, help="Output directory for depth images")
    parser.add_argument("--num-views", type=int, default=8, help="Number of depth views to generate")
    parser.add_argument("--binary-ply", action="store_true", help="Write binary PLY (faster)")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth image generation")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Set default output paths
    if args.output_ply is None:
        args.output_ply = input_path.parent / f"{input_path.stem}.ply"
    
    if args.output_depth_dir is None:
        args.output_depth_dir = input_path.parent / f"{input_path.stem}_depth_images"
    
    print("=" * 80)
    print("GLB TO PLY AND DEPTH CONVERTER")
    print("=" * 80)
    print(f"\nInput file: {args.input}")
    
    print("\nLoading GLB file...")
    gltf = GLTF2().load(args.input)
    
    # Find the large point cloud
    print("\nSearching for point cloud mesh...")
    point_cloud_mesh = None
    
    for mesh_idx, mesh in enumerate(gltf.meshes):
        for primitive in mesh.primitives:
            if primitive.mode == 0:
                position_accessor_idx = primitive.attributes.POSITION
                position_accessor = gltf.accessors[position_accessor_idx]
                
                print(f"  Found point mesh: Mesh {mesh_idx}, {position_accessor.count} points")
                
                if position_accessor.count >= 100000:
                    point_cloud_mesh = (mesh_idx, primitive)
                    print(f"  -> Selected as main point cloud")
                    break
        if point_cloud_mesh:
            break
    
    if not point_cloud_mesh:
        print("Error: Could not find large point cloud in GLB file")
        return
    
    mesh_idx, primitive = point_cloud_mesh
    
    # Extract positions
    print("\nExtracting position data...")
    position_accessor_idx = primitive.attributes.POSITION
    positions = extract_buffer_data(gltf, position_accessor_idx)
    print(f"  Extracted {len(positions)} positions")
    print(f"  Position range:")
    print(f"    X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"    Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"    Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    
    # Extract colors
    print("\nExtracting color data...")
    color_accessor_idx = primitive.attributes.COLOR_0
    colors = extract_buffer_data(gltf, color_accessor_idx)
    
    if colors.shape[1] == 4:
        colors = colors[:, :3]
        print("  Detected RGBA format, using RGB only")
    
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
        print("  Normalized colors from [0,1] to [0,255]")
    else:
        colors = colors.astype(np.uint8)
    
    print(f"  Extracted {len(colors)} colors")
    
    # Write PLY file
    print(f"\n{'='*80}")
    print(f"Writing PLY file: {args.output_ply}")
    print(f"{'='*80}")
    
    write_ply_binary(positions, colors, args.output_ply)
    
    file_size_mb = Path(args.output_ply).stat().st_size / (1024 * 1024)
    print(f"\nPLY file written successfully! (Size: {file_size_mb:.2f} MB)")
    
    # Generate depth images
    if not args.skip_depth:
        print(f"\n{'='*80}")
        print("GENERATING DEPTH IMAGES")
        print(f"{'='*80}")
        generate_depth_images(positions, colors, args.output_depth_dir, num_views=args.num_views)
        print(f"\nDepth images saved to: {args.output_depth_dir}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total points:         {len(positions):,}")
    print("Done!")

if __name__ == "__main__":
    main()