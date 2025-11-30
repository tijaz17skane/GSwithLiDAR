import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import json


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
    if not pcd.has_colors():
        raise ValueError(f"No colors found in {ply_path}")
    return pcd


def calculate_color_errors(colors_a, colors_b):
    """
    Calculate color errors between two color arrays.
    
    Args:
        colors_a: First color array (N x 3), values in [0, 1]
        colors_b: Second color array (N x 3), values in [0, 1]
        
    Returns:
        dict: Dictionary containing various error metrics
    """
    # Per-channel absolute differences
    diff = np.abs(colors_a - colors_b)
    
    # Per-channel errors
    r_error = diff[:, 0]
    g_error = diff[:, 1]
    b_error = diff[:, 2]
    
    # Overall color error (L2 distance in RGB space)
    overall_error = np.sqrt(np.sum((colors_a - colors_b) ** 2, axis=1))
    
    # Per-channel statistics
    per_channel_stats = {
        'R': {
            'MAE': np.mean(r_error),
            'MSE': np.mean(r_error ** 2),
            'RMSE': np.sqrt(np.mean(r_error ** 2)),
            'max': np.max(r_error),
            'min': np.min(r_error),
            'median': np.median(r_error),
            'std': np.std(r_error),
            'p95': np.percentile(r_error, 95),
            'p99': np.percentile(r_error, 99)
        },
        'G': {
            'MAE': np.mean(g_error),
            'MSE': np.mean(g_error ** 2),
            'RMSE': np.sqrt(np.mean(g_error ** 2)),
            'max': np.max(g_error),
            'min': np.min(g_error),
            'median': np.median(g_error),
            'std': np.std(g_error),
            'p95': np.percentile(g_error, 95),
            'p99': np.percentile(g_error, 99)
        },
        'B': {
            'MAE': np.mean(b_error),
            'MSE': np.mean(b_error ** 2),
            'RMSE': np.sqrt(np.mean(b_error ** 2)),
            'max': np.max(b_error),
            'min': np.min(b_error),
            'median': np.median(b_error),
            'std': np.std(b_error),
            'p95': np.percentile(b_error, 95),
            'p99': np.percentile(b_error, 99)
        }
    }
    
    # Overall statistics
    overall_stats = {
        'MAE': np.mean(overall_error),
        'MSE': np.mean(overall_error ** 2),
        'RMSE': np.sqrt(np.mean(overall_error ** 2)),
        'max': np.max(overall_error),
        'min': np.min(overall_error),
        'median': np.median(overall_error),
        'std': np.std(overall_error),
        'p95': np.percentile(overall_error, 95),
        'p99': np.percentile(overall_error, 99)
    }
    
    return {
        'per_channel': per_channel_stats,
        'overall': overall_stats,
        'per_point_errors': overall_error,
        'per_channel_errors': {
            'R': r_error,
            'G': g_error,
            'B': b_error
        }
    }


def create_error_heatmap(points, errors, output_path=None, colormap='jet'):
    """
    Create a point cloud colored by error magnitude (heat map).
    
    Args:
        points: Point coordinates (N x 3)
        errors: Per-point errors (N,)
        output_path: Optional path to save the colored point cloud
        colormap: Matplotlib colormap name
        
    Returns:
        open3d.geometry.PointCloud: Point cloud colored by error
    """
    # Normalize errors to [0, 1] for colormap
    errors_normalized = (errors - np.min(errors)) / (np.max(errors) - np.min(errors) + 1e-10)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Map errors to colors
    colors = cmap(errors_normalized)[:, :3]  # RGB only, discard alpha
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save if path provided
    if output_path:
        o3d.io.write_point_cloud(str(output_path), pcd)
    
    return pcd


def print_error_statistics(error_dict):
    """
    Print formatted error statistics.
    """
    print("\n" + "="*70)
    print("COLOR ERROR ANALYSIS")
    print("="*70)
    
    # Per-channel errors
    print("\n" + "-"*70)
    print("PER-CHANNEL ERRORS")
    print("-"*70)
    
    for channel in ['R', 'G', 'B']:
        stats = error_dict['per_channel'][channel]
        print(f"\n{channel} Channel:")
        print(f"  MAE (Mean Absolute Error):  {stats['MAE']:.6f}")
        print(f"  MSE (Mean Squared Error):   {stats['MSE']:.6f}")
        print(f"  RMSE (Root MSE):            {stats['RMSE']:.6f}")
        print(f"  Median:                     {stats['median']:.6f}")
        print(f"  Std Dev:                    {stats['std']:.6f}")
        print(f"  Min Error:                  {stats['min']:.6f}")
        print(f"  Max Error:                  {stats['max']:.6f}")
        print(f"  95th Percentile:            {stats['p95']:.6f}")
        print(f"  99th Percentile:            {stats['p99']:.6f}")
    
    # Overall error
    print("\n" + "-"*70)
    print("OVERALL COLOR ERROR (L2 Distance in RGB Space)")
    print("-"*70)
    
    stats = error_dict['overall']
    print(f"\n  MAE (Mean Absolute Error):  {stats['MAE']:.6f}")
    print(f"  MSE (Mean Squared Error):   {stats['MSE']:.6f}")
    print(f"  RMSE (Root MSE):            {stats['RMSE']:.6f}")
    print(f"  Median:                     {stats['median']:.6f}")
    print(f"  Std Dev:                    {stats['std']:.6f}")
    print(f"  Min Error:                  {stats['min']:.6f}")
    print(f"  Max Error:                  {stats['max']:.6f}")
    print(f"  95th Percentile:            {stats['p95']:.6f}")
    print(f"  99th Percentile:            {stats['p99']:.6f}")
    
    print("\n" + "="*70)


def save_error_statistics(error_dict, output_path):
    """
    Save error statistics to JSON file.
    """
    # Convert numpy types to native Python types for JSON serialization
    json_dict = {
        'per_channel': {},
        'overall': {}
    }
    
    for channel in error_dict['per_channel']:
        json_dict['per_channel'][channel] = {
            k: float(v) for k, v in error_dict['per_channel'][channel].items()
        }
    
    json_dict['overall'] = {
        k: float(v) for k, v in error_dict['overall'].items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


def create_error_histogram(error_dict, output_path):
    """
    Create and save histograms of color errors.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Color Error Distribution', fontsize=16, fontweight='bold')
    
    # Per-channel errors
    channels = ['R', 'G', 'B']
    colors_hist = ['red', 'green', 'blue']
    
    for i, (channel, color) in enumerate(zip(channels, colors_hist)):
        ax = axes[i // 2, i % 2]
        errors = error_dict['per_channel_errors'][channel]
        
        ax.hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Absolute Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{channel} Channel Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats = error_dict['per_channel'][channel]
        stats_text = f"MAE: {stats['MAE']:.4f}\nRMSE: {stats['RMSE']:.4f}\nMax: {stats['max']:.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    # Overall error
    ax = axes[1, 1]
    overall_errors = error_dict['per_point_errors']
    
    ax.hist(overall_errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('L2 Distance', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Overall Color Error Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats = error_dict['overall']
    stats_text = f"MAE: {stats['MAE']:.4f}\nRMSE: {stats['RMSE']:.4f}\nMax: {stats['max']:.4f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Error histogram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate color differences between two point clouds with identical geometry'
    )
    parser.add_argument('--plyA', required=True,
                       help='Path to first point cloud (reference)')
    parser.add_argument('--plyB', required=True,
                       help='Path to second point cloud (comparison)')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for results (default: same as plyB)')
    parser.add_argument('--colormap', default='jet',
                       help='Colormap for error visualization (default: jet). Options: jet, hot, cool, viridis, plasma')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the error heat map')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Tolerance for point position matching (default: 1e-6)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.plyA).exists():
        raise FileNotFoundError(f"PLY A not found: {args.plyA}")
    if not Path(args.plyB).exists():
        raise FileNotFoundError(f"PLY B not found: {args.plyB}")
    
    print("="*70)
    print("COLOR DIFFERENCE ANALYSIS")
    print("="*70)
    
    # Load point clouds
    print(f"\nLoading point cloud A: {args.plyA}")
    pcd_a = read_point_cloud(args.plyA)
    points_a = np.asarray(pcd_a.points)
    colors_a = np.asarray(pcd_a.colors)
    print(f"  Points: {len(points_a):,}")
    
    print(f"\nLoading point cloud B: {args.plyB}")
    pcd_b = read_point_cloud(args.plyB)
    points_b = np.asarray(pcd_b.points)
    colors_b = np.asarray(pcd_b.colors)
    print(f"  Points: {len(points_b):,}")
    
    # Check if point clouds have the same number of points
    if len(points_a) != len(points_b):
        raise ValueError(f"Point clouds must have the same number of points. "
                        f"A has {len(points_a)}, B has {len(points_b)}")
    
    # Check if points are in the same order (optional strict check)
    point_diff = np.max(np.abs(points_a - points_b))
    print(f"\nMaximum point position difference: {point_diff:.2e}")
    
    if point_diff > args.tolerance:
        print(f"Warning: Points may not be in the same order (max diff: {point_diff:.2e} > tolerance: {args.tolerance})")
        print("Results may not be meaningful if point clouds have different geometry.")
    else:
        print("Point positions match (within tolerance).")
    
    # Calculate color errors
    print("\nCalculating color errors...")
    error_dict = calculate_color_errors(colors_a, colors_b)
    
    # Print statistics
    print_error_statistics(error_dict)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.plyB).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save statistics to JSON
    json_path = output_dir / "color_error_statistics.json"
    save_error_statistics(error_dict, json_path)
    print(f"\nStatistics saved to: {json_path}")
    
    # Create error histogram
    histogram_path = output_dir / "color_error_histogram.png"
    create_error_histogram(error_dict, histogram_path)
    
    # Create error heat map
    print("\nCreating error heat map...")
    heatmap_path = output_dir / "color_error_heatmap.ply"
    pcd_heatmap = create_error_heatmap(
        points_b,
        error_dict['per_point_errors'],
        heatmap_path,
        colormap=args.colormap
    )
    print(f"Error heat map saved to: {heatmap_path}")
    
    # Create colorbar reference image
    print("\nCreating colorbar reference...")
    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.5)
    
    cmap = cm.get_cmap(args.colormap)
    norm = plt.Normalize(vmin=np.min(error_dict['per_point_errors']),
                        vmax=np.max(error_dict['per_point_errors']))
    
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=ax, orientation='horizontal')
    cb.set_label('Color Error (L2 Distance)', fontsize=12, fontweight='bold')
    
    colorbar_path = output_dir / "color_error_colorbar.png"
    plt.savefig(colorbar_path, dpi=300, bbox_inches='tight')
    print(f"Colorbar reference saved to: {colorbar_path}")
    plt.close()
    
    # Visualize if requested
    if args.visualize:
        print("\nLaunching visualization...")
        print("  - Blue/Cool colors: Low error")
        print("  - Red/Hot colors: High error")
        print("  - Use mouse to rotate/zoom")
        print("  - Press 'q' to close")
        o3d.visualization.draw_geometries(
            [pcd_heatmap],
            window_name="Color Error Heat Map",
            width=1024,
            height=768
        )
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()