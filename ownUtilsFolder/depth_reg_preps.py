#!/usr/bin/env python3
"""
Depth Regularization Preparation Tool

This script prepares a COLMAP dataset for depth-supervised Gaussian Splatting by:
1. Generating depth maps using Depth-Anything-V2
2. Converting COLMAP sparse model to binary format
3. Computing depth scales for regularization

Required for depth supervision during Gaussian Splatting training.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def validate_input_directory(input_dir):
    """
    Validate that the input directory contains required files and directories.
    
    Parameters:
    -----------
    input_dir : str
        Path to input directory
        
    Returns:
    --------
    dict
        Dictionary containing validated paths
    """
    input_dir = Path(input_dir)
    
    # Required paths
    required_paths = {
        'images_dir': input_dir / 'images',
        'sparse_dir': input_dir / 'sparse' / '0',
        'images_txt': input_dir / 'sparse' / '0' / 'images.txt',
        'cameras_txt': input_dir / 'sparse' / '0' / 'cameras.txt',
        'points3d_txt': input_dir / 'sparse' / '0' / 'points3D.txt'
    }
    
    print("VALIDATING INPUT DIRECTORY")
    print("-" * 70)
    print(f"Input directory: {input_dir}")
    
    missing_paths = []
    for name, filepath in required_paths.items():
        if filepath.exists():
            if name == 'images_dir':
                # Count images in directory
                image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
                image_files = [f for f in filepath.iterdir() 
                             if f.suffix.lower() in image_extensions]
                print(f"✓ Found {name}: {filepath} ({len(image_files)} images)")
            else:
                print(f"✓ Found {name}: {filepath}")
        else:
            print(f"✗ Missing {name}: {filepath}")
            missing_paths.append(str(filepath))
    
    if missing_paths:
        print(f"\nError: Missing required files/directories:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nRequired directory structure:")
        print("input_dir/")
        print("  ├── images/              (input images)")
        print("  └── sparse/0/")
        print("      ├── images.txt")
        print("      ├── cameras.txt")
        print("      └── points3D.txt")
        sys.exit(1)
    
    # Create depth_images directory if it doesn't exist
    depth_images_dir = input_dir / 'depth_images'
    depth_images_dir.mkdir(exist_ok=True)
    print(f"✓ Depth output directory: {depth_images_dir}")
    
    print("✓ All required files and directories found")
    return {name: str(filepath) for name, filepath in required_paths.items()}


def run_depth_anything_v2(images_dir, depth_output_dir):
    """
    Run Depth-Anything-V2 to generate depth maps for input images.
    
    Parameters:
    -----------
    images_dir : str
        Directory containing input images
    depth_output_dir : str
        Directory to save generated depth maps
    """
    print("\nRUNNING DEPTH-ANYTHING-V2")
    print("-" * 70)
    
    depth_script = "/mnt/data/tijaz/Depth-Anything-V2/metric_depth/run.py"
    checkpoint_path = "/mnt/data/tijaz/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"
    
    # Verify script and checkpoint exist
    if not os.path.exists(depth_script):
        print(f"Error: Depth-Anything-V2 script not found: {depth_script}")
        sys.exit(1)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Depth-Anything-V2 checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Build command
    cmd = [
        'python', depth_script,
        '--encoder', 'vitl',
        '--pred-only',
        '--grayscale',
        '--img-path', images_dir,
        '--outdir', depth_output_dir,
        '--load-from', checkpoint_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run Depth-Anything-V2
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("Depth-Anything-V2 output:")
            print(result.stdout)
        
        if result.stderr:
            print("Depth-Anything-V2 warnings/errors:")
            print(result.stderr)
        
        # Count generated depth maps
        depth_files = list(Path(depth_output_dir).glob('*.png'))
        print(f"✓ Generated {len(depth_files)} depth maps in {depth_output_dir}")
        
        if len(depth_files) == 0:
            print("Warning: No depth maps were generated!")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Depth-Anything-V2:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        sys.exit(1)


def convert_colmap_to_binary(sparse_dir):
    """
    Convert COLMAP sparse model from text to binary format.
    
    Parameters:
    -----------
    sparse_dir : str
        Directory containing COLMAP sparse model (sparse/0/)
    """
    print("\nCONVERTING COLMAP MODEL TO BINARY FORMAT")
    print("-" * 70)
    
    # Build command
    cmd = [
        'colmap', 'model_converter',
        '--input_path', sparse_dir,
        '--output_path', sparse_dir,
        '--output_type', 'BIN'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run COLMAP model converter
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("COLMAP converter output:")
            print(result.stdout)
        
        if result.stderr:
            print("COLMAP converter warnings/errors:")
            print(result.stderr)
        
        # Verify binary files were created
        binary_files = ['cameras.bin', 'images.bin', 'points3D.bin']
        missing_files = []
        
        for bin_file in binary_files:
            bin_path = os.path.join(sparse_dir, bin_file)
            if os.path.exists(bin_path):
                print(f"✓ Generated {bin_file}")
            else:
                missing_files.append(bin_file)
        
        if missing_files:
            print(f"Warning: Some binary files were not generated:")
            for file in missing_files:
                print(f"  - {file}")
            return False
        
        print(f"✓ Successfully converted COLMAP model to binary format")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running COLMAP model converter:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        sys.exit(1)


def compute_depth_scales(base_dir, depths_dir):
    """
    Compute depth scales for depth regularization.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing COLMAP model
    depths_dir : str
        Directory containing generated depth maps
    """
    print("\nCOMPUTING DEPTH SCALES")
    print("-" * 70)
    
    depth_scale_script = "/mnt/data/tijaz/gaussian-splatting/utils/make_depth_scale.py"
    
    # Verify script exists
    if not os.path.exists(depth_scale_script):
        print(f"Error: Depth scale script not found: {depth_scale_script}")
        sys.exit(1)
    
    # Build command
    cmd = [
        'python', depth_scale_script,
        '--base_dir', base_dir,
        '--depths_dir', depths_dir
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run depth scale computation
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("Depth scale computation output:")
            print(result.stdout)
        
        if result.stderr:
            print("Depth scale computation warnings/errors:")
            print(result.stderr)
        
        print(f"✓ Successfully computed depth scales")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running depth scale computation:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        sys.exit(1)


def print_summary(input_dir):
    """Print summary of completed depth regularization preparation."""
    print("\n" + "="*70)
    print("DEPTH REGULARIZATION PREPARATION COMPLETED!")
    print("="*70)
    
    print(f"\nInput directory: {input_dir}")
    
    print(f"\nGenerated outputs:")
    print(f"  ✓ Depth maps: {os.path.join(input_dir, 'depth_images')}/*.png")
    print(f"  ✓ Binary COLMAP model: {os.path.join(input_dir, 'sparse/0')}/*.bin")
    print(f"  ✓ Depth scales computed for regularization")
    
    print(f"\nYour dataset is now ready for depth-supervised Gaussian Splatting training!")
    print(f"\nExample training command:")
    print(f"python train.py -s {input_dir} -m ./output --iterations 30000 -d {os.path.join(input_dir, 'depth_images')}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare COLMAP dataset for depth-supervised Gaussian Splatting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  This tool prepares a COLMAP dataset for depth-supervised Gaussian Splatting training by:
  1. Generating depth maps using Depth-Anything-V2
  2. Converting COLMAP sparse model to binary format
  3. Computing depth scales for regularization

WORKFLOW:
  1. Run Depth-Anything-V2 on images/ directory
  2. Convert sparse/0/ model from text to binary format
  3. Compute depth scales using make_depth_scale.py

EXAMPLES:
  # Prepare dataset for depth supervision
  python depth_reg_preps.py --input_dir ./my_colmap_dataset
  
  # With verbose output
  python depth_reg_preps.py --input_dir ./my_colmap_dataset --verbose

REQUIRED DIRECTORY STRUCTURE:
  input_dir/
    ├── images/              (input images for depth generation)
    └── sparse/0/
        ├── images.txt       (COLMAP camera poses)
        ├── cameras.txt      (COLMAP camera parameters)
        └── points3D.txt     (COLMAP 3D points)

OUTPUTS GENERATED:
  input_dir/
    ├── depth_images/        (generated depth maps)
    │   ├── image1.png
    │   └── image2.png
    └── sparse/0/
        ├── cameras.bin      (binary COLMAP model)
        ├── images.bin
        ├── points3D.bin
        └── [original .txt files preserved]

DEPENDENCIES:
  - Depth-Anything-V2 installation at /mnt/data/tijaz/Depth-Anything-V2/
  - COLMAP binary in PATH
  - Gaussian Splatting utils at /mnt/data/tijaz/gaussian-splatting/utils/
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing COLMAP dataset (images/ and sparse/0/)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output from all commands')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DEPTH REGULARIZATION PREPARATION TOOL")
    print("="*70)
    print("Prepares COLMAP dataset for depth-supervised Gaussian Splatting")
    print("="*70 + "\n")
    
    # Validate input directory and get paths
    paths = validate_input_directory(args.input_dir)
    
    # Step 1: Generate depth maps using Depth-Anything-V2
    depth_output_dir = os.path.join(args.input_dir, 'depth_images')
    success = run_depth_anything_v2(paths['images_dir'], depth_output_dir)
    
    if not success:
        print("Error: Depth map generation failed. Cannot continue.")
        sys.exit(1)
    
    # Step 2: Convert COLMAP model to binary format
    success = convert_colmap_to_binary(paths['sparse_dir'])
    
    if not success:
        print("Warning: COLMAP binary conversion had issues, but continuing...")
    
    # Step 3: Compute depth scales
    success = compute_depth_scales(args.input_dir, depth_output_dir)
    
    if not success:
        print("Error: Depth scale computation failed.")
        sys.exit(1)
    
    # Print completion summary
    print_summary(args.input_dir)


if __name__ == "__main__":
    main()
