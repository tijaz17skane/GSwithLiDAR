#!/usr/bin/env python3
"""
Dataset Aligner - Orchestrates COLMAP dataset alignment workflow.

This script aligns a COLMAP sparse reconstruction with a reference dataset by:
1. Using SRTaligner.py to compute transformation between sparse/0/images.txt and colmapCompleteOutput/images.txt
2. Replacing sparse/0/images.txt with aligned camera poses
3. Applying SRT transformation to sparse/0/points3D.txt using the computed transformation matrix

The alignment ensures spatial consistency between the sparse reconstruction and reference dataset.
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path


def validate_input_directory(input_dir):
    """
    Validate that the input directory contains required files.
    
    Parameters:
    -----------
    input_dir : str
        Path to input directory
        
    Returns:
    --------
    dict
        Dictionary containing validated file paths
    """
    input_dir = Path(input_dir)
    
    # Required files
    required_files = {
        'sparse_images': input_dir / 'sparse' / '0' / 'images.txt',
        'sparse_points3d': input_dir / 'sparse' / '0' / 'points3D.txt',
        'reference_images': input_dir / 'colmapCompleteOutput' / 'images.txt'
    }
    
    print("VALIDATING INPUT DIRECTORY")
    print("-" * 70)
    print(f"Input directory: {input_dir}")
    
    missing_files = []
    for name, filepath in required_files.items():
        if filepath.exists():
            print(f"✓ Found {name}: {filepath}")
        else:
            print(f"✗ Missing {name}: {filepath}")
            missing_files.append(str(filepath))
    
    if missing_files:
        print(f"\nError: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nRequired directory structure:")
        print("input_dir/")
        print("  ├── sparse/0/images.txt")
        print("  ├── sparse/0/points3D.txt")
        print("  └── colmapCompleteOutput/images.txt")
        sys.exit(1)
    
    print("✓ All required files found")
    return {name: str(filepath) for name, filepath in required_files.items()}


def run_srt_aligner(inputA, inputB, output_dir, script_dir):
    """
    Run SRTaligner.py to compute transformation between camera poses.
    
    Parameters:
    -----------
    inputA : str
        Path to source camera poses (sparse/0/images.txt)
    inputB : str
        Path to target camera poses (colmapCompleteOutput/images.txt)
    output_dir : str
        Output directory for alignment results
    script_dir : str
        Directory containing SRTaligner.py
        
    Returns:
    --------
    dict
        Dictionary containing output file paths
    """
    print("\nRUNNING SRT ALIGNMENT")
    print("-" * 70)
    
    srt_aligner_script = os.path.join(script_dir, 'SRTaligner.py')
    
    if not os.path.exists(srt_aligner_script):
        print(f"Error: SRTaligner.py not found at: {srt_aligner_script}")
        sys.exit(1)
    
    # Build command
    cmd = [
        'python', srt_aligner_script,
        '--inputA', inputA,
        '--inputB', inputB,
        '--output_dir', output_dir,
        '--save_params',
        '--verbose'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run SRTaligner.py
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("SRTaligner output:")
            print(result.stdout)
        
        if result.stderr:
            print("SRTaligner warnings/errors:")
            print(result.stderr)
        
        # Verify expected outputs
        expected_outputs = {
            'aligned_cameras': os.path.join(output_dir, 'aligned_in_cam.txt'),
            'transform_matrix': os.path.join(output_dir, 'combined_transform.txt'),
            'aligned_ply': os.path.join(output_dir, 'aligned_in_world.ply'),
            'params_file': os.path.join(output_dir, 'transformation_params.txt'),
            'cameras_A_original': os.path.join(output_dir, 'cameras_A_original_arrows.ply'),
            'cameras_B_target': os.path.join(output_dir, 'cameras_B_target_arrows.ply'),
            'cameras_A_aligned': os.path.join(output_dir, 'cameras_A_aligned_arrows.ply')
        }
        
        missing_outputs = []
        for name, filepath in expected_outputs.items():
            if os.path.exists(filepath):
                print(f"✓ Generated {name}: {filepath}")
            else:
                missing_outputs.append(f"{name}: {filepath}")
        
        if missing_outputs:
            print(f"\nWarning: Some expected outputs were not generated:")
            for output in missing_outputs:
                print(f"  - {output}")
        
        return expected_outputs
        
    except subprocess.CalledProcessError as e:
        print(f"Error running SRTaligner.py:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        sys.exit(1)


def replace_sparse_images(aligned_cameras_file, sparse_images_file):
    """
    Replace sparse/0/images.txt with aligned camera poses.
    
    Parameters:
    -----------
    aligned_cameras_file : str
        Path to aligned_in_cam.txt from SRTaligner
    sparse_images_file : str
        Path to sparse/0/images.txt to replace
    """
    print("\nREPLACING SPARSE IMAGES")
    print("-" * 70)
    
    if not os.path.exists(aligned_cameras_file):
        print(f"Error: Aligned cameras file not found: {aligned_cameras_file}")
        sys.exit(1)
    
    # Create backup of original
    backup_file = sparse_images_file + '.backup'
    print(f"Creating backup: {backup_file}")
    shutil.copy2(sparse_images_file, backup_file)
    
    # Replace with aligned version
    print(f"Replacing {sparse_images_file} with aligned cameras")
    shutil.copy2(aligned_cameras_file, sparse_images_file)
    
    print(f"✓ Replaced sparse/0/images.txt with aligned cameras")
    print(f"✓ Original backed up to: {backup_file}")


def apply_srt_to_points3d(points3d_file, transform_matrix_file, script_dir):
    """
    Apply SRT transformation to sparse/0/points3D.txt.
    
    Parameters:
    -----------
    points3d_file : str
        Path to sparse/0/points3D.txt
    transform_matrix_file : str
        Path to combined_transform.txt
    script_dir : str
        Directory containing apply_SRT_to_points3D_txt.py
    """
    print("\nAPPLYING SRT TRANSFORMATION TO 3D POINTS")
    print("-" * 70)
    
    srt_points_script = os.path.join(script_dir, 'apply_SRT_to_points3D_txt.py')
    
    if not os.path.exists(srt_points_script):
        print(f"Error: apply_SRT_to_points3D_txt.py not found at: {srt_points_script}")
        sys.exit(1)
    
    if not os.path.exists(transform_matrix_file):
        print(f"Error: Transform matrix file not found: {transform_matrix_file}")
        sys.exit(1)
    
    # Create backup of original points3D.txt
    backup_file = points3d_file + '.backup'
    print(f"Creating backup: {backup_file}")
    shutil.copy2(points3d_file, backup_file)
    
    # Create temporary output file
    temp_output = points3d_file + '.transformed_temp'
    
    # Build command
    cmd = [
        'python', srt_points_script,
        '--input_txt', points3d_file,
        '--output_txt', temp_output,
        '--SRT_transf_matrix', transform_matrix_file,
        '--verbose'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run apply_SRT_to_points3D_txt.py
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("SRT transformation output:")
            print(result.stdout)
        
        if result.stderr:
            print("SRT transformation warnings/errors:")
            print(result.stderr)
        
        # Replace original with transformed version
        if os.path.exists(temp_output):
            shutil.move(temp_output, points3d_file)
            print(f"✓ Replaced {points3d_file} with transformed points")
            print(f"✓ Original backed up to: {backup_file}")
        else:
            print(f"Error: Transformed output file not found: {temp_output}")
            sys.exit(1)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running apply_SRT_to_points3D_txt.py:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        sys.exit(1)


def print_summary(input_dir, alignment_output_dir):
    """Print summary of completed alignment process."""
    print("\n" + "="*70)
    print("DATASET ALIGNMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nInput directory: {input_dir}")
    print(f"Alignment outputs: {alignment_output_dir}")
    
    print(f"\nKey outputs:")
    print(f"  ✓ Transformation matrix: {os.path.join(alignment_output_dir, 'combined_transform.txt')}")
    print(f"  ✓ Transformation params: {os.path.join(alignment_output_dir, 'transformation_params.txt')}")
    print(f"  ✓ Aligned cameras (PLY): {os.path.join(alignment_output_dir, 'aligned_in_world.ply')}")
    print(f"  ✓ Camera orientations (original): {os.path.join(alignment_output_dir, 'cameras_A_original_arrows.ply')}")
    print(f"  ✓ Camera orientations (target): {os.path.join(alignment_output_dir, 'cameras_B_target_arrows.ply')}")
    print(f"  ✓ Camera orientations (aligned): {os.path.join(alignment_output_dir, 'cameras_A_aligned_arrows.ply')}")
    
    print(f"\nUpdated files:")
    print(f"  ✓ {os.path.join(input_dir, 'sparse/0/images.txt')} (aligned camera poses)")
    print(f"  ✓ {os.path.join(input_dir, 'sparse/0/points3D.txt')} (transformed 3D points)")
    
    print(f"\nBackups created:")
    print(f"  ✓ {os.path.join(input_dir, 'sparse/0/images.txt.backup')}")
    print(f"  ✓ {os.path.join(input_dir, 'sparse/0/points3D.txt.backup')}")
    
    print(f"\nThe sparse reconstruction is now aligned with the reference dataset.")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Align COLMAP sparse reconstruction with reference dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  This tool orchestrates the alignment of a COLMAP sparse reconstruction with a reference dataset.
  It uses SRTaligner.py to compute the transformation and applies it to both camera poses and 3D points.

WORKFLOW:
  1. Load sparse/0/images.txt (source) and colmapCompleteOutput/images.txt (reference)
  2. Compute similarity transformation using SRTaligner.py
  3. Replace sparse/0/images.txt with aligned camera poses
  4. Apply SRT transformation to sparse/0/points3D.txt

EXAMPLES:
  # Align dataset in current directory
  python datasetAligner.py --input_dir ./my_dataset
  
  # Align dataset with custom script directory
  python datasetAligner.py --input_dir ./my_dataset --script_dir ./custom_scripts

REQUIRED DIRECTORY STRUCTURE:
  input_dir/
    ├── sparse/0/
    │   ├── images.txt      (camera poses to align)
    │   └── points3D.txt    (3D points to transform)
    └── colmapCompleteOutput/
        └── images.txt      (reference camera poses)

OUTPUTS:
  input_dir/alignment_output/
    ├── combined_transform.txt       (4x4 transformation matrix)
    ├── transformation_params.txt    (detailed alignment statistics)
    ├── aligned_in_cam.txt          (aligned camera poses)
    └── aligned_in_world.ply        (visualization)
  
  Updated files:
    ├── sparse/0/images.txt         (replaced with aligned poses)
    ├── sparse/0/points3D.txt       (transformed 3D points)
    ├── sparse/0/images.txt.backup  (backup of original)
    └── sparse/0/points3D.txt.backup (backup of original)
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing sparse/0/ and colmapCompleteOutput/')
    parser.add_argument('--script_dir', type=str, default=None,
                       help='Directory containing SRTaligner.py and apply_SRT_to_points3D_txt.py (default: same as this script)')
    
    args = parser.parse_args()
    
    # Default script directory to the same directory as this script
    if args.script_dir is None:
        args.script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*70)
    print("COLMAP DATASET ALIGNMENT TOOL")
    print("="*70)
    print("Aligns sparse reconstruction with reference dataset using SRT transformation")
    print("="*70 + "\n")
    
    # Validate input directory and get file paths
    file_paths = validate_input_directory(args.input_dir)
    
    # Create alignment output directory
    alignment_output_dir = os.path.join(args.input_dir, 'alignment_output')
    os.makedirs(alignment_output_dir, exist_ok=True)
    print(f"\nAlignment output directory: {alignment_output_dir}")
    
    # Step 1: Run SRTaligner.py
    srt_outputs = run_srt_aligner(
        inputA=file_paths['sparse_images'],
        inputB=file_paths['reference_images'],
        output_dir=alignment_output_dir,
        script_dir=args.script_dir
    )
    
    # Step 2: Replace sparse/0/images.txt with aligned version
    replace_sparse_images(
        aligned_cameras_file=srt_outputs['aligned_cameras'],
        sparse_images_file=file_paths['sparse_images']
    )
    
    # Step 3: Apply SRT transformation to sparse/0/points3D.txt
    apply_srt_to_points3d(
        points3d_file=file_paths['sparse_points3d'],
        transform_matrix_file=srt_outputs['transform_matrix'],
        script_dir=args.script_dir
    )
    
    # Print completion summary
    print_summary(args.input_dir, alignment_output_dir)


if __name__ == "__main__":
    main()
