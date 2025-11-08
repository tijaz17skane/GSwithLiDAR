#!/usr/bin/env python3
"""
Replace quaternions in COLMAP images.txt file.

This script takes two COLMAP images.txt files and replaces the quaternion values
(QW, QX, QY, QZ) from inputA with the quaternion values from inputB, while keeping
all other values from inputA (translation, camera_id, etc.).
"""

import argparse
import sys
import os


def load_images_data(filepath):
    """
    Load COLMAP images.txt file and extract all data.
    
    Returns:
    --------
    images_dict : dict
        Dictionary mapping image name to all image data
    images_list : list
        List of image names in the order they appear in file (by IMAGE_ID)
    header_lines : list
        Comment/header lines at the beginning
    """
    images_dict = {}
    images_list = []  # Preserve order
    header_lines = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract header comments
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('#') or not line.strip():
            header_lines.append(line)
            i += 1
        else:
            break
    
    # Parse image data
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Parse camera pose line
        parts = line.split()
        if len(parts) >= 10:
            image_id = parts[0]
            qw, qx, qy, qz = parts[1], parts[2], parts[3], parts[4]
            tx, ty, tz = parts[5], parts[6], parts[7]
            camera_id = parts[8]
            name = parts[9]
            
            # Store all data
            images_dict[name] = {
                'image_id': image_id,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'camera_id': camera_id,
                'name': name,
                'pose_line': lines[i],
                'points2d_line': lines[i + 1] if i + 1 < len(lines) else '\n'
            }
            
            # Preserve order by IMAGE_ID
            images_list.append(name)
            
            i += 2
        else:
            i += 1
    
    return images_dict, images_list, header_lines


def replace_quaternions(inputA_path, inputB_path, output_path):
    """
    Replace quaternions in inputA with quaternions from inputB (matched by image name).
    Keep all other values (translation, camera_id, etc.) from inputA.
    
    Parameters:
    -----------
    inputA_path : str
        Path to first images.txt file (base file - translations come from here)
    inputB_path : str
        Path to second images.txt file (quaternions come from here)
    output_path : str
        Path to save the output file
    """
    print("\n" + "="*70)
    print("REPLACING QUATERNIONS IN COLMAP IMAGES.TXT")
    print("="*70 + "\n")
    
    # Load both files
    print(f"Loading inputA: {inputA_path}")
    images_A, images_A_order, header_A = load_images_data(inputA_path)
    print(f"  → Loaded {len(images_A)} images")
    
    print(f"\nLoading inputB: {inputB_path}")
    images_B, images_B_order, _ = load_images_data(inputB_path)
    print(f"  → Loaded {len(images_B)} images")
    
    # Find common images (matching by NAME column)
    common_names = set(images_A.keys()) & set(images_B.keys())
    
    print(f"\n" + "-"*70)
    print(f"Images in A:           {len(images_A)}")
    print(f"Images in B:           {len(images_B)}")
    print(f"Common images:         {len(common_names)}")
    print(f"Only in A:             {len(set(images_A.keys()) - set(images_B.keys()))}")
    print(f"Only in B:             {len(set(images_B.keys()) - set(images_A.keys()))}")
    
    if len(common_names) == 0:
        print("\n⚠ Error: No matching images found between the two files!")
        print("  Check that image filenames match.")
        sys.exit(1)
    
    # Create output file
    print(f"\n" + "-"*70)
    print("CREATING OUTPUT FILE")
    print("-"*70)
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    replaced_count = 0
    skipped_count = 0
    
    with open(output_path, 'w') as f:
        # Write header
        for header_line in header_A:
            f.write(header_line)
        
        # Process all images from A in their original order (by IMAGE_ID)
        for name in images_A_order:
            img_a = images_A[name]
            
            if name in images_B:
                # Replace quaternion with values from B (matched by NAME)
                img_b = images_B[name]
                
                # Write modified pose line with quaternions from B
                f.write(f"{img_a['image_id']} "
                       f"{img_b['qw']} {img_b['qx']} {img_b['qy']} {img_b['qz']} "
                       f"{img_a['tx']} {img_a['ty']} {img_a['tz']} "
                       f"{img_a['camera_id']} {img_a['name']}\n")
                
                replaced_count += 1
            else:
                # Keep original line if no match in B (NAME not found in inputB)
                f.write(img_a['pose_line'])
                skipped_count += 1
            
            # Write POINTS2D line as-is
            f.write(img_a['points2d_line'])
    
    print(f"✓ Output saved to: {output_path}")
    print(f"\n  Replaced quaternions: {replaced_count} images")
    print(f"  Kept original:        {skipped_count} images")
    
    # Show sample of replacements (by matching NAME)
    if replaced_count > 0:
        print(f"\n" + "-"*70)
        print("SAMPLE REPLACEMENTS (first 5 matched by NAME):")
        print("-"*70)
        
        sample_count = 0
        for name in images_A_order:
            if name in images_B and sample_count < 5:
                img_a = images_A[name]
                img_b = images_B[name]
                
                print(f"\n{sample_count+1}. NAME: {name}")
                print(f"   Original (A): QW={img_a['qw']:>12s} QX={img_a['qx']:>12s} "
                      f"QY={img_a['qy']:>12s} QZ={img_a['qz']:>12s}")
                print(f"   New (from B): QW={img_b['qw']:>12s} QX={img_b['qx']:>12s} "
                      f"QY={img_b['qy']:>12s} QZ={img_b['qz']:>12s}")
                
                sample_count += 1
        
        if replaced_count > 5:
            print(f"\n   ... and {replaced_count - 5} more images")
    
    print("\n" + "="*70)
    print("✓ QUATERNION REPLACEMENT COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Replace quaternions in COLMAP images.txt file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  This tool takes two COLMAP images.txt files and creates a new file where:
  - Quaternions (QW, QX, QY, QZ) are taken from inputB
  - All other values (TX, TY, TZ, CAMERA_ID, etc.) are kept from inputA
  
  Images are matched by their filename (NAME field).

EXAMPLES:
  # Replace quaternions in fileA with quaternions from fileB
  python replace_quaternions.py --inputA images_A.txt --inputB images_B.txt --output output.txt
  
  # Process COLMAP sparse reconstruction files
  python replace_quaternions.py \\
    --inputA /path/to/sparse/0/images.txt \\
    --inputB /path/to/aligned/images.txt \\
    --output /path/to/output/images.txt

WORKFLOW:
  1. Load both images.txt files
  2. Match images by filename (NAME field)
  3. For each matching image:
     - Take quaternion (QW, QX, QY, QZ) from inputB
     - Keep translation (TX, TY, TZ) and other fields from inputA
  4. Write output file in COLMAP format

INPUT FORMAT:
  COLMAP images.txt format:
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    1 0.123 0.456 0.789 0.012 1.0 2.0 3.0 1 image001.jpg
    
        """
    )
    
    parser.add_argument('--inputA', type=str, required=True,
                       help='First images.txt file (base file - translations come from here)')
    parser.add_argument('--inputB', type=str, required=True,
                       help='Second images.txt file (quaternions come from here)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output images.txt file path')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.inputA):
        print(f"Error: Input file A not found: {args.inputA}")
        sys.exit(1)
    
    if not os.path.exists(args.inputB):
        print(f"Error: Input file B not found: {args.inputB}")
        sys.exit(1)
    
    # Run the replacement
    replace_quaternions(args.inputA, args.inputB, args.output)


if __name__ == "__main__":
    main()
