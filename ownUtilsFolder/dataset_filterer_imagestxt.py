#!/usr/bin/env python3
"""
Filter COLMAP images.txt by Image Folder

This script reads a COLMAP images.txt file and filters it to only include
entries (both odd and even lines) for images that exist in a specified folder.

Usage:
    python filter_images_txt.py --images_txt /path/to/images.txt --images_folder /path/to/images --output /path/to/filtered_images.txt
"""

import os
import argparse


def get_image_filenames(folder_path):
    """
    Get set of image filenames from a folder.
    
    Returns:
    --------
    set : Set of filenames (case-sensitive)
    """
    if not os.path.exists(folder_path):
        print(f"Warning: Folder does not exist: {folder_path}")
        return set()
    
    filenames = set()
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            filenames.add(item)
    
    return filenames


def filter_images_txt(images_txt_path, images_folder, output_path):
    """
    Filter COLMAP images.txt file to only include entries for images in the folder.
    
    Parameters:
    -----------
    images_txt_path : str
        Path to input COLMAP images.txt file
    images_folder : str
        Path to folder containing images to keep
    output_path : str
        Path to output filtered images.txt file
    """
    print("="*70)
    print("FILTER COLMAP IMAGES.TXT BY IMAGE FOLDER")
    print("="*70)
    print(f"Input images.txt:  {images_txt_path}")
    print(f"Images folder:     {images_folder}")
    print(f"Output images.txt: {output_path}")
    print("="*70 + "\n")
    
    # Validate input file exists
    if not os.path.exists(images_txt_path):
        print(f"❌ Error: Input file does not exist: {images_txt_path}")
        return
    
    # Get set of image filenames from folder
    print("📂 Reading images folder...")
    image_filenames = get_image_filenames(images_folder)
    print(f"   Found {len(image_filenames)} images in folder")
    
    if len(image_filenames) == 0:
        print("⚠️  Warning: No images found in folder!")
    
    # Read and filter images.txt
    print("\n📄 Reading and filtering images.txt...")
    
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    
    # Separate header and data
    header_lines = []
    data_lines = []
    
    for line in lines:
        if line.strip().startswith('#') or not line.strip():
            header_lines.append(line)
        else:
            data_lines.append(line)
    
    print(f"   Total lines in input: {len(lines)}")
    print(f"   Header lines: {len(header_lines)}")
    print(f"   Data lines: {len(data_lines)}")
    
    # Process data lines in pairs (odd line = camera pose, even line = points2D)
    filtered_data = []
    kept_count = 0
    removed_count = 0
    
    i = 0
    while i < len(data_lines):
        # Get odd line (camera pose)
        if i >= len(data_lines):
            break
        
        odd_line = data_lines[i]
        parts = odd_line.strip().split()
        
        # Check if this is a valid camera pose line (should have at least 10 parts)
        if len(parts) < 10:
            # Not a camera pose line, skip it
            i += 1
            continue
        
        # Extract image name (last element)
        image_name = parts[9]
        
        # Get even line (points2D data)
        even_line = data_lines[i + 1] if i + 1 < len(data_lines) else "\n"
        
        # Check if image exists in folder
        if image_name in image_filenames:
            filtered_data.append(odd_line)
            filtered_data.append(even_line)
            kept_count += 1
        else:
            removed_count += 1
        
        # Move to next pair
        i += 2
    
    print(f"\n📊 Filtering results:")
    print(f"   Images kept:    {kept_count}")
    print(f"   Images removed: {removed_count}")
    
    # Update header with new count
    updated_header = []
    for line in header_lines:
        if line.startswith('# Number of images:'):
            # Update the count
            updated_header.append(f"# Number of images: {kept_count}\n")
        else:
            updated_header.append(line)
    
    # Write output file
    print(f"\n💾 Writing filtered images.txt...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write header
        f.writelines(updated_header)
        # Write filtered data
        f.writelines(filtered_data)
    
    print(f"   ✓ Wrote {kept_count} images to {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input file:  {images_txt_path}")
    print(f"Output file: {output_path}")
    print(f"\nImages in folder:          {len(image_filenames)}")
    print(f"Images in input file:      {kept_count + removed_count}")
    print(f"Images in output file:     {kept_count}")
    print(f"Images removed:            {removed_count}")
    
    if removed_count > 0:
        print(f"\n⚠️  {removed_count} images were removed (not found in folder)")
    if kept_count < len(image_filenames):
        print(f"\n⚠️  {len(image_filenames) - kept_count} images in folder were not found in images.txt")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Filter COLMAP images.txt to only include images present in a folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  This script reads a COLMAP images.txt file and filters it to only include
  entries for images that exist in the specified images folder.
  
  For each image in images.txt:
  - If the image file exists in --images_folder, both the camera pose line (odd)
    and the POINTS2D line (even) are kept in the output
  - If the image file does NOT exist in --images_folder, both lines are removed

FORMAT:
  Input/Output COLMAP images.txt format:
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: N
    
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    POINTS2D data...
    ...

EXAMPLES:
  # Filter images.txt to only include images in a folder
  python filter_images_txt.py --images_txt sparse/0/images.txt --images_folder images/ --output sparse/0/images_filtered.txt
  
  # Use absolute paths
  python filter_images_txt.py --images_txt /path/to/images.txt --images_folder /path/to/images --output /path/to/output.txt

USE CASE:
  - You have a COLMAP reconstruction with many images but only want to keep a subset
  - Filter images.txt to match a specific set of images in a folder
  - Prepare data for training that only uses certain images
        """
    )
    
    parser.add_argument('--images_txt', type=str, required=True,
                       help='Path to input COLMAP images.txt file')
    parser.add_argument('--images_folder', type=str, required=True,
                       help='Path to folder containing images to keep')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output filtered images.txt file')
    
    args = parser.parse_args()
    
    filter_images_txt(args.images_txt, args.images_folder, args.output)


if __name__ == "__main__":
    main()
