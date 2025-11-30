#!/usr/bin/env python3
"""
Delete Non-Matching Images

This script compares two image folders and deletes all images from folder2
that are NOT present in folder1 (based on filename).

Usage:
    python delete_non_matching_images.py --folder1 /path/to/reference --folder2 /path/to/filter
    python delete_non_matching_images.py --folder1 /path/to/reference --folder2 /path/to/filter --dry-run
"""

import os
import argparse
from pathlib import Path


def get_image_filenames(folder_path, extensions=None):
    """
    Get set of image filenames (without path and without extensions) from a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder
    extensions : list or None
        List of valid image extensions. If None, uses common image formats.
    
    Returns:
    --------
    set : Set of filenames without extensions (lowercase for case-insensitive matching)
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}
    else:
        extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions}
    
    filenames = set()
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder does not exist: {folder_path}")
        return filenames
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            name, ext = os.path.splitext(item)
            if ext.lower() in extensions:
                filenames.add(name.lower())  # Store name without extension, lowercase
    
    return filenames


def delete_non_matching_images(folder1, folder2, dry_run=False, extensions=None):
    """
    Delete images from folder2 that are not present in folder1.
    
    Parameters:
    -----------
    folder1 : str
        Reference folder (images to keep)
    folder2 : str
        Target folder (images will be deleted if not in folder1)
    dry_run : bool
        If True, only report what would be deleted without actually deleting
    extensions : list or None
        List of valid image extensions
    """
    print("="*70)
    print("DELETE NON-MATCHING IMAGES")
    print("="*70)
    print(f"Reference folder (keep): {folder1}")
    print(f"Target folder (filter):  {folder2}")
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be deleted")
    print("="*70 + "\n")
    
    # Validate folders exist
    if not os.path.exists(folder1):
        print(f"❌ Error: Reference folder does not exist: {folder1}")
        return
    
    if not os.path.exists(folder2):
        print(f"❌ Error: Target folder does not exist: {folder2}")
        return
    
    # Get image filenames from folder1 (reference)
    print("📂 Reading reference folder (folder1)...")
    folder1_images = get_image_filenames(folder1, extensions)
    print(f"   Found {len(folder1_images)} images in reference folder")
    
    # Get image filenames from folder2 (target)
    print("\n📂 Reading target folder (folder2)...")
    folder2_images_with_case = {}  # Map lowercase name (no ext) -> original full filename
    for item in os.listdir(folder2):
        item_path = os.path.join(folder2, item)
        if os.path.isfile(item_path):
            name, ext = os.path.splitext(item)
            valid_extensions = extensions if extensions else {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}
            if isinstance(extensions, list):
                valid_extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions}
            if ext.lower() in valid_extensions:
                folder2_images_with_case[name.lower()] = item  # Map name without extension to full filename
    
    print(f"   Found {len(folder2_images_with_case)} images in target folder")
    
    # Find images in folder2 that are NOT in folder1 (compare names without extensions)
    print("\n🔍 Finding non-matching images (comparing names without extensions)...")
    to_delete = []
    for img_name_lower, img_original in folder2_images_with_case.items():
        if img_name_lower not in folder1_images:
            to_delete.append(img_original)
    
    print(f"   Found {len(to_delete)} images to delete from folder2")
    print(f"   Keeping {len(folder2_images_with_case) - len(to_delete)} images that match folder1")
    
    if len(to_delete) == 0:
        print("\n✅ No images to delete. All images in folder2 are present in folder1.")
        return
    
    # Delete or report files
    print(f"\n{'🔍 Would delete' if dry_run else '🗑️  Deleting'} {len(to_delete)} images...")
    
    deleted_count = 0
    failed_count = 0
    
    for i, filename in enumerate(to_delete, 1):
        filepath = os.path.join(folder2, filename)
        
        if dry_run:
            print(f"   [{i}/{len(to_delete)}] Would delete: {filename}")
            deleted_count += 1
        else:
            try:
                os.remove(filepath)
                print(f"   [{i}/{len(to_delete)}] Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"   [{i}/{len(to_delete)}] ❌ Failed to delete {filename}: {e}")
                failed_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Reference folder: {folder1}")
    print(f"Target folder:    {folder2}")
    print(f"\nImages in reference folder (folder1): {len(folder1_images)}")
    print(f"Images in target folder (folder2):    {len(folder2_images_with_case)}")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN: {deleted_count} images would be deleted")
    else:
        print(f"\n✅ Successfully deleted: {deleted_count} images")
        if failed_count > 0:
            print(f"❌ Failed to delete:     {failed_count} images")
        print(f"\n📊 Remaining images in folder2: {len(folder2_images_with_case) - deleted_count}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Delete images from folder2 that are NOT present in folder1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  This script compares two image folders and deletes all images from folder2
  that are not present in folder1 (based on filename WITHOUT extension, case-insensitive).
  
  Folder1 is the reference folder - its images are preserved.
  Folder2 is the target folder - images not in folder1 will be deleted.
  
  NOTE: Comparison is by filename only, ignoring extensions. For example:
  - folder1/image001.jpg will match folder2/image001.png
  - Files are matched by name regardless of extension

EXAMPLES:
  # Dry run (see what would be deleted without actually deleting)
  python delete_non_matching_images.py --folder1 /path/to/reference --folder2 /path/to/filter --dry-run
  
  # Actually delete non-matching images
  python delete_non_matching_images.py --folder1 /path/to/reference --folder2 /path/to/filter
  
  # Only process specific image extensions
  python delete_non_matching_images.py --folder1 /path/to/reference --folder2 /path/to/filter --extensions jpg png

USE CASE:
  - You have a reference set of images (folder1) and want folder2 to contain only those same images
  - Clean up a target folder to match a reference folder's image set
  - Remove images from folder2 that don't exist in folder1

WARNING:
  This script permanently deletes files! Use --dry-run first to preview changes.
        """
    )
    
    parser.add_argument('--folder1', type=str, required=True,
                       help='Reference folder (images to keep as reference)')
    parser.add_argument('--folder2', type=str, required=True,
                       help='Target folder (images will be deleted if not in folder1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode - show what would be deleted without actually deleting')
    parser.add_argument('--extensions', nargs='+', type=str,
                       help='Image extensions to process (e.g., jpg png). If not specified, uses common formats.')
    
    args = parser.parse_args()
    
    
    print("\n⚠️  WARNING: This will permanently delete files from folder2!")
    print(f"   Reference folder (keep): {args.folder1}")
    print(f"   Target folder (delete):  {args.folder2}")
        
    delete_non_matching_images(args.folder1, args.folder2, args.dry_run, args.extensions)


if __name__ == "__main__":
    main()
