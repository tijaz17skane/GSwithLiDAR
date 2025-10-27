#!/usr/bin/env python3
"""
Filter meta.json to only keep entries for images that exist in the given images/ and spherical_images/ folders.
"""
import os
import json
import argparse

def filter_meta(meta_path, images_dir, spherical_images_dir, out_path):
    with open(meta_path) as f:
        meta = json.load(f)

    # Helper to get all filenames in a folder
    def get_filenames(folder):
        if not os.path.isdir(folder):
            return set()
        return set(os.listdir(folder))

    images_files = get_filenames(images_dir)
    spherical_files = get_filenames(spherical_images_dir)
    print(f"First 5 files in spherical_images_dir: {sorted(list(spherical_files))[:5]}")
    # Show first 5 expected basenames from meta.json
    if 'spherical_images' in meta:
        expected_sph = [os.path.basename(img['path']) for img in meta['spherical_images'] if 'path' in img]
        print(f"First 5 expected spherical_images basenames: {expected_sph[:5]}")

    # Filter 'images' list
    if 'images' in meta:
        filtered_images = [img for img in meta['images'] if 'path' in img and os.path.basename(img['path']) in images_files]
        print(f"Filtered images: {len(filtered_images)} of {len(meta['images'])}")
        meta['images'] = filtered_images

    # Filter 'spherical_images' list
    if 'spherical_images' in meta:
        filtered_spherical = [img for img in meta['spherical_images'] if 'path' in img and os.path.basename(img['path']) in spherical_files]
        print(f"Filtered spherical_images: {len(filtered_spherical)} of {len(meta['spherical_images'])}")
        meta['spherical_images'] = filtered_spherical

    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Filtered meta.json written to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Filter meta.json to only keep entries for existing images.")
    parser.add_argument('--meta', required=True, help='Input meta.json')
    parser.add_argument('--images-dir', required=True, help='Path to images/ folder')
    parser.add_argument('--spherical-images-dir', required=True, help='Path to spherical_images/ folder')
    parser.add_argument('--out', required=True, help='Output filtered meta.json')
    args = parser.parse_args()
    filter_meta(args.meta, args.images_dir, args.spherical_images_dir, args.out)

if __name__ == '__main__':
    main()
