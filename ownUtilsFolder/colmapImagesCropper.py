#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from typing import Set


def collect_image_basenames(images_folder: Path) -> Set[str]:
    """Recursively collects all file basenames from a directory."""
    if not images_folder.is_dir():
        print(f"Error: Provided images folder is not a valid directory: {images_folder}")
        return set()
    return {p.name for p in images_folder.rglob("*") if p.is_file()}


def filter_images_txt(input_file: Path, output_file: Path, image_basenames: Set[str]):
    """
    Filters an images.txt file. If a line contains a pose for an image in
    image_basenames, that line and the next line are written to the output.
    """
    try:
        with input_file.open("r", encoding="utf-8") as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return

    kept_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip header lines
        if line.strip().startswith('#') or not line.strip():
            i += 1
            continue
        
        # Check if this line contains an image name that's in our folder
        parts = line.split()
        if len(parts) >= 10:
            image_name = parts[9]
            if os.path.basename(image_name) in image_basenames:
                # Write this line and the next line
                kept_lines.append(line)
                if (i + 1) < len(lines):
                    kept_lines.append(lines[i + 1])
                i += 2
            else:
                i += 2
        else:
            i += 1

    # Write the results to the output file
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f_out:
            # Write a simple header with the correct final count
            f_out.write("# Image list with two lines of data per image:\n")
            f_out.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f_out.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f_out.write(f"# Number of images: {len(kept_lines) // 2}\n\n")
            f_out.writelines(kept_lines)
        print(f"Kept {len(kept_lines) // 2} images. Output written to: {output_file}")
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Filter COLMAP images.txt by images present in a folder.")
    parser.add_argument("--input_txt", type=Path, required=True, help="Path to the input COLMAP images.txt file.")
    parser.add_argument("--images_folder", type=Path, required=True, help="Path to the folder containing images to keep.")
    parser.add_argument("--output_txt", type=Path, required=True, help="Path for the output filtered images.txt file.")
    args = parser.parse_args()

    image_basenames = collect_image_basenames(args.images_folder)
    if not image_basenames:
        print(f"Warning: No image files found in {args.images_folder}. The output file will be empty.")

    filter_images_txt(args.input_txt, args.output_txt, image_basenames)


if __name__ == "__main__":
    main()
