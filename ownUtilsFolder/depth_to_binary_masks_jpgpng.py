#!/usr/bin/env python3
"""
Convert a folder of grayscale depth images (0-255) into binary masks based on a percentage threshold.

For each depth image:
  - Pixels with value < (remove_below_percent / 100.0) * 255 become 0 in the mask.
  - All other pixels become 1.

Output masks retain original spatial resolution and are saved as single-channel PNGs
with pixel values exactly 0 or 1.

Example:
  python depth_to_binary_masks.py \
      --depth_folder /path/to/depth_images \
      --mask_folder /path/to/output_masks \
      --remove_below 10

This produces masks where any depth < 25.5 (10% of 255) is set to 0, others 1.
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate binary masks from depth images using percentage threshold")
    p.add_argument("--depth_folder", type=Path, required=True, help="Folder containing input depth images (grayscale 0-255)")
    p.add_argument("--mask_folder", type=Path, required=True, help="Output folder for generated binary masks")
    p.add_argument("--remove_below", type=float, required=True, help="Percentage threshold (0-100). Depth values below this % of 255 become 0 in mask")
    p.add_argument("--exts", type=str, default=".png", help="Comma-separated list of image extensions to process")
    p.add_argument("--keep_structure", action="store_true", help="Preserve any subdirectory structure under depth_folder in mask_folder")
    return p.parse_args()


def collect_depth_files(root: Path, exts):
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def main():
    args = parse_args()

    if not (0 <= args.remove_below <= 100):
        print(f"--remove_below must be within [0,100], got {args.remove_below}", file=sys.stderr)
        sys.exit(1)

    depth_root: Path = args.depth_folder
    mask_root: Path = args.mask_folder

    if not depth_root.exists():
        print(f"Depth folder does not exist: {depth_root}", file=sys.stderr)
        sys.exit(1)

    mask_root.mkdir(parents=True, exist_ok=True)

    exts = [e.strip().lower() for e in args.exts.split(',') if e.strip()]
    threshold_value = (args.remove_below / 100.0) * 255.0
    print(f"Threshold (absolute): {threshold_value:.2f} (percentage: {args.remove_below}%)")

    depth_files = collect_depth_files(depth_root, exts)
    if not depth_files:
        print("No depth images found matching extensions: " + ", ".join(exts))
        sys.exit(0)

    for depth_path in tqdm(depth_files, desc="Processing depth images"):
        # Load as grayscale
        img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not read image {depth_path}")
            continue

        # If multi-channel, convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ensure 0-255 uint8
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            # Normalize if not in 0-255 range
            max_val = img.max() if img.max() > 0 else 1.0
            img = (img / max_val) * 255.0
            img = img.clip(0, 255).astype(np.uint8)

        mask = (img >= threshold_value).astype(np.uint8)  # True->1, False->0

        # Derive output relative path
        if args.keep_structure:
            rel = depth_path.relative_to(depth_root)
            out_dir = mask_root / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = mask_root

        # Naming: output mask as image_name.jpg.png (preserving original stem + .jpg.png)
        original_name = depth_path.name
        original_stem = depth_path.stem  # e.g., "image_name" from "image_name.png"
        
        # Construct output name as stem + .jpg.png
        output_name = original_stem + '.jpg.png'

        out_path = out_dir / output_name

        # Save mask (0/1 values). If you want 0/255, multiply by 255.
        cv2.imwrite(str(out_path), mask*255)

    print(f"Done. Wrote {len(depth_files)} mask files to {mask_root}")


if __name__ == "__main__":
    main()
