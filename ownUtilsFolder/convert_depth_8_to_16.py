#!/usr/bin/env python
"""Convert 8-bit depth images to 16-bit.

Features:
  * Recursively scans an --input_folder for images (configurable extensions)
  * Converts 8-bit per-channel data to 16-bit using one of three modes:
	  scale   : value * 257 (preserves integer ratios across full 0-65535 span)
	  stretch : per-image min/max normalization to 0-65535 (enhances contrast)
	  copy    : direct cast without scaling (values 0-255 remain 0-255)
  * Creates --output_folder if missing (can mirror subfolder structure with --keep_structure)
  * Optionally overwrites existing outputs (--overwrite)
  * Prints progress and simple statistics

Example:
  python convert_depth_8_to_16.py \
	  --input_folder /path/8bit_depth \
	  --output_folder /path/16bit_depth \
	  --mode scale --exts .png .jpg --keep_structure
"""

import argparse
import os
from pathlib import Path
import sys
from typing import List

try:
	import cv2
	import numpy as np
except ImportError as e:
	print("ERROR: This script requires opencv-python and numpy. Install them first.")
	raise


def parse_args():
	p = argparse.ArgumentParser(description="Convert 8-bit depth images to 16-bit.")
	p.add_argument("--input_folder", required=True, type=Path,
				   help="Folder containing 8-bit depth images.")
	p.add_argument("--output_folder", required=True, type=Path,
				   help="Destination folder for 16-bit outputs.")
	p.add_argument("--mode", choices=["scale", "stretch", "copy"], default="scale",
				   help="Conversion mode: scale (value*257), stretch (normalize), copy (cast).")
	p.add_argument("--exts", nargs="*", default=[".png", ".jpg", ".jpeg"],
				   help="List of file extensions to include (case-insensitive).")
	p.add_argument("--keep_structure", action="store_true",
				   help="Preserve relative subdirectory structure in output.")
	p.add_argument("--overwrite", action="store_true",
				   help="Overwrite existing output files if they exist.")
	p.add_argument("--dry_run", action="store_true",
				   help="List planned conversions without writing files.")
	return p.parse_args()


def find_images(root: Path, exts: List[str]) -> List[Path]:
	norm_exts = {e.lower() for e in exts}
	files = []
	for p in root.rglob('*'):
		if p.is_file() and p.suffix.lower() in norm_exts:
			files.append(p)
	return sorted(files)


def convert_image(img8: 'np.ndarray', mode: str) -> 'np.ndarray':
	if img8.dtype != np.uint8:
		raise ValueError(f"Expected uint8 image, got {img8.dtype}")
	if mode == 'copy':
		return img8.astype(np.uint16)
	if mode == 'scale':
		# Map 0..255 -> 0..65535 using value*257 (255*257=65535)
		return (img8.astype(np.uint16) * 257)
	if mode == 'stretch':
		mn = int(img8.min())
		mx = int(img8.max())
		if mx == mn:
			# All pixels identical; just scale using *257 to avoid division by zero
			return (img8.astype(np.uint16) * 257)
		# Normalize to 0..65535
		stretched = ((img8.astype(np.float32) - mn) / (mx - mn) * 65535.0).round().astype(np.uint16)
		return stretched
	raise ValueError(f"Unknown mode {mode}")


def ensure_output_path(base_out: Path, input_root: Path, file_path: Path, keep_structure: bool) -> Path:
	if keep_structure:
		rel = file_path.parent.relative_to(input_root)
		out_dir = base_out / rel
	else:
		out_dir = base_out
	out_dir.mkdir(parents=True, exist_ok=True)
	# Force PNG for 16-bit reliability (JPEG generally limited to 8-bit) even if source was jpg
	return out_dir / (file_path.stem + '.png')


def main():
	args = parse_args()

	if not args.input_folder.is_dir():
		print(f"ERROR: Input folder does not exist: {args.input_folder}")
		sys.exit(1)

	args.output_folder.mkdir(parents=True, exist_ok=True)
	images = find_images(args.input_folder, args.exts)
	if not images:
		print("No images found matching extensions.")
		sys.exit(0)

	print(f"Found {len(images)} input images.")
	print(f"Mode: {args.mode}")
	print(f"Output: {args.output_folder}")
	if args.dry_run:
		print("--dry_run enabled: no files will be written.")

	converted = 0
	skipped = 0
	for idx, img_path in enumerate(images, start=1):
		try:
			img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
			if img is None:
				print(f"[WARN] Could not read image: {img_path}")
				skipped += 1
				continue
			if img.dtype != 'uint8':
				print(f"[INFO] Skipping non-8-bit image: {img_path} dtype={img.dtype}")
				skipped += 1
				continue

			# If colored, process each channel independently
			if len(img.shape) == 3 and img.shape[2] > 1:
				channels = cv2.split(img)
				converted_channels = [convert_image(ch, args.mode) for ch in channels]
				img16 = cv2.merge(converted_channels)
			else:
				img16 = convert_image(img, args.mode)

			out_path = ensure_output_path(args.output_folder, args.input_folder, img_path, args.keep_structure)
			if out_path.exists() and not args.overwrite:
				print(f"[SKIP] Exists (use --overwrite): {out_path}")
				skipped += 1
				continue

			if not args.dry_run:
				success = cv2.imwrite(str(out_path), img16)
				if not success:
					print(f"[ERROR] Failed to write: {out_path}")
					skipped += 1
					continue
			converted += 1
			if converted % 25 == 0:
				print(f"Progress: {converted} written / {idx} processed")
		except Exception as e:
			print(f"[ERROR] {img_path}: {e}")
			skipped += 1

	print("\nConversion complete.")
	print(f"Images processed: {len(images)}")
	print(f"Images converted: {converted}")
	print(f"Images skipped:   {skipped}")
	if args.mode == 'stretch':
		print("Note: 'stretch' mode applied per-image min/max normalization.")
	elif args.mode == 'scale':
		print("Note: 'scale' mode used value*257 mapping (0..255 -> 0..65535).")
	else:
		print("Note: 'copy' mode performed direct uint8->uint16 casting.")


if __name__ == '__main__':
	main()

