#!/usr/bin/env python3
"""
Invert binary mask images in a folder and write to an output folder.

Behavior:
- Accepts --input_folder containing mask images (alpha masks with values 0/1 or 0/255)
- Inverts values: 1 <-> 0 (and 255 <-> 0 when applicable)
- Ensures single-channel output
- Renames outputs by removing a trailing ".jpg" before ".png": e.g. image_name.jpg.png -> image_name.png

Usage:
  python invert_masks.py --input_folder <path> --output_folder <path>
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def load_mask_grayscale(path: Path) -> np.ndarray:
	"""Load an image as grayscale and return a uint8 array.

	Handles images that may have multiple channels by converting to grayscale and
	returning a single-channel image in [0,255].
	"""
	img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if img is None:
		raise RuntimeError(f"Failed to read image: {path}")

	# If image has alpha or multiple channels, convert to grayscale
	if img.ndim == 3:
		# If it has an alpha channel, prefer alpha as mask; else convert to gray
		if img.shape[2] == 4:
			# Use alpha channel as mask
			gray = img[:, :, 3]
		else:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img

	# Ensure uint8
	if gray.dtype != np.uint8:
		# Normalize to [0,255] if needed
		gmin, gmax = gray.min(), gray.max()
		if gmax > 1:
			gray = np.clip(gray, 0, 255).astype(np.uint8)
		else:
			gray = (gray * 255).astype(np.uint8)

	return gray


def invert_binary_mask(mask: np.ndarray) -> np.ndarray:
	"""Invert a binary mask robustly.

	Supports masks encoded as 0/1 or 0/255. Any non-zero is treated as 1.
	Output is uint8 with values 0 or 255.
	"""
	# Binarize: non-zero -> 1
	bin_mask = (mask > 0).astype(np.uint8)
	inv = 1 - bin_mask
	return (inv * 255).astype(np.uint8)


def make_output_name(input_name: str) -> str:
	"""Rename 'image_name.jpg.png' to 'image_name.png'; otherwise replace suffix with .png.
	Keeps the original base if it already ends with .png (no .jpg before it).
	"""
	if input_name.lower().endswith('.jpg.png'):
		return input_name[:-8] + '.png'  # strip ".jpg.png" and add ".png"
	# If name endswith .jpeg.png as well
	if input_name.lower().endswith('.jpeg.png'):
		return input_name[:-9] + '.png'

	# Otherwise, ensure .png extension
	if input_name.lower().endswith('.png'):
		return input_name  # already .png and no .jpg case
	# Replace common image extensions by .png
	for ext in ['.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
		if input_name.lower().endswith(ext):
			return input_name[: -len(ext)] + '.png'
	# Default: append .png
	return input_name + '.png'


def main():
	ap = argparse.ArgumentParser(description='Invert binary mask images and rename outputs')
	ap.add_argument('--input_folder', required=True, type=Path, help='Folder with input masks')
	ap.add_argument('--output_folder', required=True, type=Path, help='Folder to write inverted masks')
	args = ap.parse_args()

	if not args.input_folder.exists():
		raise FileNotFoundError(f"Input folder not found: {args.input_folder}")
	args.output_folder.mkdir(parents=True, exist_ok=True)

	exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG')
	files = [p for p in args.input_folder.iterdir() if p.is_file() and p.suffix in exts or any(str(p.name).lower().endswith(suf) for suf in ['.jpg.png', '.jpeg.png'])]

	print(f"Found {len(files)} mask files in {args.input_folder}")
	for inp in tqdm(sorted(files)):
		try:
			gray = load_mask_grayscale(inp)
			inv = invert_binary_mask(gray)
			out_name = make_output_name(inp.name)
			out_path = args.output_folder / out_name
			cv2.imwrite(str(out_path), inv)
		except Exception as e:
			print(f"Warning: failed to process {inp}: {e}")

	print(f"Done. Wrote inverted masks to {args.output_folder}")


if __name__ == '__main__':
	main()

