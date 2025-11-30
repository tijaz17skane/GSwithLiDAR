#!/usr/bin/env python3
"""
Filter a COLMAP images.txt-style file (inputA) by keeping only images whose
NAME appears in another images.txt-style file (inputB).

Format reminder (two lines per image):
  IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
  POINTS2D[] as (X, Y, POINT3D_ID)

What this script does:
- Reads all NAME values from inputB
- Reads inputA and retains only the two-line blocks whose NAME is in inputB
- Preserves comments (lines starting with '#')
- Updates the 'Number of images:' line to the new count
- Writes result to --output (defaults to inputA.filtered.txt)

Usage:
  python filter_colmap_images.py --inputA path/to/images_A.txt \
								 --inputB path/to/images_B.txt \
								 --output path/to/images_filtered.txt
"""

import argparse
from pathlib import Path
from typing import List, Set, Tuple


def parse_name_from_image_line(line: str) -> str:
	"""Extract NAME (the last field) from an image line.

	Handles comma- or whitespace-separated formats. Typical COLMAP images.txt
	is whitespace-separated with NAME as the last token. This function strips
	trailing commas if present.
	"""
	# If commas exist, split on commas; else split on whitespace
	parts = [p.strip() for p in (line.split(',') if ',' in line else line.split())]
	if not parts:
		return ""
	name = parts[-1]
	# Remove any trailing commas
	return name.rstrip(',')


def collect_names_from_images_file(path: Path) -> Set[str]:
	"""Collect all image NAMEs from a COLMAP images.txt-style file.

	We consider non-comment lines and assume images are defined on the first
	of two successive lines. The first line contains metadata (ending with NAME),
	the second line contains POINTS2D.
	"""
	names: Set[str] = set()
	with path.open('r', encoding='utf-8', errors='ignore') as f:
		lines = f.readlines()

	i = 0
	n = len(lines)
	while i < n:
		line = lines[i].strip()
		if not line or line.startswith('#'):
			i += 1
			continue
		# This is an image header line, next line is points (if present)
		name = parse_name_from_image_line(line)
		if name:
			names.add(name)
		# Skip the following points line if it exists
		if i + 1 < n and not lines[i + 1].lstrip().startswith('#'):
			i += 2
		else:
			i += 1
	return names


def filter_images_file(input_a: Path, keep_names: Set[str]) -> List[str]:
	"""Return filtered file content lines for inputA keeping only images with NAME in keep_names.

	- Preserves leading comment lines.
	- Updates 'Number of images:' line if present (first occurrence).
	- Keeps two-line blocks (image header + points line) for matching NAMEs.
	"""
	with input_a.open('r', encoding='utf-8', errors='ignore') as f:
		lines = f.readlines()

	# Separate header/comment block and content
	header: List[str] = []
	content_start = 0
	for idx, line in enumerate(lines):
		if not line.lstrip().startswith('#') and line.strip():
			content_start = idx
			break
		header.append(line)
	else:
		# No content, just return header
		return header

	# Parse through content two lines at a time, preserving pairs that match
	filtered_pairs: List[str] = []
	i = content_start
	n = len(lines)
	while i < n:
		line = lines[i]
		if not line.strip():
			i += 1
			continue
		if line.lstrip().startswith('#'):
			# mid-file comments are appended to header to preserve positioning
			header.append(line)
			i += 1
			continue

		name = parse_name_from_image_line(line)
		keep = name in keep_names

		if keep:
			filtered_pairs.append(line)
			# Append following points line if present and is not a comment
			if i + 1 < n and lines[i + 1].strip() and not lines[i + 1].lstrip().startswith('#'):
				filtered_pairs.append(lines[i + 1])
				i += 2
			else:
				i += 1
		else:
			# Skip this image header and its points line if present
			if i + 1 < n and lines[i + 1].strip() and not lines[i + 1].lstrip().startswith('#'):
				i += 2
			else:
				i += 1

	# Update 'Number of images:' in header if present (first occurrence)
	num_images = sum(1 for l in filtered_pairs if l.strip() and not l.lstrip().startswith('#')) // 2
	updated_header: List[str] = []
	replaced = False
	for hline in header:
		if not replaced and hline.lower().startswith('# number of images:'):
			updated_header.append(f"# Number of images: {num_images}\n")
			replaced = True
		else:
			updated_header.append(hline)
	if not replaced:
		# If header exists, append a count line after initial comment block
		if updated_header and updated_header[-1].endswith('\n'):
			updated_header.append(f"# Number of images: {num_images}\n")

	return updated_header + filtered_pairs


def main():
	ap = argparse.ArgumentParser(description='Filter COLMAP images.txt (two-line per image) by list from another file')
	ap.add_argument('--inputA', required=True, type=Path, help='Path to images.txt to filter')
	ap.add_argument('--inputB', required=True, type=Path, help='Path to images.txt whose NAMEs will be kept')
	ap.add_argument('--output', type=Path, help='Output path (default: inputA.filtered.txt)')
	args = ap.parse_args()

	if not args.inputA.exists():
		raise FileNotFoundError(f"inputA not found: {args.inputA}")
	if not args.inputB.exists():
		raise FileNotFoundError(f"inputB not found: {args.inputB}")

	keep_names = collect_names_from_images_file(args.inputB)
	print(f"Collected {len(keep_names)} NAMEs from {args.inputB}")

	filtered_lines = filter_images_file(args.inputA, keep_names)

	out_path = args.output or args.inputA.with_suffix(args.inputA.suffix + '.filtered.txt')
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open('w', encoding='utf-8') as f:
		f.writelines(filtered_lines)

	# Summary
	kept_images = sum(1 for l in filtered_lines if l.strip() and not l.lstrip().startswith('#')) // 2
	print(f"Wrote {kept_images} images to {out_path}")


if __name__ == '__main__':
	main()

