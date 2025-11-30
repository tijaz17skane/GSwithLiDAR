#!/usr/bin/env python3
"""
Scan a directory tree for COLMAP `images.txt` files (typically at
`<dataset>/sparse/0/images.txt`). For each found file, go to the dataset
root and compare the `images.txt` entries against the files in
`<dataset>/images/`. Any image entry not present in that folder is removed.

By default the script writes a new file named `images2.txt` next to the
original `images.txt`. Use `--overwrite` to replace the original file.

Usage:
  python remove_extras_from_images.py --input_dir /path/to/root [--overwrite] [--output-name images2.txt]

The script preserves header comment lines (starting with `#`) and the
two-line-per-image layout in COLMAP `images.txt` (IMAGE_LINE + POINTS2D_LINE).
"""

from pathlib import Path
import argparse
import sys
import os
from typing import List, Tuple


class ImageEntry:
    def __init__(self, image_line: str, points_line: str):
        self.image_line = image_line.rstrip('\n')
        self.points_line = points_line.rstrip('\n')
        self.parse_image_line()

    def parse_image_line(self):
        toks = self.image_line.strip().split()
        if len(toks) < 10:
            raise ValueError(f"Unexpected image line format: '{self.image_line}'")
        self.image_id = int(toks[0])
        # tokens 1-4: qw qx qy qz
        # tokens 5-7: tx ty tz
        # token 8: camera_id
        # token 9..end: name (may contain spaces)
        self.name = " ".join(toks[9:])


def read_images_file(path: Path) -> Tuple[List[str], List[ImageEntry]]:
    header_lines: List[str] = []
    entries: List[ImageEntry] = []
    with path.open('r') as f:
        lines = f.readlines()

    i = 0
    # preserve leading comment lines
    while i < len(lines) and lines[i].strip().startswith('#'):
        header_lines.append(lines[i].rstrip('\n'))
        i += 1

    # If there's a single blank line separating header from data (common), skip it
    if i < len(lines) and lines[i].strip() == '':
        i += 1

    remaining = lines[i:]
    # Ensure an even number of lines so we can pair IMAGE_LINE + POINTS2D_LINE.
    if len(remaining) % 2 == 1:
        remaining.append('')

    for j in range(0, len(remaining), 2):
        img_line = remaining[j]
        pts_line = remaining[j+1] if j+1 < len(remaining) else ''
        entries.append(ImageEntry(img_line, pts_line))

    return header_lines, entries


def write_images_file(path: Path, header_lines: List[str], entries: List[ImageEntry]):
    new_header = []
    written_count = len(entries)
    for h in header_lines:
        if h.strip().lower().startswith('# number of images:'):
            new_header.append(f"# Number of images: {written_count}")
        else:
            new_header.append(h)
    with path.open('w') as f:
        for h in new_header:
            f.write(h + '\n')
        f.write('\n')
        for e in entries:
            f.write(e.image_line + '\n')
            f.write(e.points_line + '\n')


def find_dataset_root_for_images_txt(images_txt_path: Path) -> Path:
    """
    Given a path to images.txt, try to determine the dataset root folder such
    that the images folder is at <root>/images. Typical layout is
    <root>/sparse/0/images.txt -> images_txt_path.parents[2] is <root>.
    This function is defensive and tries to find an ancestor named 'sparse'
    and return its parent; otherwise falls back to going up two directories.
    """
    p = images_txt_path
    # common case: .../<root>/sparse/0/images.txt
    if p.parent.name == '0' and p.parent.parent.name == 'sparse':
        return p.parent.parent.parent

    # try to find an ancestor named 'sparse'
    for anc in p.parents:
        if anc.name == 'sparse':
            return anc.parent

    # fallback: go up two levels from images.txt
    if len(p.parents) >= 2:
        return p.parents[1]

    # otherwise return parent
    return p.parent


def collect_images_in_folder(images_dir: Path) -> set:
    files = []
    if not images_dir.exists() or not images_dir.is_dir():
        return set()
    for f in images_dir.iterdir():
        if f.is_file():
            files.append(f.name.lower())
    return set(files)


def process_images_txt(images_txt_path: Path, overwrite: bool = False, output_name: str = 'images2.txt', dry_run: bool = False):
    print(f"Processing: {images_txt_path}")
    try:
        headers, entries = read_images_file(images_txt_path)
    except Exception as e:
        print(f"  ERROR reading {images_txt_path}: {e}")
        return

    root = find_dataset_root_for_images_txt(images_txt_path)
    images_dir = root / 'images'
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"  WARNING: images folder not found at expected location: {images_dir}. Skipping.")
        return

    avail = collect_images_in_folder(images_dir)
    if len(avail) == 0:
        print(f"  WARNING: no files found in images folder: {images_dir}. Skipping.")
        return

    kept = []
    removed = []
    for e in entries:
        base = os.path.basename(e.name).lower()
        if base in avail:
            kept.append(e)
        else:
            removed.append(e)

    print(f"  Found {len(entries)} entries, keeping {len(kept)}, removing {len(removed)}")
    #if len(removed) > 0:
        #print("  Removed image names:")
        #for r in removed:
        #    print(f"    {r.name}")

    out_path = images_txt_path if overwrite else images_txt_path.parent / output_name
    if dry_run:
        print(f"  Dry-run: would write {len(kept)} entries to {out_path}")
        return

    write_images_file(out_path, headers, kept)
    if overwrite:
        print(f"  Overwrote original images.txt at {images_txt_path}")
    else:
        print(f"  Wrote filtered images to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Remove entries from images.txt that are not present in the corresponding images/ folder')
    parser.add_argument('--input_dir', required=True, help='Root directory to scan recursively for images.txt')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite original images.txt files')
    parser.add_argument('--output-name', default='images2.txt', help='Output filename when not overwriting (default: images2.txt)')
    parser.add_argument('--dry-run', action='store_true', help="Don't write files, only report what would change")
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        print(f"Input directory does not exist: {root}")
        sys.exit(1)

    matches = list(root.rglob('images.txt'))
    if len(matches) == 0:
        print(f"No images.txt files found under {root}")
        return

    print(f"Found {len(matches)} images.txt files. Scanning and filtering...")
    for p in matches:
        process_images_txt(p, overwrite=args.overwrite, output_name=args.output_name, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
