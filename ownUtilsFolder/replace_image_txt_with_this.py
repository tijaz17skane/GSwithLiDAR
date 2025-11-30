#!/usr/bin/env python3
"""
Replace all `images.txt` files found under `--input_dir` with the contents
of `--source_images_txt`.

By default the script will overwrite each found `images.txt`. Use
`--dry-run` to only print what would be replaced.

Usage:
  python replace_image_txt_with_this.py --input_dir /path/to/root --source_images_txt /path/to/source/images.txt [--dry-run]
"""

from pathlib import Path
import argparse
import sys
import shutil


def load_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Source images.txt not found: {path}")
    return path.read_text()


def replace_file(target: Path, content: str):
    # write to a temp file then atomically replace
    tmp = target.with_suffix('.tmp')
    tmp.write_text(content)
    # preserve mode if exists
    try:
        shutil.copymode(target, tmp)
    except Exception:
        pass
    tmp.replace(target)


def main():
    parser = argparse.ArgumentParser(description='Replace images.txt files under a directory with a source images.txt')
    parser.add_argument('--input_dir', required=True, help='Directory to scan recursively for images.txt')
    parser.add_argument('--source_images_txt', required=True, help='Path to the source images.txt to copy into each found file')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without writing files')
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists() or not root.is_dir():
        print(f"Input directory does not exist or is not a directory: {root}")
        sys.exit(1)

    source_path = Path(args.source_images_txt)
    try:
        source_text = load_source(source_path)
    except Exception as e:
        print(f"Error reading source images.txt: {e}")
        sys.exit(1)

    matches = list(root.rglob('images.txt'))
    if not matches:
        print(f"No images.txt files found under {root}")
        return

    print(f"Found {len(matches)} images.txt files under {root}")
    replaced = 0
    skipped = 0
    for p in matches:
        # Avoid accidentally replacing the source file if it's inside the tree
        if p.resolve() == source_path.resolve():
            print(f"Skipping source file itself: {p}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"DRY-RUN: would replace {p} with contents of {source_path}")
            replaced += 1
            continue

        try:
            replace_file(p, source_text)
            print(f"Replaced {p}")
            replaced += 1
        except Exception as e:
            print(f"Failed to replace {p}: {e}")
            skipped += 1

    print(f"Done. Replaced: {replaced}, Skipped/Failed: {skipped}")


if __name__ == '__main__':
    main()
