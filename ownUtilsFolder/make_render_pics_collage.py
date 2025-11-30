#!/usr/bin/env python3
"""
Create a vertical 4-image collage from images in a directory.

Usage:
  python make_render_pics_collage.py --dir /path/to/dir --outdir /path/to/outdir --border 20 --outname collage.png

Options:
  --dir      Directory containing exactly 4 images (will pick common image extensions and sort by name)
  --outdir   Directory where the collage will be written (created if missing)
  --border   Border width in pixels to leave around each image (default: 10)
  --outname  Output filename (default: collage.png)

The script arranges the four images top-to-bottom in a single column, centers them horizontally,
and leaves a white border around each image of the requested width.
"""
import os
import argparse
from PIL import Image, ImageOps


def make_collage(input_dir, out_dir, border=10, out_name='collage1.png', orientation='horizontal'):
    assert os.path.isdir(input_dir), f"Input dir not found: {input_dir}"
    os.makedirs(out_dir, exist_ok=True)

    # collect image files
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in exts]
    files = sorted(files)
    if len(files) != 4:
        raise SystemExit(f"Expected exactly 4 images in {input_dir}, found {len(files)}: {files}")

    imgs = [Image.open(os.path.join(input_dir, f)).convert('RGB') for f in files]

    # compute canvas size and placement depending on orientation
    max_w = max(im.width for im in imgs)
    max_h = max(im.height for im in imgs)
    total_w = sum(im.width for im in imgs)
    total_h = sum(im.height for im in imgs)

    if orientation == 'vertical':
        canvas_w = max_w + 2 * border
        canvas_h = total_h + 2 * border * len(imgs)
    else:
        # horizontal
        canvas_h = max_h + 2 * border
        canvas_w = total_w + 2 * border * len(imgs)

    # create white background
    collage = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

    if orientation == 'vertical':
        y = 0
        for im in imgs:
            # position x so image is centered horizontally inside canvas minus borders
            x = border + (max_w - im.width) // 2
            y += border
            collage.paste(im, (x, y))
            y += im.height
    else:
        x = 0
        for im in imgs:
            # position y so image is centered vertically inside canvas minus borders
            y = border + (max_h - im.height) // 2
            x += border
            collage.paste(im, (x, y))
            x += im.width

    out_path = os.path.join(out_dir, out_name)
    # If target exists, avoid overwriting by creating a numbered copy
    def _unique_path(path):
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        i = 1
        while True:
            candidate = f"{base}_copy{i}{ext}"
            if not os.path.exists(candidate):
                return candidate
            i += 1

    final_out = _unique_path(out_path)
    collage.save(final_out)
    if final_out != out_path:
        print(f"Destination exists, saved as copy: {final_out}")
    else:
        print(f"Saved collage to: {out_path}")


def main():
    p = argparse.ArgumentParser(description='Make a 4-image collage with white borders')
    p.add_argument('--dir', required=True, help='Input directory containing 4 images')
    p.add_argument('--outdir', required=True, help='Output directory to save collage')
    p.add_argument('--border', type=int, default=10, help='Border width in pixels around each image')
    p.add_argument('--outname', default='collage.png', help='Output filename')
    p.add_argument('--orientation', choices=['horizontal', 'vertical'], default='horizontal',
                   help='Layout orientation: horizontal (default) places images left-to-right; vertical places top-to-bottom')
    args = p.parse_args()

    make_collage(args.dir, args.outdir, border=args.border, out_name=args.outname, orientation=args.orientation)


if __name__ == '__main__':
    main()
