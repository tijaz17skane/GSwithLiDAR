#!/usr/bin/env python3
"""
Filter and invert DA3 depth outputs.

Usage:
  python filter_da3_depths.py --depths_input /path/to/depths --output_dir /path/to/out

What it does:
 - Reads depth files from --depths_input (supports PNG, TIFF, NPY, NPZ).
 - Removes the substring "_depth_monocular_vis" from filenames when writing outputs.
 - Writes the cleaned original depths as 16-bit PNGs into <output_dir>/depths/
 - Writes inverted depths into <output_dir>/inverted_depths/.

Inversion rule:
 For each valid pixel (finite and >0), inverted = d_min + d_max - d. This flips near/far
 while preserving the numeric range. Invalid/zero pixels are kept as zero.
"""

import argparse
from pathlib import Path
import numpy as np
import imageio
import sys


def read_depth(path: Path):
    """Read depth data from supported file types.

    Returns: numpy array (H,W) as float32 and original dtype
    """
    suffix = path.suffix.lower()
    if suffix in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
        arr = imageio.imread(str(path))
        # If RGB, convert to luminance by taking first channel
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32), arr.dtype
    elif suffix == ".npy":
        arr = np.load(str(path))
        return arr.astype(np.float32), arr.dtype
    elif suffix == ".npz":
        data = np.load(str(path))
        # pick first array in archive
        if len(data.files) == 0:
            raise ValueError(f"Empty npz: {path}")
        arr = data[data.files[0]]
        return arr.astype(np.float32), arr.dtype
    else:
        raise ValueError(f"Unsupported file type: {path}")


def write_png_uint16(path: Path, arr_float: np.ndarray):
    """Write array to 16-bit PNG. arr_float expected as float32 in range 0..65535 or scaled appropriately.
    We'll clip and cast to uint16.
    """
    arr = np.asarray(arr_float)
    if arr.ndim != 2:
        raise ValueError("Only 2D arrays supported for PNG output")
    out = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, 0, 65535).astype(np.uint16)
    imageio.imwrite(str(path), out)


def process_folder(depths_input: Path, output_dir: Path, remove_substr: str = "_depth_monocular_vis"):
    depths_input = Path(depths_input)
    output_dir = Path(output_dir)
    if not depths_input.exists():
        raise SystemExit(f"Input folder does not exist: {depths_input}")

    out_depths = output_dir / "depths"
    out_inverted = output_dir / "inverted_depths"
    out_depths.mkdir(parents=True, exist_ok=True)
    out_inverted.mkdir(parents=True, exist_ok=True)

    exts = ("*.png", "*.tif", "*.tiff", "*.npy", "*.npz", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in exts:
        files.extend(sorted(depths_input.glob(ext)))

    if not files:
        print(f"No depth files found in {depths_input}")
        return

    for p in files:
        try:
            arr, orig_dtype = read_depth(p)
        except Exception as e:
            print(f"Skipping {p.name}: read error: {e}")
            continue

        # Determine cleaned name
        stem = p.stem
        # If input was .npz or .npy, p.stem may include extra suffix like 'name_depth', but remove target substring
        clean_stem = stem.replace(remove_substr, "")

        # Compute valid mask
        valid = np.isfinite(arr) & (arr > 0)
        if valid.any():
            vmin = float(np.min(arr[valid]))
            vmax = float(np.max(arr[valid]))
        else:
            vmin = 0.0
            vmax = 0.0

        # Save original depths as 16-bit PNG normalized to 0..65535 using (arr - vmin)/(vmax-vmin)
        if vmax > vmin:
            norm = np.zeros_like(arr, dtype=np.float32)
            norm[valid] = (arr[valid] - vmin) / (vmax - vmin) * 65535.0
        else:
            norm = np.zeros_like(arr, dtype=np.float32)

        out_path = out_depths / f"{clean_stem}.png"
        try:
            write_png_uint16(out_path, norm)
        except Exception as e:
            print(f"Failed to write {out_path}: {e}")
            continue

        # Invert depths: inverted = vmin + vmax - arr for valid pixels; then normalize same way
        if vmax > vmin and valid.any():
            inv = np.zeros_like(arr, dtype=np.float32)
            inv[valid] = (vmin + vmax - arr[valid])
            # Normalize inverted to 0..65535
            inv_norm = np.zeros_like(inv, dtype=np.float32)
            inv_valid = valid
            inv_vmin = float(np.min(inv[inv_valid]))
            inv_vmax = float(np.max(inv[inv_valid]))
            if inv_vmax > inv_vmin:
                inv_norm[inv_valid] = (inv[inv_valid] - inv_vmin) / (inv_vmax - inv_vmin) * 65535.0
            else:
                inv_norm = np.zeros_like(inv, dtype=np.float32)
        else:
            inv_norm = np.zeros_like(arr, dtype=np.float32)

        out_inv_path = out_inverted / f"{clean_stem}.png"
        try:
            write_png_uint16(out_inv_path, inv_norm)
        except Exception as e:
            print(f"Failed to write {out_inv_path}: {e}")
            continue

        print(f"Wrote: {out_path.name} and {out_inv_path.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Filter and invert DA3 depth outputs")
    parser.add_argument("--depths_input", required=True, help="Folder containing depth outputs from DA3")
    parser.add_argument("--output_dir", required=True, help="Output folder to write depths and inverted_depths")
    parser.add_argument("--remove-substr", default="_depth_monocular_vis", help="Substring to remove from filenames")
    args = parser.parse_args(argv)

    process_folder(Path(args.depths_input), Path(args.output_dir), remove_substr=args.remove_substr)


if __name__ == "__main__":
    main()
