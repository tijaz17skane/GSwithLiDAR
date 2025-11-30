#!/usr/bin/env python3
"""
Normalize and visualize depth scale JSON produced by make_depth_scale.

Features:
 1. Plot per-image scale and offset values (global + per camera group) and highlight outliers.
 2. Detect outliers per camera (front/right/left/retro) using robust statistics (MAD or IQR).
 3. Replace outlier scale/offset values with imputed values (linear interpolation over temporal order when possible, else group median).
 4. Output a normalized JSON and save plots to a directory.

Usage:
  python normalize_depth_scale.py \
	  --depth_scales path/to/depth_params.json \
	  --plots_dir path/to/plots_out \
	  --normalized_json path/to/depth_params_normalized.json \
	  [--method mad --mad_thresh 3.5] [--iqr_mult 1.5] [--no_offsets]

Notes:
 - Image names must start with the camera prefix (front_, right_, left_, retro_)
 - Frame ordering inferred from last numeric token in name; if missing, ordering falls back to alphabetical.
 - Outlier replacement strives to keep continuity: if both neighbors are non-outliers, uses linear interpolation; otherwise median.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np

# Ensure a non-interactive backend for matplotlib in headless environments
_HAVE_MPL = False
try:
	# Prefer explicit Agg backend to avoid GUI blocking on servers without DISPLAY
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	_HAVE_MPL = True
except Exception:
	_HAVE_MPL = False

CAM_PREFIXES = ["front", "right", "left", "retro"]

def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--depth_scales", required=True, help="Input depth scales JSON produced by make_depth_scale.py")
	p.add_argument("--plots_dir", required=True, help="Directory to write plots (will be created)")
	p.add_argument("--normalized_json", required=True, help="Output JSON with normalized (imputed) scales/offsets")
	p.add_argument("--method", choices=["mad", "iqr"], default="mad", help="Outlier detection method")
	p.add_argument("--mad_thresh", type=float, default=3.5, help="MAD z-score threshold (if method=mad)")
	p.add_argument("--iqr_mult", type=float, default=1.5, help="IQR multiplier (if method=iqr)")
	p.add_argument("--no_offsets", action="store_true", help="Skip offset outlier processing (only scales)")
	p.add_argument("--min_valid", type=int, default=5, help="Minimum group size required for robust stats; else fallback to mean/std")
	# Add new arguments to allow selective processing of scales and offsets
	p.add_argument("--process_scales", action="store_true", help="Enable processing of scales")
	p.add_argument("--process_offsets", action="store_true", help="Enable processing of offsets")
	return p.parse_args()

def load_depth_scales(path: str) -> Dict[str, Dict[str, float]]:
	with open(path, 'r') as f:
		return json.load(f)

def extract_camera(name: str) -> str:
	first = name.split('_')[0].lower()
	return first if first in CAM_PREFIXES else 'unknown'

def extract_frame_index(name: str) -> int:
	# Look for last integer in name
	nums = re.findall(r"(\d+)", name)
	if nums:
		try:
			return int(nums[-1])
		except Exception:
			pass
	# fallback: hash to stable ordering
	return abs(hash(name)) % (10**9)

def group_by_camera(data: Dict[str, Dict[str, float]]):
	groups = {}
	for img, vals in data.items():
		cam = extract_camera(img)
		groups.setdefault(cam, []).append((img, vals))
	# sort within group by frame index
	for cam in groups:
		groups[cam].sort(key=lambda x: extract_frame_index(x[0]))
	return groups

def robust_outlier_mask(values: np.ndarray, method: str, mad_thresh: float, iqr_mult: float) -> np.ndarray:
	if values.size == 0:
		return np.zeros((0,), dtype=bool)
	v = values.astype(float)
	if method == 'mad':
		med = np.median(v)
		mad = np.median(np.abs(v - med))
		if mad < 1e-12:
			# fallback to std
			std = v.std() if v.size > 1 else 0.0
			if std < 1e-12:
				return np.zeros_like(v, dtype=bool)
			z = np.abs((v - v.mean()) / (std + 1e-12))
			return z > mad_thresh
		z = 0.6745 * (v - med) / (mad + 1e-12)  # normalized MAD z-score
		return np.abs(z) > mad_thresh
	else:  # iqr
		q1, q3 = np.percentile(v, [25, 75])
		iqr = q3 - q1
		if iqr < 1e-12:
			return np.zeros_like(v, dtype=bool)
		lower = q1 - iqr_mult * iqr
		upper = q3 + iqr_mult * iqr
		return (v < lower) | (v > upper)

def interpolate_or_median(index: int, non_outlier_indices: List[int], values: np.ndarray, group_median: float) -> float:
	# Try interpolation using nearest lower and upper non-outlier indices
	lower_candidates = [i for i in non_outlier_indices if i < index]
	upper_candidates = [i for i in non_outlier_indices if i > index]
	if lower_candidates and upper_candidates:
		lo = max(lower_candidates)
		hi = min(upper_candidates)
		# Linear interpolation by relative position between lo and hi
		t = (index - lo) / (hi - lo + 1e-12)
		return (1 - t) * values[lo] + t * values[hi]
	# else fallback to group median
	return group_median

def process_group(entries: List[Tuple[str, Dict[str, float]]], method: str, mad_thresh: float, iqr_mult: float, do_offsets: bool, min_valid: int, global_scale_mask_slice=None, global_offset_mask_slice=None):
	# entries sorted by frame index
	names = [e[0] for e in entries]
	scales = np.array([e[1]['scale'] for e in entries], dtype=float)
	offsets = np.array([e[1]['offset'] for e in entries], dtype=float) if do_offsets else None

	out_scale_mask = robust_outlier_mask(scales, method, mad_thresh, iqr_mult) if len(scales) >= min_valid else np.zeros_like(scales, dtype=bool)
	out_offset_mask = robust_outlier_mask(offsets, method, mad_thresh, iqr_mult) if (do_offsets and len(scales) >= min_valid) else (np.zeros_like(scales, dtype=bool) if do_offsets else None)
    
	# Combine with global mask if provided (global mask is same shape as global concatenated array,
	# but we accept a slice of same length as this group)
	if global_scale_mask_slice is not None:
		out_scale_mask = np.logical_or(out_scale_mask, np.asarray(global_scale_mask_slice, dtype=bool))
	if do_offsets and global_offset_mask_slice is not None:
		out_offset_mask = np.logical_or(out_offset_mask, np.asarray(global_offset_mask_slice, dtype=bool))

	scale_median = np.median(scales)
	offset_median = np.median(offsets) if do_offsets else None

	# Impute outliers
	scale_indices = list(range(len(scales)))
	non_out_scale_indices = [i for i in scale_indices if not out_scale_mask[i]]
	imputed_scales = scales.copy()
	for i in scale_indices:
		if out_scale_mask[i]:
			imputed_scales[i] = interpolate_or_median(i, non_out_scale_indices, scales, scale_median)

	if do_offsets:
		offset_indices = list(range(len(offsets)))
		non_out_offset_indices = [i for i in offset_indices if not out_offset_mask[i]]
		imputed_offsets = offsets.copy()
		for i in offset_indices:
			if out_offset_mask[i]:
				imputed_offsets[i] = interpolate_or_median(i, non_out_offset_indices, offsets, offset_median)
	else:
		imputed_offsets = offsets

	stats = {
		'num': len(scales),
		'scale_outliers': int(out_scale_mask.sum()),
		'offset_outliers': int(out_offset_mask.sum()) if do_offsets else 0,
		'scale_median': float(scale_median),
		'offset_median': float(offset_median) if do_offsets else None
	}

	# Build normalized entries list
	normalized = []
	for i, name in enumerate(names):
		normalized.append({
			'image_name': name,
			'scale': float(imputed_scales[i]),
			'offset': float(imputed_offsets[i]) if do_offsets else float(entries[i][1]['offset'])
		})

	return normalized, out_scale_mask, out_offset_mask, stats

def plot_group(cam: str, entries: List[Dict[str, float]], out_scale_mask: np.ndarray, out_offset_mask, plots_dir: str, do_offsets: bool):
	if not _HAVE_MPL:
		return
	os.makedirs(plots_dir, exist_ok=True)
	scales = [e['scale'] for e in entries]
	offsets = [e['offset'] for e in entries]
	x = list(range(len(entries)))
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(x, scales, label='scale', color='tab:blue')
	ax.scatter([i for i in x if out_scale_mask[i]], [scales[i] for i in x if out_scale_mask[i]], color='red', label='scale outlier', zorder=3)
	ax.set_title(f"Scales - {cam}")
	ax.set_xlabel('index (sorted by frame)')
	ax.set_ylabel('scale')
	ax.legend()
	fig.tight_layout()
	fig.savefig(os.path.join(plots_dir, f"{cam}_scales.png"))
	plt.close(fig)
	if do_offsets:
		fig2, ax2 = plt.subplots(figsize=(10,4))
		ax2.plot(x, offsets, label='offset', color='tab:green')
		ax2.scatter([i for i in x if out_offset_mask[i]], [offsets[i] for i in x if out_offset_mask[i]], color='orange', label='offset outlier', zorder=3)
		ax2.set_title(f"Offsets - {cam}")
		ax2.set_xlabel('index (sorted by frame)')
		ax2.set_ylabel('offset')
		ax2.legend()
		fig2.tight_layout()
		fig2.savefig(os.path.join(plots_dir, f"{cam}_offsets.png"))
		plt.close(fig2)

def plot_global(all_entries: List[Dict[str, float]], plots_dir: str, do_offsets: bool):
	if not _HAVE_MPL:
		return
	os.makedirs(plots_dir, exist_ok=True)
	cams = [extract_camera(e['image_name']) for e in all_entries]
	scales = [e['scale'] for e in all_entries]
	offsets = [e['offset'] for e in all_entries]
	x = list(range(len(all_entries)))
	colors = {"front":"tab:blue","right":"tab:green","left":"tab:purple","retro":"tab:brown","unknown":"gray"}
	fig, ax = plt.subplots(figsize=(12,5))
	for i, s in enumerate(scales):
		ax.scatter(i, s, color=colors.get(cams[i], 'gray'), s=12)
	ax.set_title('Global Scales by Camera')
	ax.set_xlabel('image index')
	ax.set_ylabel('scale')
	fig.tight_layout()
	fig.savefig(os.path.join(plots_dir, 'global_scales.png'))
	plt.close(fig)
	if do_offsets:
		fig2, ax2 = plt.subplots(figsize=(12,5))
		for i, off in enumerate(offsets):
			ax2.scatter(i, off, color=colors.get(cams[i], 'gray'), s=12)
		ax2.set_title('Global Offsets by Camera')
		ax2.set_xlabel('image index')
		ax2.set_ylabel('offset')
		fig2.tight_layout()
		fig2.savefig(os.path.join(plots_dir, 'global_offsets.png'))
		plt.close(fig2)

def main():
	args = parse_args()
	data = load_depth_scales(args.depth_scales)
	groups = group_by_camera(data)
	# Update the main function to use the new arguments
	do_scales = args.process_scales
	do_offsets = args.process_offsets

	if not do_scales and not do_offsets:
	    print("Error: At least one of --process_scales or --process_offsets must be specified.")
	    return

	do_offsets = do_offsets and not args.no_offsets  # Ensure --no_offsets is respected

	# Build global arrays in deterministic camera order (CAM_PREFIXES then unknown)
	ordered_cameras = [c for c in CAM_PREFIXES if c in groups] + [c for c in groups.keys() if c not in CAM_PREFIXES]
	all_entries_ordered = []
	for cam in ordered_cameras:
		all_entries_ordered.extend(groups[cam])

	global_scales = np.array([e[1]['scale'] for e in all_entries_ordered], dtype=float)
	global_offsets = np.array([e[1]['offset'] for e in all_entries_ordered], dtype=float)

	global_scale_mask = robust_outlier_mask(global_scales, args.method, args.mad_thresh, args.iqr_mult) if len(global_scales) >= args.min_valid else np.zeros_like(global_scales, dtype=bool)
	global_offset_mask = robust_outlier_mask(global_offsets, args.method, args.mad_thresh, args.iqr_mult) if (do_offsets and len(global_offsets) >= args.min_valid) else (np.zeros_like(global_offsets, dtype=bool) if do_offsets else None)
	print(f"Global: total_images={len(global_scales)} global_scale_outliers={int(global_scale_mask.sum())} global_offset_outliers={int(global_offset_mask.sum()) if global_offset_mask is not None else 0}")

	all_normalized = []
	summary = {}
	# Index into global arrays
	cursor = 0
	for cam in ordered_cameras:
		entries = groups[cam]
		n = len(entries)
		gscale_slice = global_scale_mask[cursor:cursor+n] if global_scale_mask is not None else None
		goffset_slice = global_offset_mask[cursor:cursor+n] if do_offsets and global_offset_mask is not None else None
		normalized, out_scale_mask, out_offset_mask, stats = process_group(entries, args.method, args.mad_thresh, args.iqr_mult, do_offsets, args.min_valid, global_scale_mask_slice=gscale_slice, global_offset_mask_slice=goffset_slice)
		cursor += n
		summary[cam] = stats
		plot_group(cam, normalized, out_scale_mask, out_offset_mask, args.plots_dir, do_offsets)
		all_normalized.extend(normalized)
	# global plots
	plot_global(all_normalized, args.plots_dir, do_offsets)
	# Write normalized JSON keyed by original image basename (without enforced suffix removal here)
	normalized_dict = {e['image_name']: {'scale': e['scale'], 'offset': e['offset']} for e in all_normalized}
	with open(args.normalized_json, 'w') as f:
		json.dump(normalized_dict, f, indent=2)
	# Print summary
	print("Normalization summary:")
	for cam, stats in summary.items():
		print(f"  Camera={cam}: N={stats['num']} scale_outliers={stats['scale_outliers']} offset_outliers={stats['offset_outliers']} scale_median={stats['scale_median']:.6f}" + (f" offset_median={stats['offset_median']:.6f}" if stats['offset_median'] is not None else ""))
	if not _HAVE_MPL:
		print("[WARN] matplotlib not available; plots skipped.")

if __name__ == '__main__':
	main()

