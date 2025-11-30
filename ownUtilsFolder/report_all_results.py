#!/usr/bin/env python3
"""
Aggregate COLMAP/experiment results from results_test.json and results_train.json.

This script scans a directory tree for results_test.json and results_train.json files,
extracts metrics along with the experiment name (the parent folder name of each file),
and writes aggregated outputs in both JSON and CSV formats.

Usage:
  python report_all_results.py --dir <root_dir> --all_results <out_dir_or_prefix>

If --all_results is a directory path (existing or ending with '/'), the script will
write:
  <out_dir>/all_results.json
  <out_dir>/all_results.csv

If it's a file prefix, it will write:
  <prefix>.json and <prefix>.csv
"""

from pathlib import Path
import argparse
import json
import csv
import os
from typing import List, Dict, Any


def find_results_files(root_dir: str) -> Dict[str, List[Path]]:
	root = Path(root_dir)
	return {
		'test': list(root.rglob('results_test.json')),
		'train': list(root.rglob('results_train.json')),
	}


def extract_experiment_name(file_path: Path) -> str:
	# Experiment name is the parent directory containing the results file
	return file_path.parent.name


def load_results(file_path: Path, kind: str) -> List[Dict[str, Any]]:
	# kind is 'test' or 'train'
	with open(file_path, 'r') as f:
		data = json.load(f)

	exp_name = extract_experiment_name(file_path)
	rows: List[Dict[str, Any]] = []
	for method_name, metrics in data.items():
		row: Dict[str, Any] = {
			'experiment_name': exp_name,
			'result_type': kind,
			'method': method_name,
		}
		# Copy over metrics as-is; typical keys: SSIM, PSNR, LPIPS
		if isinstance(metrics, dict):
			for k, v in metrics.items():
				row[k] = v
		rows.append(row)
	return rows


def aggregate(root_dir: str) -> List[Dict[str, Any]]:
	files = find_results_files(root_dir)
	print(f"Found {len(files['test'])} results_test.json files")
	print(f"Found {len(files['train'])} results_train.json files")

	all_rows: List[Dict[str, Any]] = []
	# Preserve discovery order
	for f in files['test']:
		all_rows.extend(load_results(f, 'test'))
	for f in files['train']:
		all_rows.extend(load_results(f, 'train'))
	return all_rows


def save_json(rows: List[Dict[str, Any]], out_path: str) -> None:
	with open(out_path, 'w') as f:
		json.dump(rows, f, indent=2)
	print(f"Saved JSON results to: {out_path}")


def save_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
	if not rows:
		print("No records to write to CSV")
		return
	# Minimal, ordered columns first; include metrics if present
	base_cols = ['experiment_name', 'result_type', 'method', 'SSIM', 'PSNR', 'LPIPS']
	# Determine any extra columns encountered
	extra_cols: List[str] = []
	for r in rows:
		for k in r.keys():
			if k not in base_cols and k not in extra_cols:
				extra_cols.append(k)
	fieldnames = base_cols + extra_cols

	with open(out_path, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
	print(f"Saved CSV results to: {out_path}")


def resolve_output_paths(all_results: str) -> (str, str):
	# If directory or endswith '/', write named files inside; else use as prefix
	if all_results.endswith('/') or os.path.isdir(all_results):
		os.makedirs(all_results, exist_ok=True)
		json_out = os.path.join(all_results, 'all_results.json')
		csv_out = os.path.join(all_results, 'all_results.csv')
	else:
		json_out = f"{all_results}.json"
		csv_out = f"{all_results}.csv"
	return json_out, csv_out


def main():
	ap = argparse.ArgumentParser(description='Aggregate results_test.json and results_train.json into JSON and CSV')
	ap.add_argument('--dir', required=True, help='Root directory to search for results files')
	ap.add_argument('--all_results', '--all-results', dest='all_results', required=True, help='Output directory or file prefix for aggregated results')
	args = ap.parse_args()

	if not os.path.isdir(args.dir):
		print(f"Error: Directory not found: {args.dir}")
		return

	print(f"Scanning: {args.dir}")
	rows = aggregate(args.dir)
	if not rows:
		print("No results found.")
		return

	json_out, csv_out = resolve_output_paths(args.all_results)
	save_json(rows, json_out)
	save_csv(rows, csv_out)
	print('Done.')


if __name__ == '__main__':
	main()

