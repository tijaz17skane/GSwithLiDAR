#!/usr/bin/env python3
"""
Downsample a COLMAP points3D.txt by an approximate factor using voxel-grid random selection.

Input format expected (header + points):
  # 3D point list with one line of data per point:
  #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
  # Number of points: N
  POINT3D_ID X Y Z R G B ERROR [TRACK...]

We ignore TRACK and only require the first 8 columns. Output keeps the same format and
rewrites POINT3D_ID sequentially (unless --preserve-ids is set).

The algorithm adapts the voxel size via binary search so that the number of selected voxels
is close to N / factor. Within each voxel, one point is chosen at random.
"""

from pathlib import Path
import argparse
import random
import math
import sys


def parse_args():
    p = argparse.ArgumentParser(description='Voxel-based random downsampling of COLMAP points3D.txt')
    p.add_argument('--input_txt', required=True, type=str, help='Path to input points3D.txt')
    p.add_argument('--output_txt', required=True, type=str, help='Path to output downsampled points3D.txt')
    p.add_argument('--factor', required=True, type=float, help='Downsampling factor: 2 -> half, 32 -> 1/32, etc.')
    p.add_argument('--seed', type=int, default=42, help='Random seed for voxel random picks')
    p.add_argument('--preserve-ids', action='store_true', help='Preserve original POINT3D_IDs (default: reindex 1..M)')
    p.add_argument('--max-iters', type=int, default=22, help='Max binary-search iterations for voxel size')
    p.add_argument('--tolerance', type=float, default=0.05, help='Relative tolerance for target size (e.g., 0.05=5%)')
    return p.parse_args()


def read_points3D(path: Path):
    ids = []
    xs, ys, zs = [], [], []
    rs, gs, bs = [], [], []
    errs = []
    header_lines = []

    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith('#'):
                header_lines.append(line.rstrip('\n'))
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                # skip malformed rows
                continue
            try:
                ids.append(int(parts[0]))
                xs.append(float(parts[1])); ys.append(float(parts[2])); zs.append(float(parts[3]))
                rs.append(int(float(parts[4]))); gs.append(int(float(parts[5]))); bs.append(int(float(parts[6])))
                errs.append(float(parts[7]))
            except Exception:
                # skip rows that fail to parse
                continue

    return {
        'ids': ids,
        'x': xs, 'y': ys, 'z': zs,
        'r': rs, 'g': gs, 'b': bs,
        'err': errs,
        'header': header_lines,
    }


def compute_bbox(xs, ys, zs):
    xmin = min(xs); xmax = max(xs)
    ymin = min(ys); ymax = max(ys)
    zmin = min(zs); zmax = max(zs)
    return (xmin, ymin, zmin), (xmax, ymax, zmax)


def select_indices_with_voxel_size(xs, ys, zs, voxel_size, seed):
    # Hash points into voxels; keep one random index per voxel
    random.seed(seed)
    (xmin, ymin, zmin), _ = compute_bbox(xs, ys, zs)
    chosen = {}
    for idx, (x, y, z) in enumerate(zip(xs, ys, zs)):
        vx = math.floor((x - xmin) / voxel_size)
        vy = math.floor((y - ymin) / voxel_size)
        vz = math.floor((z - zmin) / voxel_size)
        key = (vx, vy, vz)
        if key not in chosen:
            chosen[key] = idx
        else:
            # replace with small probability to keep randomness
            if random.random() < 0.5:
                chosen[key] = idx
    return list(chosen.values())


def adaptive_voxel_selection(xs, ys, zs, factor, seed=42, max_iters=22):
    n = len(xs)
    target = max(1, int(round(n / max(factor, 1.0))))
    if target >= n:
        return list(range(n))

    # Initial bounds for voxel size: tiny -> ~n points, huge -> ~1 point
    (xmin, ymin, zmin), (xmax, ymax, zmax) = compute_bbox(xs, ys, zs)
    max_dim = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if max_dim <= 0:
        # Degenerate: random sample to target
        idxs = list(range(n))
        random.Random(seed).shuffle(idxs)
        return sorted(idxs[:target])

    low = max_dim / (n * 10.0)  # very small
    high = max_dim * 2.0         # very large

    best = None
    best_diff = n
    for _ in range(max_iters):
        mid = (low + high) / 2.0
        sel = select_indices_with_voxel_size(xs, ys, zs, mid, seed)
        cnt = len(sel)
        diff = abs(cnt - target)
        if diff < best_diff:
            best = sel
            best_diff = diff
        if cnt > target:
            # too many -> increase voxel size
            low = mid
        elif cnt < target:
            # too few -> decrease voxel size
            high = mid
        else:
            best = sel
            break

    # Ensure deterministic order in output
    return sorted(best if best is not None else [])


def write_points3D(path: Path, sel_idx, data, preserve_ids=False):
    xs, ys, zs = data['x'], data['y'], data['z']
    rs, gs, bs = data['r'], data['g'], data['b']
    errs = data['err']
    ids = data['ids']
    n_out = len(sel_idx)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write(f'# Number of points: {n_out}\n')
        pid = 1
        for i in sel_idx:
            point_id = ids[i] if preserve_ids else pid
            x, y, z = xs[i], ys[i], zs[i]
            r, g, b = rs[i], gs[i], bs[i]
            err = errs[i]
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {err:.6f}\n")
            pid += 1


def main():
    args = parse_args()
    in_path = Path(args.input_txt)
    out_path = Path(args.output_txt)

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    data = read_points3D(in_path)
    n = len(data['x'])
    if n == 0:
        print('No valid points parsed from input.', file=sys.stderr)
        sys.exit(2)

    sel_idx = adaptive_voxel_selection(data['x'], data['y'], data['z'], factor=max(args.factor, 1.0), seed=args.seed, max_iters=args.max_iters)
    write_points3D(out_path, sel_idx, data, preserve_ids=args.preserve_ids)
    print(f"Downsampled {n} -> {len(sel_idx)} points (factor target {args.factor}). Wrote: {out_path}")


if __name__ == '__main__':
    main()
