#!/usr/bin/env python3
"""
Filter a COLMAP-style points3D.txt file by removing an adjustable amount of 3D points.

used distance outlier python remove_extra_points3D.py --input_txt /mnt/data/tijaz/data3/section_3Lidar/22_Manual/sparse/model/triangulated/points3D.txt --output_ply /mnt/data/tijaz/data3/section_3Lidar/22_Manual/sparse/model/triangulated/points3D.ply --strategy distance_outlier --outlier_sigma 7
on section_3. Made it better manually

"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import List, Tuple


def parse_args():
    ap = argparse.ArgumentParser(description="Filter COLMAP points3D.txt by removing a subset of points")
    ap.add_argument('--input_txt', type=Path, help='Input points3D.txt file')
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument('--remove_pct', type=float, help='Percentage of points to remove (0-100)')
    mode.add_argument('--remove_n', type=int, help='Number of points to remove')
    mode.add_argument('--keep_n', type=int, help='Number of points to keep')
    ap.add_argument('--strategy', choices=['error_outlier','distance_outlier','combined_outlier'], required=True, help='Point removal strategy')
    ap.add_argument('--outlier_sigma', type=float, default=3.0, help='Sigma threshold for outlier strategies (default: 3.0)')
    ap.add_argument('--output_txt', type=Path, help='Output filtered points3D filename')
    ap.add_argument('--output_ply', type=Path, help='Optional PLY export of kept points')
    ap.add_argument('--dry_run', action='store_true', help='Only report intended changes')
    return ap.parse_args()


class PointRecord:
    __slots__ = ('id','x','y','z','r','g','b','error','track_pairs','raw_line')
    def __init__(self, id_:int,x:float,y:float,z:float,r:int,g:int,b:int,error:float,track_pairs:List[Tuple[int,int]],raw_line:str):
        self.id=id_; self.x=x; self.y=y; self.z=z; self.r=r; self.g=g; self.b=b; self.error=error; self.track_pairs=track_pairs; self.raw_line=raw_line


def read_points3d(path:Path)->Tuple[List[str],List[PointRecord]]:
    header_lines=[]
    points=[]
    with path.open('r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            stripped=line.strip()
            if stripped.startswith('#'):
                header_lines.append(line.rstrip('\n'))
                continue
            if not stripped:
                continue
            parts=stripped.split()
            if len(parts)<8:
                continue
            pid=int(parts[0]); x=float(parts[1]); y=float(parts[2]); z=float(parts[3]); r=int(parts[4]); g=int(parts[5]); b=int(parts[6]); err=float(parts[7])
            remaining=parts[8:]
            track_pairs=[]
            # TRACK[] is list of IMAGE_ID POINT2D_IDX pairs
            if remaining:
                if len(remaining)%2!=0:
                    # If odd number remaining, drop last stray
                    remaining=remaining[:-1]
                for i in range(0,len(remaining),2):
                    try:
                        img_id=int(remaining[i]); p2d_id=int(remaining[i+1])
                        track_pairs.append((img_id,p2d_id))
                    except ValueError:
                        continue
            points.append(PointRecord(pid,x,y,z,r,g,b,err,track_pairs,line.rstrip('\n')))
    return header_lines, points


def compute_mean_track_length(points:List[PointRecord])->float:
    if not points:
        return 0.0
    total=sum(len(p.track_pairs) for p in points)
    return total/len(points)


def compute_statistics(points:List[PointRecord])->dict:
    """Compute error and spatial statistics for outlier detection."""
    if not points:
        return {}
    n=len(points)
    errors=[p.error for p in points]
    mean_err=sum(errors)/n
    std_err=(sum((e-mean_err)**2 for e in errors)/n)**0.5
    
    # Compute centroid
    cx=sum(p.x for p in points)/n
    cy=sum(p.y for p in points)/n
    cz=sum(p.z for p in points)/n
    
    # Compute distances from centroid
    distances=[((p.x-cx)**2+(p.y-cy)**2+(p.z-cz)**2)**0.5 for p in points]
    mean_dist=sum(distances)/n
    std_dist=(sum((d-mean_dist)**2 for d in distances)/n)**0.5
    
    return {
        'mean_error': mean_err,
        'std_error': std_err,
        'centroid': (cx,cy,cz),
        'distances': distances,
        'mean_distance': mean_dist,
        'std_distance': std_dist
    }


def select_kept_indices(points:List[PointRecord], args)->List[int]:
    n=len(points)
    
    # Handle outlier strategies that don't use keep_n/remove_n/remove_pct explicitly
    if args.strategy in ['error_outlier','distance_outlier','combined_outlier']:
        stats=compute_statistics(points)
        outlier_mask=[False]*n
        
        if args.strategy=='error_outlier':
            threshold=stats['mean_error']+args.outlier_sigma*stats['std_error']
            outlier_mask=[points[i].error>threshold for i in range(n)]
        elif args.strategy=='distance_outlier':
            threshold=stats['mean_distance']+args.outlier_sigma*stats['std_distance']
            outlier_mask=[stats['distances'][i]>threshold for i in range(n)]
        elif args.strategy=='combined_outlier':
            err_threshold=stats['mean_error']+args.outlier_sigma*stats['std_error']
            dist_threshold=stats['mean_distance']+args.outlier_sigma*stats['std_distance']
            outlier_mask=[points[i].error>err_threshold and stats['distances'][i]>dist_threshold for i in range(n)]
        
        # Now handle removal modes with detected outliers
        outlier_indices=[i for i in range(n) if outlier_mask[i]]
        inlier_indices=[i for i in range(n) if not outlier_mask[i]]
        
        if args.keep_n is not None:
            # Keep N inliers (prefer inliers over outliers)
            keep_n=min(args.keep_n,n)
            if len(inlier_indices)>=keep_n:
                return inlier_indices[:keep_n]
            else:
                # Not enough inliers, need some outliers too
                return inlier_indices+outlier_indices[:keep_n-len(inlier_indices)]
        elif args.remove_pct is not None or args.remove_n is not None:
            if args.remove_pct is not None:
                remove_n=int(round(n*args.remove_pct/100.0))
            else:
                remove_n=min(args.remove_n,n)
            # Remove outliers first, then additional points if needed
            if len(outlier_indices)>=remove_n:
                to_remove=set(outlier_indices[:remove_n])
            else:
                to_remove=set(outlier_indices+inlier_indices[:remove_n-len(outlier_indices)])
            return [i for i in range(n) if i not in to_remove]
        else:
            # No explicit count, just remove all detected outliers
            return inlier_indices


def write_points3d(path:Path, header_lines:List[str], kept_points:List[PointRecord]):
    mean_track=compute_mean_track_length(kept_points)
    # Replace or append the number-of-points header
    out_header=[]
    replaced=False
    for h in header_lines:
        if h.lower().startswith('# number of points:'):
            out_header.append(f"# Number of points: {len(kept_points)}, mean track length: {mean_track}")
            replaced=True
        else:
            out_header.append(h)
    if not replaced:
        out_header.append(f"# Number of points: {len(kept_points)}, mean track length: {mean_track}")
    with path.open('w',encoding='utf-8') as f:
        for h in out_header:
            f.write(h+'\n')
        for p in kept_points:
            # Reconstruct line to ensure consistency (ignore original formatting beyond spacing)
            tracks=' '.join(f"{img} {p2d}" for img,p2d in p.track_pairs)
            base=f"{p.id} {p.x} {p.y} {p.z} {p.r} {p.g} {p.b} {p.error}"
            if tracks:
                f.write(base+ ' ' + tracks + '\n')
            else:
                f.write(base+'\n')


def write_ply(path:Path, kept_points:List[PointRecord]):
    with path.open('w',encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(kept_points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for p in kept_points:
            f.write(f"{p.x} {p.y} {p.z} {p.r} {p.g} {p.b}\n")


def main():
    args=parse_args()
    if not args.input_txt.exists():
        print(f"Input file not found: {args.input_txt}", file=sys.stderr)
        sys.exit(1)
    
    header, points=read_points3d(args.input_txt)
    original_n=len(points)
    
    # Compute and report statistics
    stats=compute_statistics(points)
    print(f"Original points: {original_n}")
    print(f"Strategy: {args.strategy} (sigma={args.outlier_sigma})")
    print(f"Error statistics: mean={stats['mean_error']:.6f}, std={stats['std_error']:.6f}")
    print(f"  -> Error threshold: {stats['mean_error']+args.outlier_sigma*stats['std_error']:.6f}")
    print(f"Distance statistics: mean={stats['mean_distance']:.6f}, std={stats['std_distance']:.6f}")
    print(f"  -> Distance threshold: {stats['mean_distance']+args.outlier_sigma*stats['std_distance']:.6f}")
    print(f"Centroid: ({stats['centroid'][0]:.3f}, {stats['centroid'][1]:.3f}, {stats['centroid'][2]:.3f})")
    
    kept_indices=select_kept_indices(points,args)
    kept_points=[points[i] for i in kept_indices]
    removed_n=original_n-len(kept_points)
    
    if args.remove_pct is not None:
        print(f"Requested remove_pct: {args.remove_pct}% -> removing {removed_n}")
    elif args.remove_n is not None:
        print(f"Requested remove_n: {args.remove_n} -> removing {removed_n}")
    elif args.keep_n is not None:
        print(f"Requested keep_n: {args.keep_n} -> keeping {len(kept_points)} (removed {removed_n})")
    else:
        print(f"Removed {removed_n} outliers automatically")
    
    print(f"Final: kept {len(kept_points)} points, removed {removed_n} points")
    print(f"Mean track length (original): {compute_mean_track_length(points):.6f}")
    print(f"Mean track length (kept): {compute_mean_track_length(kept_points):.6f}")

    if args.dry_run:
        print("Dry run enabled: no files written.")
        return

    out_txt=args.output_txt or args.input_txt.with_name(args.input_txt.stem + '_filtered.txt')
    write_points3d(out_txt, header, kept_points)
    print(f"Wrote filtered points file: {out_txt}")

    if args.output_ply:
        write_ply(args.output_ply, kept_points)
        print(f"Wrote PLY file: {args.output_ply}")

if __name__=='__main__':
    main()
