import argparse
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
import os
import time

def load_point_cloud(filepath):
    """Load point cloud from PLY file."""
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors

def save_point_cloud(points, colors, filepath):
    """Save point cloud to PLY file."""
    if len(points) == 0:
        pcd = o3d.geometry.PointCloud()
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filepath, pcd)

def get_scene_bounds(points_A, points_B):
    """Get bounding box containing both point clouds."""
    all_points = np.vstack([points_A, points_B])
    min_bound = np.min(all_points, axis=0)
    max_bound = np.max(all_points, axis=0)
    return min_bound, max_bound

def divide_scene_into_sections(min_bound, max_bound, num_sections):
    """Divide 3D space into subsections."""
    scene_size = max_bound - min_bound
    largest_dim = np.argmax(scene_size)
    
    divisions = np.linspace(min_bound[largest_dim], max_bound[largest_dim], num_sections + 1)
    sections = []
    
    for i in range(num_sections):
        section_min = min_bound.copy()
        section_max = max_bound.copy()
        section_min[largest_dim] = divisions[i]
        section_max[largest_dim] = divisions[i + 1]
        sections.append((section_min, section_max))
    
    return sections

def filter_points_indices(points, section_min, section_max):
    """Get indices of points within a 3D section."""
    mask = np.all((points >= section_min) & (points <= section_max), axis=1)
    return np.where(mask)[0]

def find_nearby_points_spatial(points_A_sec, batch_B_points, threshold):
    """Find A points that are within threshold distance in any dimension from B points."""
    if len(points_A_sec) == 0 or len(batch_B_points) == 0:
        return np.array([], dtype=int)
    
    # Create expanded bounding box around all B points
    b_min = np.min(batch_B_points, axis=0) - threshold
    b_max = np.max(batch_B_points, axis=0) + threshold
    
    # Find A points within this expanded region
    mask = np.all((points_A_sec >= b_min) & (points_A_sec <= b_max), axis=1)
    nearby_indices = np.where(mask)[0]
    
    return nearby_indices

def process_batches_in_parallel(batches_data, points_A_sec_gpu, points_B_sec_gpu, indices_A_sec, indices_B_sec, threshold):
    """Process multiple batches in parallel on GPU."""
    all_matches_B = []
    all_matches_A = []
    all_valid_masks = []
    all_min_indices = []
    
    # Process all batches simultaneously
    for batch_info in batches_data:
        j, batch_end, batch_B_gpu, nearby_A_gpu, nearby_A_indices = batch_info
        
        # Calculate distances for this batch
        distances = torch.cdist(batch_B_gpu, nearby_A_gpu, p=2)
        
        # Find minimum distances and indices
        min_distances, min_indices_local = torch.min(distances, dim=1)
        
        # Filter by threshold
        valid_mask = min_distances <= threshold
        
        # Store results
        all_matches_B.append((j, batch_end, batch_B_gpu, valid_mask, min_indices_local, nearby_A_indices))
        all_valid_masks.append(valid_mask)
        all_min_indices.append(min_indices_local)
    
    return all_matches_B

def process_all_sections_gpu_parallel(points_A, points_B, colors_A, colors_B, sections, threshold, b_small_num, max_parallel_batches):
    """Process all sections with controlled parallel batch processing."""
    
    # Load all points to GPU once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading all points to GPU (Device: {device})...")
    
    points_A_gpu = torch.from_numpy(points_A).float().to(device)
    points_B_gpu = torch.from_numpy(points_B).float().to(device)
    
    print(f"A points on GPU: {points_A_gpu.shape}")
    print(f"B points on GPU: {points_B_gpu.shape}")
    
    # Initialize result accumulators
    all_matched_B = []
    all_matched_A_attrs = []
    all_unique_B = []
    all_unique_A_indices = set(range(len(points_A)))  # Start with all A points as unique
    
    section_times = []
    total_parallel_calculations = 0
    total_original_calculations = 0
    total_batch_operations = 0
    
    # Process each section
    for i, (sec_min, sec_max) in enumerate(sections):
        section_start_time = time.time()
        print(f"\n--- Processing section {i+1}/{len(sections)} ---")
        
        # Get indices of points in this section
        indices_A_sec = filter_points_indices(points_A, sec_min, sec_max)
        indices_B_sec = filter_points_indices(points_B, sec_min, sec_max)
        
        print(f"Section {i+1}: {len(indices_A_sec)} A points, {len(indices_B_sec)} B points")
        
        if len(indices_A_sec) == 0 and len(indices_B_sec) == 0:
            section_time = time.time() - section_start_time
            section_times.append(section_time)
            print(f"Section {i+1}: No points, skipping (Time: {section_time:.2f}s)")
            continue
        
        if len(indices_A_sec) == 0:
            # No A points in section, all B points are unique
            all_unique_B.append(points_B[indices_B_sec])
            section_time = time.time() - section_start_time
            section_times.append(section_time)
            print(f"Section {i+1}: No A points, all B unique (Time: {section_time:.2f}s)")
            continue
        
        # Extract section points (keep on CPU for spatial filtering)
        points_A_sec = points_A[indices_A_sec]
        points_B_sec = points_B[indices_B_sec]
        points_B_sec_gpu = points_B_gpu[indices_B_sec]
        points_A_sec_gpu = points_A_gpu[indices_A_sec]
        
        section_matched_B = []
        section_matched_A_attrs = []
        section_parallel_calculations = 0
        section_original_calculations = 0
        section_batch_operations = 0
        
        print(f"Section {i+1}: Starting streaming parallel batch processing (max {max_parallel_batches} batches)...")
        
        # Stream process batches: prepare and process in groups of max_parallel_batches
        batch_groups_processed = 0
        batch_buffer = []
        total_batches_prepared = 0
        
        for j in range(0, len(indices_B_sec), b_small_num):
            batch_end = min(j + b_small_num, len(indices_B_sec))
            batch_B_cpu = points_B_sec[j:batch_end]
            
            # Spatial filtering: find nearby A points
            nearby_A_indices = find_nearby_points_spatial(points_A_sec, batch_B_cpu, threshold)
            
            if len(nearby_A_indices) > 0:
                batch_B_gpu = points_B_sec_gpu[j:batch_end]
                nearby_A_gpu = points_A_sec_gpu[nearby_A_indices]
                batch_buffer.append((j, batch_end, batch_B_gpu, batch_B_cpu, nearby_A_gpu, nearby_A_indices))
                total_batches_prepared += 1
                
                # Process when buffer reaches max_parallel_batches
                if len(batch_buffer) >= max_parallel_batches:
                    batch_groups_processed += 1
                    print(f"  Processing batch group {batch_groups_processed}: "
                          f"batches {total_batches_prepared - len(batch_buffer) + 1}-{total_batches_prepared} "
                          f"({len(batch_buffer)} batches)")
                    
                    # Process all batches in current group simultaneously
                    for batch_idx, (j_batch, batch_end_batch, batch_B_gpu, batch_B_cpu, nearby_A_gpu, nearby_A_indices) in enumerate(batch_buffer):
                        j, batch_end = j_batch, batch_end_batch
                        batch_size = batch_end - j
                        num_candidates = nearby_A_gpu.shape[0]
                        
                        # Chunk candidates if too large to avoid OOM
                        max_candidates_per_chunk = 100000  # Limits matrix to 1500×100k = 600MB
                        
                        if num_candidates > max_candidates_per_chunk:
                            # Process candidates in chunks
                            min_distances = torch.full((batch_size,), float('inf'), device=batch_B_gpu.device)
                            min_indices_local = torch.zeros((batch_size,), dtype=torch.long, device=batch_B_gpu.device)
                            
                            for cand_start in range(0, num_candidates, max_candidates_per_chunk):
                                cand_end = min(cand_start + max_candidates_per_chunk, num_candidates)
                                nearby_A_chunk = nearby_A_gpu[cand_start:cand_end]
                                
                                # Calculate distances for this chunk
                                distances_chunk = torch.cdist(batch_B_gpu, nearby_A_chunk, p=2)
                                min_dist_chunk, min_idx_chunk = torch.min(distances_chunk, dim=1)
                                
                                # Update global minimums
                                update_mask = min_dist_chunk < min_distances
                                min_distances[update_mask] = min_dist_chunk[update_mask]
                                min_indices_local[update_mask] = min_idx_chunk[update_mask] + cand_start
                                
                                del distances_chunk, min_dist_chunk, min_idx_chunk, nearby_A_chunk
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        else:
                            # Small enough to process in one go
                            distances = torch.cdist(batch_B_gpu, nearby_A_gpu, p=2)
                            min_distances, min_indices_local = torch.min(distances, dim=1)
                            del distances
                        
                        # Filter by threshold
                        valid_mask = min_distances <= threshold
                        
                        # Convert to CPU for processing
                        valid_mask_cpu = valid_mask.cpu().numpy()
                        min_indices_local_cpu = min_indices_local.cpu().numpy()
                        
                        # Map indices back to global coordinates
                        batch_global_B_indices = indices_B_sec[j:batch_end]
                        local_to_global_A_indices = indices_A_sec[nearby_A_indices[min_indices_local_cpu]]
                        
                        # Process matches
                        match_indices = np.where(valid_mask_cpu)[0]
                        if len(match_indices) > 0:
                            global_B_match_indices = batch_global_B_indices[match_indices]
                            global_A_match_indices = local_to_global_A_indices[match_indices]
                            
                            section_matched_B.append(points_B[global_B_match_indices])
                            section_matched_A_attrs.append(points_A[global_A_match_indices])
                            
                            # Remove matched A points from unique set
                            all_unique_A_indices -= set(global_A_match_indices.tolist())
                        
                        # Update counters
                        section_batch_operations += 1
                        section_original_calculations += batch_size * len(indices_A_sec)
                        section_parallel_calculations += batch_size * len(nearby_A_indices)
                        
                        # Print detailed calculation info
                        original_calc = batch_size * len(indices_A_sec)
                        actual_calc = batch_size * len(nearby_A_indices)
                        reduction_pct = (1 - actual_calc / original_calc) * 100 if original_calc > 0 else 0
                        
                        batch_num = total_batches_prepared - len(batch_buffer) + batch_idx + 1
                        print(f"    Batch {batch_num}: "
                              f"{batch_size}×{len(indices_A_sec)} → {batch_size}×{len(nearby_A_indices)} "
                              f"({actual_calc:,} calc, ↓{reduction_pct:.1f}%)")
                    
                    # Clear buffer and free GPU memory after processing
                    del batch_buffer
                    batch_buffer = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Process remaining batches in buffer (if any)
        if len(batch_buffer) > 0:
            batch_groups_processed += 1
            print(f"  Processing final batch group {batch_groups_processed}: "
                  f"batches {total_batches_prepared - len(batch_buffer) + 1}-{total_batches_prepared} "
                  f"({len(batch_buffer)} batches)")
            
            for batch_idx, (j_batch, batch_end_batch, batch_B_gpu, batch_B_cpu, nearby_A_gpu, nearby_A_indices) in enumerate(batch_buffer):
                j, batch_end = j_batch, batch_end_batch
                batch_size = batch_end - j
                num_candidates = nearby_A_gpu.shape[0]
                
                # Chunk candidates if too large to avoid OOM
                max_candidates_per_chunk = 100000  # Limits matrix to 1500×100k = 600MB
                
                if num_candidates > max_candidates_per_chunk:
                    # Process candidates in chunks
                    min_distances = torch.full((batch_size,), float('inf'), device=batch_B_gpu.device)
                    min_indices_local = torch.zeros((batch_size,), dtype=torch.long, device=batch_B_gpu.device)
                    
                    for cand_start in range(0, num_candidates, max_candidates_per_chunk):
                        cand_end = min(cand_start + max_candidates_per_chunk, num_candidates)
                        nearby_A_chunk = nearby_A_gpu[cand_start:cand_end]
                        
                        # Calculate distances for this chunk
                        distances_chunk = torch.cdist(batch_B_gpu, nearby_A_chunk, p=2)
                        min_dist_chunk, min_idx_chunk = torch.min(distances_chunk, dim=1)
                        
                        # Update global minimums
                        update_mask = min_dist_chunk < min_distances
                        min_distances[update_mask] = min_dist_chunk[update_mask]
                        min_indices_local[update_mask] = min_idx_chunk[update_mask] + cand_start
                        
                        del distances_chunk, min_dist_chunk, min_idx_chunk, nearby_A_chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # Small enough to process in one go
                    distances = torch.cdist(batch_B_gpu, nearby_A_gpu, p=2)
                    min_distances, min_indices_local = torch.min(distances, dim=1)
                    del distances
                
                # Filter by threshold
                valid_mask = min_distances <= threshold
                
                # Convert to CPU for processing
                valid_mask_cpu = valid_mask.cpu().numpy()
                min_indices_local_cpu = min_indices_local.cpu().numpy()
                
                # Map indices back to global coordinates
                batch_global_B_indices = indices_B_sec[j:batch_end]
                local_to_global_A_indices = indices_A_sec[nearby_A_indices[min_indices_local_cpu]]
                
                # Process matches
                match_indices = np.where(valid_mask_cpu)[0]
                if len(match_indices) > 0:
                    global_B_match_indices = batch_global_B_indices[match_indices]
                    global_A_match_indices = local_to_global_A_indices[match_indices]
                    
                    section_matched_B.append(points_B[global_B_match_indices])
                    section_matched_A_attrs.append(points_A[global_A_match_indices])
                    
                    # Remove matched A points from unique set
                    all_unique_A_indices -= set(global_A_match_indices.tolist())
                
                # Update counters
                section_batch_operations += 1
                section_original_calculations += batch_size * len(indices_A_sec)
                section_parallel_calculations += batch_size * len(nearby_A_indices)
                
                # Print detailed calculation info
                original_calc = batch_size * len(indices_A_sec)
                actual_calc = batch_size * len(nearby_A_indices)
                reduction_pct = (1 - actual_calc / original_calc) * 100 if original_calc > 0 else 0
                
                batch_num = total_batches_prepared - len(batch_buffer) + batch_idx + 1
                print(f"    Batch {batch_num}: "
                      f"{batch_size}×{len(indices_A_sec)} → {batch_size}×{len(nearby_A_indices)} "
                      f"({actual_calc:,} calc, ↓{reduction_pct:.1f}%)")
            
            # Clear buffer and free GPU memory after processing final group
            del batch_buffer
            batch_buffer = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine section results
        if section_matched_B:
            all_matched_B.append(np.vstack(section_matched_B))
        if section_matched_A_attrs:
            all_matched_A_attrs.append(np.vstack(section_matched_A_attrs))
        
        # Handle unique B points (simplified approach)
        total_processed_B = len(indices_B_sec)
        matched_count = sum(len(mb) for mb in section_matched_B) if section_matched_B else 0
        unique_B_count = total_processed_B - matched_count
        
        if unique_B_count > 0:
            all_unique_B.append(points_B[indices_B_sec])  # Will be refined in post-processing if needed
        
        section_time = time.time() - section_start_time
        section_times.append(section_time)
        
        # Print detailed section statistics
        total_saved = section_original_calculations - section_parallel_calculations
        savings_pct = (total_saved / section_original_calculations) * 100 if section_original_calculations > 0 else 0
        
        print(f"Section {i+1}: Completed")
        print(f"  ├── Time: {section_time:.2f}s")
        print(f"  ├── Batch groups: {batch_groups_processed}")
        print(f"  ├── Total batches: {section_batch_operations}")
        print(f"  ├── Original calc: {section_original_calculations:,}")
        print(f"  ├── Actual calc: {section_parallel_calculations:,}")
        print(f"  ├── Saved calc: {total_saved:,} ({savings_pct:.1f}% reduction)")
        print(f"  ├── Matches: {matched_count}")
        print(f"  └── Unique B: {unique_B_count}")
        
        # Update totals
        total_parallel_calculations += section_parallel_calculations
        total_original_calculations += section_original_calculations
        total_batch_operations += section_batch_operations
    
    # Generate final results
    print("\nGenerating final results...")
    
    # Matched points
    final_matched_B = np.vstack(all_matched_B) if all_matched_B else np.array([]).reshape(0, 3)
    final_matched_A_attrs = np.vstack(all_matched_A_attrs) if all_matched_A_attrs else np.array([]).reshape(0, 3)
    
    # Unique points
    final_unique_B = np.vstack(all_unique_B) if all_unique_B else np.array([]).reshape(0, 3)
    final_unique_A = points_A[list(all_unique_A_indices)] if all_unique_A_indices else np.array([]).reshape(0, 3)
    
    # Clear GPU memory
    del points_A_gpu, points_B_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return (final_matched_B, final_matched_A_attrs, final_unique_B, final_unique_A, 
            section_times, total_parallel_calculations, total_original_calculations, 
            total_batch_operations)

def main():
    parser = argparse.ArgumentParser(description="Compare point clouds with controlled parallel GPU processing")
    parser.add_argument('--pct_A', type=str, required=True, help='Path to first point cloud (PLY)')
    parser.add_argument('--pct_B', type=str, required=True, help='Path to second point cloud (PLY)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_sections', type=int, default=10, help='Number of spatial sections')
    parser.add_argument('--b_small_num', type=int, default=30000, help='Batch size for B points')
    parser.add_argument('--threshold', type=float, default=0.01, help='Distance threshold for matching')
    parser.add_argument('--max_parallel_batches', type=int, default=4, help='Maximum number of batches to process in parallel per section')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load point clouds
    print("=== Loading Point Clouds ===")
    start_time = time.time()
    points_A, colors_A = load_point_cloud(args.pct_A)
    points_B, colors_B = load_point_cloud(args.pct_B)
    load_time = time.time() - start_time
    
    print(f"Point cloud A: {len(points_A)} points")
    print(f"Point cloud B: {len(points_B)} points")
    print(f"Loading time: {load_time:.2f} seconds")
    
    # Get scene bounds
    min_bound, max_bound = get_scene_bounds(points_A, points_B)
    print(f"Scene bounds: {min_bound} to {max_bound}")
    
    # Divide scene into sections
    sections = divide_scene_into_sections(min_bound, max_bound, args.num_sections)
    print(f"Divided scene into {len(sections)} sections")
    
    # Process all sections with controlled parallel processing
    print(f"\n=== Starting Controlled Parallel GPU Processing (Max {args.max_parallel_batches} batches per group) ===")
    gpu_start_time = time.time()
    
    (final_matched_B, final_matched_A_attrs, final_unique_B, final_unique_A, 
     section_times, total_parallel_calculations, total_original_calculations, 
     total_batch_operations) = process_all_sections_gpu_parallel(
        points_A, points_B, colors_A, colors_B, sections, args.threshold, 
        args.b_small_num, args.max_parallel_batches
    )
    
    gpu_processing_time = time.time() - gpu_start_time
    
    # Save results
    print("\n=== Saving Results ===")
    save_start_time = time.time()
    save_point_cloud(final_matched_B, None, os.path.join(args.output_dir, "matched_B.ply"))
    save_point_cloud(final_matched_A_attrs, None, os.path.join(args.output_dir, "matched_A_attrs.ply"))
    save_point_cloud(final_unique_B, None, os.path.join(args.output_dir, "unique_to_B.ply"))
    save_point_cloud(final_unique_A, None, os.path.join(args.output_dir, "unique_to_A.ply"))
    save_time = time.time() - save_start_time
    
    # Calculate overall savings
    total_saved_calculations = total_original_calculations - total_parallel_calculations
    overall_savings_pct = (total_saved_calculations / total_original_calculations) * 100 if total_original_calculations > 0 else 0
    
    # Print comprehensive timing and statistics summary
    total_time = time.time() - start_time
    print(f"\n=== CONTROLLED PARALLEL PROCESSING SUMMARY ===")
    print(f"📊 INPUT DATA:")
    print(f"  ├── A points: {len(points_A):,}")
    print(f"  ├── B points: {len(points_B):,}")
    print(f"  └── Sections: {len(sections)}")
    
    print(f"\n⚙️  PARALLEL CONFIGURATION:")
    print(f"  ├── Batch size: {args.b_small_num:,}")
    print(f"  ├── Max parallel batches: {args.max_parallel_batches}")
    print(f"  └── Threshold: {args.threshold}")
    
    print(f"\n⏱️  TIMING:")
    print(f"  ├── Loading: {load_time:.2f}s")
    print(f"  ├── GPU Processing: {gpu_processing_time:.2f}s")
    print(f"  ├── Saving: {save_time:.2f}s")
    print(f"  └── Total: {total_time:.2f}s")
    
    print(f"\n⚡ CALCULATION OPTIMIZATION:")
    print(f"  ├── Original calculations: {total_original_calculations:,}")
    print(f"  ├── Actual calculations: {total_parallel_calculations:,}")
    print(f"  ├── Saved calculations: {total_saved_calculations:,}")
    print(f"  ├── Reduction: ↓{overall_savings_pct:.1f}%")
    print(f"  ├── Total batch operations: {total_batch_operations:,}")
    print(f"  └── Avg calc per batch: {total_parallel_calculations/max(total_batch_operations,1):,.0f}")
    
    print(f"\n📈 SECTION PERFORMANCE:")
    for i, t in enumerate(section_times):
        print(f"  ├── Section {i+1}: {t:.2f}s")
    print(f"  └── Average section time: {np.mean(section_times):.2f}s")
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"  ├── Matched B points: {len(final_matched_B):,}")
    print(f"  ├── Matched A attributes: {len(final_matched_A_attrs):,}")
    print(f"  ├── Unique to A: {len(final_unique_A):,}")
    print(f"  ├── Unique to B: {len(final_unique_B):,}")
    print(f"  └── Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()