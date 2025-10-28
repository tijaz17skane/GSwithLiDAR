import argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description="Align COLMAP camera positions to another reference using NAME correspondences.")
parser.add_argument('--input_poses', type=str, required=True, help='Reference camera positions TXT')
parser.add_argument('--input_colmap', type=str, required=True, help='COLMAP camera positions TXT to align')
parser.add_argument('--out_transform', type=str, required=True, help='Output transformation matrix TXT')
# Add argument for tolerance
parser.add_argument('--tolerance', type=float, default=0.01, help='Alignment tolerance (distance)')
args = parser.parse_args()

def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    # Ignore first 4 rows, then take odd rows (lines[4], lines[6], ...)
    data_lines = [lines[i] for i in range(4, len(lines), 2)]
    rows = []
    for l in data_lines:
        parts = l.strip().split()
        if len(parts) < 10:
            continue
        row = {
            'IMAGE_ID': parts[0],
            'QW': float(parts[1]), 'QX': float(parts[2]), 'QY': float(parts[3]), 'QZ': float(parts[4]),
            'TX': float(parts[5]), 'TY': float(parts[6]), 'TZ': float(parts[7]),
            'CAMERA_ID': parts[8],
            'NAME': parts[9]
        }
        rows.append(row)
    return pd.DataFrame(rows)

def colmap_to_world(qw, qx, qy, qz, tx, ty, tz, norm_offset=None):
    q_cam = [qx, qy, qz, qw]  # Quaternion (x, y, z, w)
    t_cam = np.array([tx, ty, tz])
    R_cam = R.from_quat(q_cam).as_matrix()
    R_world = R_cam.T
    t_world = -R_world @ t_cam
    if norm_offset is not None:
        t_world = t_world + norm_offset
    T_world = np.eye(4)
    T_world[:3, :3] = R_world
    T_world[:3, 3] = t_world
    rot_world = R_world
    pos = t_world
    return pos, rot_world

def get_world_positions(df, norm_offset=None):
    positions = []
    for _, row in df.iterrows():
        pos, _ = colmap_to_world(row['QW'], row['QX'], row['QY'], row['QZ'], row['TX'], row['TY'], row['TZ'], norm_offset)
        positions.append(pos)
    return np.array(positions)

def write_ply(points, ply_path, color):
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(points)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in points:
            f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(p[0], p[1], p[2], color[0], color[1], color[2]))

poses_df = read_txt(args.input_poses)
colmap_df = read_txt(args.input_colmap)

# --- PART 1: Camera-to-world conversion and PLY generation ---

# Convert all poses (reference) to world coordinates (yellow)
poses_xyz_full = get_world_positions(poses_df)
# Convert all colmap (to align) to world coordinates (red)
colmap_xyz_full = get_world_positions(colmap_df)

# Write reference (poses) as yellow (255,255,0)
write_ply(poses_xyz_full, args.input_poses.replace('.txt', '_yellow.ply'), (255,255,0))
# Write colmap as red (255,0,0)
write_ply(colmap_xyz_full, args.input_colmap.replace('.txt', '_red.ply'), (255,0,0))

# Write merged ply (yellow for poses, red for colmap)
merged_points = np.vstack([
    np.column_stack([poses_xyz_full, np.tile([255,255,0], (poses_xyz_full.shape[0],1))]),
    np.column_stack([colmap_xyz_full, np.tile([255,0,0], (colmap_xyz_full.shape[0],1))])
])
merged_ply_path = args.out_transform.replace('.txt', '_merged.ply')
with open(merged_ply_path, "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(merged_points)))
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p in merged_points:
        f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(p[0], p[1], p[2], int(p[3]), int(p[4]), int(p[5])))
print(f"PLY files written: {args.input_poses.replace('.txt', '_yellow.ply')}, {args.input_colmap.replace('.txt', '_red.ply')}, {merged_ply_path}")

# --- PART 2: Iterative alignment with translation and rotation ---

# Use the converted world coordinates and names for correspondences
poses_df['x_world'], poses_df['y_world'], poses_df['z_world'] = poses_xyz_full[:,0], poses_xyz_full[:,1], poses_xyz_full[:,2]
colmap_df['x_world'], colmap_df['y_world'], colmap_df['z_world'] = colmap_xyz_full[:,0], colmap_xyz_full[:,1], colmap_xyz_full[:,2]

# Establish correspondences by NAME
merged = pd.merge(poses_df[['NAME','x_world','y_world','z_world']], colmap_df[['NAME','x_world','y_world','z_world']], on='NAME', suffixes=('_poses', '_colmap'))

if len(merged) < 1:
    raise ValueError('No correspondences found for alignment.')

colmap_iter = colmap_xyz_full.copy()

# Step 1: Translate so first correspondence matches
ref_point = np.array([merged.iloc[0]['x_world_poses'], merged.iloc[0]['y_world_poses'], merged.iloc[0]['z_world_poses']])
colmap_point = np.array([merged.iloc[0]['x_world_colmap'], merged.iloc[0]['y_world_colmap'], merged.iloc[0]['z_world_colmap']])
translation = ref_point - colmap_point
colmap_iter = colmap_iter + translation

for idx in range(1, len(merged)):
    # Fix previous correspondence
    prev_ref = np.array([
        merged.iloc[idx-1]['x_world_poses'],
        merged.iloc[idx-1]['y_world_poses'],
        merged.iloc[idx-1]['z_world_poses']
    ])
    prev_colmap = np.array([
        merged.iloc[idx-1]['x_world_colmap'],
        merged.iloc[idx-1]['y_world_colmap'],
        merged.iloc[idx-1]['z_world_colmap']
    ]) + translation
    # Get current correspondence
    curr_ref = np.array([
        merged.iloc[idx]['x_world_poses'],
        merged.iloc[idx]['y_world_poses'],
        merged.iloc[idx]['z_world_poses']
    ])
    curr_colmap = np.array([
        merged.iloc[idx]['x_world_colmap'],
        merged.iloc[idx]['y_world_colmap'],
        merged.iloc[idx]['z_world_colmap']
    ]) + translation
    # Compute vectors from previous to current
    v_ref = curr_ref - prev_ref
    v_colmap = curr_colmap - prev_colmap
    # Compute rotation to align v_colmap to v_ref
    v_ref_norm = v_ref / np.linalg.norm(v_ref)
    v_colmap_norm = v_colmap / np.linalg.norm(v_colmap)
    axis = np.cross(v_colmap_norm, v_ref_norm)
    angle = np.arccos(np.clip(np.dot(v_colmap_norm, v_ref_norm), -1.0, 1.0))
    if np.linalg.norm(axis) > 1e-8 and angle > 1e-8:
        axis = axis / np.linalg.norm(axis)
        rot = R.from_rotvec(angle * axis)
        # Rotate all colmap points about prev_colmap
        colmap_iter = rot.apply(colmap_iter - prev_colmap) + prev_colmap
        # Update curr_colmap for translation
        curr_colmap = rot.apply(curr_colmap - prev_colmap) + prev_colmap
    # After rotation, translate so current correspondence matches
    step_translation = curr_ref - curr_colmap
    colmap_iter = colmap_iter + step_translation
    # Check if aligned within tolerance
    aligned_dist = np.linalg.norm((colmap_iter[idx] - curr_ref))
    print(f'Correspondence {idx+1}: aligned distance = {aligned_dist:.6f}')
    if aligned_dist > args.tolerance:
        print(f'Stopping: correspondence {idx+1} not aligned within tolerance {args.tolerance}')
        break

# Output final merged PLY after last successful alignment
merged_points_iter = np.vstack([
    np.column_stack([poses_xyz_full, np.tile([255,255,0], (poses_xyz_full.shape[0],1))]),
    np.column_stack([colmap_iter, np.tile([255,0,0], (colmap_iter.shape[0],1))])
])
merged_ply_path_iter = args.out_transform.replace('.txt', '_merged_iterative.ply')
with open(merged_ply_path_iter, "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(merged_points_iter)))
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p in merged_points_iter:
        f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(p[0], p[1], p[2], int(p[3]), int(p[4]), int(p[5])))
print(f"Iterative merged PLY written: {merged_ply_path_iter}")
