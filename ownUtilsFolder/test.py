import numpy as np
from scipy.spatial.transform import Rotation as Rot
from cam_world_conversions import cam2world, world2cam

# Combined transformation matrix (T*R*S)

combined_matrix = np.array([
    [0.05379868, -0.05044009, -0.00045038, 246006.22837041],
    [-0.00602602, -0.00577300, -0.07327389, 33213.69602362],
    [0.05008086, 0.05348994, -0.00833292, -305156.02125236],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
])

# Extract top-left 3x3 (this is s*R)

combined_3x3 = combined_matrix[:3, :3]
translation = combined_matrix[:3, 3]

# Extract scale

col1_norm = np.linalg.norm(combined_3x3[:, 0])
col2_norm = np.linalg.norm(combined_3x3[:, 1])
col3_norm = np.linalg.norm(combined_3x3[:, 2])
s = (col1_norm + col2_norm + col3_norm) / 3

print(f"Scale: {s}")
print(f"Translation: {translation}")

# Extract pure rotation

R = combined_3x3 / s

print(f"\nRotation matrix R:")
print(R)
print(f"det(R) = {np.linalg.det(R):.6f}")

# SOURCE: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

# 1 -0.6436177286945087 0.7229199131032635 -0.20270558360220156 0.14850409060151423 -1.92909158 -0.70317685 1.90406876 1 front_20220920_DTAG_T4_2292003_1881.jpg

src_qw = -0.6436177286945087
src_qx = 0.7229199131032635
src_qy = -0.20270558360220156
src_qz = 0.14850409060151423
src_tx = -1.92909158
src_ty = -0.70317685  # You need to provide this
src_tz = 1.90406876   # You need to provide this

# TARGET: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

# 10 0.99242625659172123 0.010745247400959944 0.12210138110287065 -0.0081189665849143514 0.5634458813902955 -0.38267402373714721 2.7020540112077334 1 front_20220920_DTAG_T4_2292003_1881.jpg

tgt_qw = 0.99242625659172123
tgt_qx = 0.010745247400959944
tgt_qy = 0.12210138110287065
tgt_qz = -0.0081189665849143514
tgt_tx = 0.5634458813902955
tgt_ty = -0.38267402373714721
tgt_tz = 2.7020540112077334

print("\n" + "="*70)
print("STEP 1: Convert COLMAP camera coordinates to world coordinates")
print("="*70)

# Source: Convert from camera coordinates to world coordinates

C_src_world, R_src_c2w = cam2world(src_qw, src_qx, src_qy, src_qz, src_tx, src_ty, src_tz)
print(f"\nSource (COLMAP format):")
print(f"  QW, QX, QY, QZ: [{src_qw:.6f}, {src_qx:.6f}, {src_qy:.6f}, {src_qz:.6f}]")
print(f"  TX, TY, TZ:     [{src_tx:.6f}, {src_ty:.6f}, {src_tz:.6f}]")
print(f"Source (world coordinates):")
print(f"  Camera center: {C_src_world}")
print(f"  R_c2w:")
print(R_src_c2w)

# Target: Convert from camera coordinates to world coordinates

C_tgt_world, R_tgt_c2w = cam2world(tgt_qw, tgt_qx, tgt_qy, tgt_qz, tgt_tx, tgt_ty, tgt_tz)
print(f"\nTarget (COLMAP format):")
print(f"  QW, QX, QY, QZ: [{tgt_qw:.6f}, {tgt_qx:.6f}, {tgt_qy:.6f}, {tgt_qz:.6f}]")
print(f"  TX, TY, TZ:     [{tgt_tx:.6f}, {tgt_ty:.6f}, {tgt_tz:.6f}]")
print(f"Target (world coordinates):")
print(f"  Camera center: {C_tgt_world}")
print(f"  R_c2w:")
print(R_tgt_c2w)

# Get world-to-cam matrices

R_src_w2c = R_src_c2w.T
R_tgt_w2c = R_tgt_c2w.T

print("\n" + "="*70)
print("STEP 2: Apply transformation to world-space position")
print("="*70)

# Transform position in world space

C_aligned_world = s * (R @ C_src_world) + translation
print(f"\nAligned camera center (world): {C_aligned_world}")
print(f"Target camera center (world):  {C_tgt_world}")
print(f"Position error: {np.linalg.norm(C_aligned_world - C_tgt_world):.6f}")

print("\n" + "="*70)
print("STEP 3: Test all rotation transformation methods")
print("="*70)

# Test transformations on camera-to-world rotation

methods_c2w = {
    "1. R * R_src_c2w":           R @ R_src_c2w,
    "2. R_src_c2w * R":           R_src_c2w @ R,
    "3. R^T * R_src_c2w":         R.T @ R_src_c2w,
    "4. R_src_c2w * R^T":         R_src_c2w @ R.T,
    "5. R * R_src_c2w * R^T":     R @ R_src_c2w @ R.T,
    "6. R^T * R_src_c2w * R":     R.T @ R_src_c2w @ R,
}

results = []
for method_name, R_aligned_c2w in methods_c2w.items():
    # Convert aligned camera-to-world to world-to-cam
    R_aligned_w2c = R_aligned_c2w.T
    
    # Convert to quaternion (world-to-cam)
    q_aligned_w2c_scipy = Rot.from_matrix(R_aligned_w2c).as_quat()
    q_aligned_w2c = np.array([q_aligned_w2c_scipy[3], q_aligned_w2c_scipy[0], 
                             q_aligned_w2c_scipy[1], q_aligned_w2c_scipy[2]])
    
    # Convert back to COLMAP format
    t_aligned_cam, R_check = world2cam(
        q_aligned_w2c[0], q_aligned_w2c[1], q_aligned_w2c[2], q_aligned_w2c[3],
        C_aligned_world[0], C_aligned_world[1], C_aligned_world[2]
    )
    
    # Get final quaternion in COLMAP format
    q_final_scipy = Rot.from_matrix(R_check).as_quat()
    q_final = np.array([q_final_scipy[3], q_final_scipy[0], 
                       q_final_scipy[1], q_final_scipy[2]])
    
    # Compute angular difference with target quaternion
    tgt_q = np.array([tgt_qw, tgt_qx, tgt_qy, tgt_qz])
    dot_product = np.abs(np.dot(q_final, tgt_q))
    dot_product = np.clip(dot_product, 0.0, 1.0)
    angle_deg = np.degrees(2 * np.arccos(dot_product))
    
    results.append((method_name, q_final, angle_deg, t_aligned_cam))
    print(f"\n{method_name}")
    print(f"  Aligned quat: [{q_final[0]:9.6f}, {q_final[1]:9.6f}, {q_final[2]:9.6f}, {q_final[3]:9.6f}]")
    print(f"  Aligned t_cam: [{t_aligned_cam[0]:9.6f}, {t_aligned_cam[1]:9.6f}, {t_aligned_cam[2]:9.6f}]")
    print(f"  Angular error: {angle_deg:.3f}°")

# Sort by angular error

results.sort(key=lambda x: x[2])

print("\n" + "="*70)
print("RANKING (Best to Worst)")
print("="*70)
for i, (method_name, q_result, angle_deg, t_cam) in enumerate(results):
    print(f"{i+1}. {method_name:25s} → {angle_deg:6.3f}°")

print("\n" + "="*70)
print(f"Target quat:     [{tgt_qw:9.6f}, {tgt_qx:9.6f}, {tgt_qy:9.6f}, {tgt_qz:9.6f}]")
print(f"Target t_cam:    [{tgt_tx:9.6f}, {tgt_ty:9.6f}, {tgt_tz:9.6f}]")
print(f"\nBest quat:       [{results[0][1][0]:9.6f}, {results[0][1][1]:9.6f}, {results[0][1][2]:9.6f}, {results[0][1][3]:9.6f}]")
print(f"Best t_cam:      [{results[0][3][0]:9.6f}, {results[0][3][1]:9.6f}, {results[0][3][2]:9.6f}]")
print("="*70)