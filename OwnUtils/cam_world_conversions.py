import numpy as np
from scipy.spatial.transform import Rotation as R

def cam2world(qw, qx, qy, qz, tx, ty, tz, norm_offset=None):
    """
    Convert camera coordinates (quaternion + translation) to world coordinates.
    Returns (position_world, rotation_world_matrix)
    """
    q_cam = [qx, qy, qz, qw]  # Quaternion (x, y, z, w)
    t_cam = np.array([tx, ty, tz], dtype=np.float64)
    R_cam = R.from_quat(q_cam).as_matrix()
    R_world = R_cam.T  # Inverse of rotation matrix
    t_world = -R_world @ t_cam  # Inverse translation
    if norm_offset is not None:
        t_world = t_world + norm_offset
    return t_world, R_world

def world2cam(qw, qx, qy, qz, tx, ty, tz, norm_offset=None):
    """
    Convert world coordinates (camera center) back to COLMAP camera coordinates.
    Quaternion should represent R_w2c (world-to-camera).
    tx, ty, tz should be camera center in world coordinates.
    Returns (t_cam, R_w2c)
    """
    q_w2c = [qx, qy, qz, qw]  # Quaternion for R_w2c
    C_world = np.array([tx, ty, tz], dtype=np.float64)
    R_w2c = R.from_quat(q_w2c).as_matrix()
    
    if norm_offset is not None:
        C_world = C_world - norm_offset
    
    # Compute translation: t = -R_w2c @ C_world
    t_cam = -R_w2c @ C_world  # ✅ Fixed: use R_w2c, not R_w2c^T
    return t_cam, R_w2c