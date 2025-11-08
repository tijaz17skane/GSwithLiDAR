import numpy as np
from scipy.spatial.transform import Rotation as R

def cam2world(qw, qx, qy, qz, tx, ty, tz, norm_offset=None):
    """
    Inputs:

    - q*: quaternion (w, x, y, z) representing R_cam (world-to-camera)
    - t*: translation t_cam where x_c = R_cam @ x_w + t_cam

    Returns:

    - t_world: camera center in world coordinates
    - R_world: rotation from camera to world (R_cam^T)

    - q_world: quaternion (w, x, y, z) for R_world

    """
    q_xyzw = [qx, qy, qz, qw]  # SciPy expects [x, y, z, w]
    t_cam = np.array([tx, ty, tz], dtype=np.float64)

    R_cam = R.from_quat(q_xyzw).as_matrix()
    R_world = R_cam.T
    t_world = -R_world @ t_cam

    if norm_offset is not None:
        t_world = t_world + norm_offset

    q_world_xyzw = R.from_matrix(R_world).as_quat()  # [x, y, z, w]
    q_world = np.array([q_world_xyzw[3], q_world_xyzw[0], q_world_xyzw[1], q_world_xyzw[2]])  # [w, x, y, z]

    return t_world, R_world, q_world

def world2cam(qw, qx, qy, qz, tx, ty, tz, norm_offset=None):
    """
    Inputs:

    - q*: quaternion (w, x, y, z) representing R_cam (world-to-camera)
    - t*: camera center in world coordinates (t_world)

    Returns:

    - t_cam: translation t_cam = -R_cam @ t_world
    - R_cam: rotation matrix from world to camera

    - q_cam: quaternion (w, x, y, z) for R_cam

    """
    q_xyzw = [qx, qy, qz, qw]  # SciPy expects [x, y, z, w]
    R_cam = R.from_quat(q_xyzw).as_matrix()

    t_world = np.array([tx, ty, tz], dtype=np.float64)
    if norm_offset is not None:
        t_world = t_world - norm_offset

    t_cam = -R_cam @ t_world
    q_cam = np.array([qw, qx, qy, qz], dtype=np.float64)

    return t_cam, R_cam, q_cam





'''

TESTING cam2world and world2cam functions

def quat_angle_error_wxyz(q1_wxyz, q2_wxyz):
    """
    Angular difference between two quaternions (wxyz), accounting for double-cover.
    Returns radians and degrees.
    """
    q1 = np.asarray(q1_wxyz, dtype=np.float64)
    q2 = np.asarray(q2_wxyz, dtype=np.float64)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_rad, angle_deg

# Given world pose (camera-to-world)

t_world = np.array([413279.516, 5317992.592, 303.57], dtype=np.float64)
q_world_xyzw = np.array([-0.7209, 0.2108, -0.1528, 0.6422], dtype=np.float64)  # [x, y, z, w]
q_world_wxyz = np.array([q_world_xyzw[3], q_world_xyzw[0], q_world_xyzw[1], q_world_xyzw[2]])  # [w, x, y, z]
R_world_data = R.from_quat(q_world_xyzw).as_matrix()

# Convert world to camera

# q_cam is the inverse (conjugate) of q_world

q_cam_xyzw = np.array([-q_world_xyzw[0], -q_world_xyzw[1], -q_world_xyzw[2], q_world_xyzw[3]])
q_cam_wxyz = np.array([q_cam_xyzw[3], q_cam_xyzw[0], q_cam_xyzw[1], q_cam_xyzw[2]])

t_cam, R_cam, q_cam = world2cam(q_cam_wxyz[0], q_cam_wxyz[1], q_cam_wxyz[2], q_cam_wxyz[3],
                                t_world[0], t_world[1], t_world[2])

# Round-trip back to world

t_world_rt, R_world_rt, q_world_rt = cam2world(q_cam[0], q_cam[1], q_cam[2], q_cam[3],
                                               t_cam[0], t_cam[1], t_cam[2])

# Errors

trans_err = np.linalg.norm(t_world_rt - t_world)
rot_mat_err = np.linalg.norm(R_world_rt - R_world_data, ord='fro')
quat_err_rad, quat_err_deg = quat_angle_error_wxyz(q_world_rt, q_world_wxyz)

print("t_cam:", t_cam)
print("R_cam:\n", R_cam)
print("q_cam (wxyz):", q_cam)

print("\nRound-trip checks:")
print("||Δt_world||:", trans_err)
print("||R_world_rt - R_world_data||_F:", rot_mat_err)
print("q_world (input, wxyz):", q_world_wxyz)
print("q_world_rt (wxyz):     ", q_world_rt)
print("Quaternion angle error: {:.6f} rad, {:.6f} deg".format(quat_err_rad, quat_err_deg))
'''

