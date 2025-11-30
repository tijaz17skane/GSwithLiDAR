import numpy as np
from scipy.spatial.transform import Rotation as R

def cam2world(qw, qx, qy, qz, tx, ty, tz):
    """
    Input: camera pose in world

      - q_cw = (qw,qx,qy,qz): rotation from camera to world
      - t_cw = (tx,ty,tz): camera origin in world

      Model: X_w = R_cw X_c + t_cw

    Output: world-to-camera extrinsics

      - t_wc, R_wc, q_wc such that X_c = R_wc X_w + t_wc

    """
    # SciPy uses [x,y,z,w]
    q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    q_xyzw /= np.linalg.norm(q_xyzw)

    R_cw = R.from_quat(q_xyzw).as_matrix()
    R_wc = R_cw.T

    t_cw = np.array([tx, ty, tz], dtype=np.float64)
    t_wc = -R_wc @ t_cw

    # q_wc = conjugate(q_cw) in [w,x,y,z]
    q_wc = np.array([q_xyzw[3], -q_xyzw[0], -q_xyzw[1], -q_xyzw[2]], dtype=np.float64)
    return t_wc, R_wc, q_wc

def world2cam(qw, qx, qy, qz, tx, ty, tz):
    """
    Input: world-to-camera extrinsics

      - q_wc = (qw,qx,qy,qz): rotation from world to camera
      - t_wc = (tx,ty,tz)

      Model: X_c = R_wc X_w + t_wc

    Output: camera pose in world

      - t_cw, R_cw, q_cw such that X_w = R_cw X_c + t_cw

    """
    q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    q_xyzw /= np.linalg.norm(q_xyzw)

    R_wc = R.from_quat(q_xyzw).as_matrix()
    R_cw = R_wc.T

    t_wc = np.array([tx, ty, tz], dtype=np.float64)
    t_cw = -R_cw @ t_wc

    # q_cw = conjugate(q_wc)
    q_cw = np.array([q_xyzw[3], -q_xyzw[0], -q_xyzw[1], -q_xyzw[2]], dtype=np.float64)
    return t_cw, R_cw, q_cw



import numpy as np
from scipy.spatial.transform import Rotation as R

def cam2world_batch(quats_cw: np.ndarray, trans_cw: np.ndarray):
    """
    Batch version of cam2world.

    Inputs:
      quats_cw: (N, 4) as [qw, qx, qy, qz] rotation from camera to world
      trans_cw: (N, 3) as [tx, ty, tz] camera origin in world

    Outputs:
      trans_wc: (N, 3)
      R_wc:     (N, 3, 3)
      quats_wc: (N, 4) as [qw, qx, qy, qz]
    """
    quats_cw = np.asarray(quats_cw, dtype=np.float64)
    trans_cw = np.asarray(trans_cw, dtype=np.float64)

    # to SciPy's [x,y,z,w]
    q_xyzw = np.stack([quats_cw[:, 1], quats_cw[:, 2], quats_cw[:, 3], quats_cw[:, 0]], axis=1)
    q_xyzw /= np.linalg.norm(q_xyzw, axis=1, keepdims=True)

    R_cw = R.from_quat(q_xyzw).as_matrix()            # (N,3,3)
    R_wc = np.transpose(R_cw, (0, 2, 1))              # transpose per batch

    t_cw = trans_cw                                    # (N,3)
    trans_wc = -(R_wc @ t_cw[..., None]).squeeze(-1)   # (N,3)

    # conjugate: [w, -x, -y, -z] -> still [qw,qx,qy,qz]
    quats_wc = np.stack([q_xyzw[:, 3], -q_xyzw[:, 0], -q_xyzw[:, 1], -q_xyzw[:, 2]], axis=1)

    return trans_wc, R_wc, quats_wc

def world2cam_batch(quats_wc: np.ndarray, trans_wc: np.ndarray):
    """
    Batch version of world2cam.

    Inputs:
      quats_wc: (N, 4) as [qw, qx, qy, qz] rotation from world to camera
      trans_wc: (N, 3) as [tx, ty, tz]

    Outputs:
      trans_cw: (N, 3)
      R_cw:     (N, 3, 3)
      quats_cw: (N, 4) as [qw, qx, qy, qz]
    """
    quats_wc = np.asarray(quats_wc, dtype=np.float64)
    trans_wc = np.asarray(trans_wc, dtype=np.float64)

    # to SciPy's [x,y,z,w]
    q_xyzw = np.stack([quats_wc[:, 1], quats_wc[:, 2], quats_wc[:, 3], quats_wc[:, 0]], axis=1)
    q_xyzw /= np.linalg.norm(q_xyzw, axis=1, keepdims=True)

    R_wc = R.from_quat(q_xyzw).as_matrix()            # (N,3,3)
    R_cw = np.transpose(R_wc, (0, 2, 1))

    t_wc = trans_wc
    trans_cw = -(R_cw @ t_wc[..., None]).squeeze(-1)

    # conjugate
    quats_cw = np.stack([q_xyzw[:, 3], -q_xyzw[:, 0], -q_xyzw[:, 1], -q_xyzw[:, 2]], axis=1)

    return trans_cw, R_cw, quats_cw