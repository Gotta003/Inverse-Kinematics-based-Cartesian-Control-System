"""
helper.py
General-purpose helpers

Phase 1: cross products of screw axes, pose errors, unit conversion, joint limits fromm MuJoCo, interpolation of dicts 
Phase 2: helpers for Jacobian (skew, adjoint)


"""

import numpy as np

def vectorial_product(A, B):
    """Row-wise cross product A[i]xB[i] (used for V=q x omega)"""
    results = []
    for i, _ in enumerate(A):
        prod = []
        prod.append(A[i][1] * B[i][2] - A[i][2] * B[i][1])
        prod.append(A[i][2] * B[i][0] - A[i][0] * B[i][2])
        prod.append(A[i][0] * B[i][1] - A[i][1] * B[i][0])
        results.append(prod)
    return np.array(results)

def position_error(A, B):
    """Distance between the translation of two 4x4 transforms [m]"""
    return float(np.linalg.norm(A[:3, 3]-B[:3, 3]))

def rotation_error(A, B):
    """Angle between the rotations of two 4x4 transforms [rad]. arccos cannot resolve angles below ~1e-8 rad for tiny angles use an atan2-based formula"""
    cos_angle=(np.trace(A.T@B)-1.0)/2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def deg_to_rad(theta_cal):
    """Degrees to radians"""
    return theta_cal * np.pi / 180

def derive_limits(m, njts):
    """Joint ranges of first joints of a MuJoCo model {name: [lo, hi]}"""
    limits = {m.joint(i).name: m.jnt_range[i].tolist() for i in range(njts)}
    return limits

def interp(A, B, s, joints_list):
    """Linear interpolation between two joint dictionaries"""
    return {j: (1-s)*A[j]+s*B[j] for j in joints_list}

def skew(w):
    """3 vector -> 3x3 skew-symmetric matrix [w]"""
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0]
    ])

def adjoint(T):
    """6x6 adjoint Ad_T for twists ordered [omega; v] with Ad_T=[R,0,[[p]Rm R]], mapping a twist from frame of T to base frame"""
    R=T[:3, :3]
    p=T[:3, 3]
    Ad=np.zeros((6, 6))
    Ad[:3, :3]=R
    Ad[3:, :3]=skew(p)@R
    Ad[3:, 3:]=R
    return Ad