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
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3]))


def rotation_error(A, B):
    """Angle between the rotations of two 4x4 transforms [rad]. arccos cannot resolve angles below ~1e-8 rad for tiny angles use an atan2-based formula"""
    cos_angle = (np.trace(A.T @ B) - 1.0) / 2.0
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
    return {j: (1 - s) * A[j] + s * B[j] for j in joints_list}


def skew(w):
    """3 vector -> 3x3 skew-symmetric matrix [w]"""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def adjoint(T):
    """6x6 adjoint Ad_T for twists ordered [omega; v] with Ad_T=[R,0,[[p]Rm R]], mapping a twist from frame of T to base frame"""
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def damped_pseudo_inverse(J, eps=0.0, lam_max=0.0, rank=None, condition=1e-10):
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    r = min(J.shape) if rank is None else rank
    sigma = float(s[r - 1]) if r >= 1 else 0.0
    lam2 = lam_max**2 * (1.0 - (sigma / eps) ** 2) if (eps > 0 and sigma < eps) else 0.0
    keep = np.zeros_like(s, dtype=bool)
    keep[:r] = True
    keep &= s > condition * (s[0] if s.size else 0.0)
    gain = np.zeros_like(s)
    gain[keep] = s[keep] / (s[keep] ** 2 + lam2)
    Vr = Vt[keep].T
    N = np.eye(J.shape[1]) - Vr @ Vr.T
    return (Vt.T * gain) @ U.T, sigma, float(np.sqrt(lam2)), N


def vee_so3(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def vector_to_so3(V):
    return np.array(
        [
            [0, -V[2], V[1]],
            [V[2], 0, -V[0]],
            [-V[1], V[0], 0],
        ]
    )


def trace(R):
    return R[0, 0] + R[1, 1] + R[2, 2]


def so3_log(R):
    """Rotation vector (omega_hat * theta) of a 3x3 rotation matrix. Robust near theta=0 and theta=pi (atan2-based)"""
    if np.shape(R) != (3, 3):
        raise ValueError("Matrix is not a 3x3 rotation matrix")
    w = vee_so3(R - R.T) / 2.0  # = sin(theta) * omega_hat
    s = np.linalg.norm(w)
    c = (trace(R) - 1.0) / 2.0  # = cos(theta)
    theta = np.arctan2(s, c)
    if s > 1e-6:  # general case
        return w / s * theta
    if c > 0:  # theta ~ 0
        return w
    # theta ~ pi (axis sign is arbitrary)
    i = np.argmax(np.diag(R))
    v = R[:, i].copy()
    v[i] += 1
    return np.pi * v / np.linalg.norm(v)
