import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import time
from dataclasses import dataclass, field

import numpy as np

from src.forward_kinematics import SO_101_fk
from src.jacobian import tcp_jacobian
from src.utils.constants import (
    IK_MAX_ITERATIONS,
    IK_RESTARTS,
    IK_STALL_ITERATIONS,
    IK_W_ROT,
    N_JOINTS,
    POSES,
    POSITIONS,
    RND_SEED,
)
from src.utils.limits import inv_kin_limits

MODES = ("position", "pitch", "pose")
TASK_RANK2 = {"position": 0, "pitch": 1, "pose": 2}
METHODS = ("priority", "stacked")
LIMITS = None
PAN_AXIS_XY = POSITIONS[0, :2]


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    status: str  # converged | max_iters | stalled | stationary
    pos_err: float
    ori_err: float
    iterations: int
    restarts: int
    sigma: float
    time_s: float = 0.0
    history: list = field(default_factory=list)


def default_limits():
    global LIMITS
    if LIMITS is None:
        LIMITS = inv_kin_limits()
    return LIMITS


def heading_of(p):
    d = p[:2] - PAN_AXIS_XY
    return float(np.arctan2(d[1], d[0]))


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _seeds(q0, p_dest, lim, restarts, seed):
    out = [np.zeros(N_JOINTS) if q0 is None else q0]
    h = heading_of(p_dest)
    pan_forward = np.clip(_wrap(-h), lim[0, 0], lim[0, 1])
    pan_back = np.clip(_wrap(-h + np.pi), lim[0, 0], lim[0, 1])
    for v in POSES.values():
        s = np.array(v)
        s[0] = pan_forward
        out.append(s)
    rng = np.random.default_rng(seed)
    k = 0
    while len(out) < restarts + 1:
        s = rng.uniform(lim[:, 0], lim[:, 1])
        s[0] = pan_back if k % 2 == 0 else pan_forward
        out.append(s)
        k += 1
    return out[: restarts + 1]


def solve(q0, p_dest, mode, pitch, R_dest, method, lim, max_iterations, record):
    q = np.clip(np.asarray(q0, dtype=float), lim[:, 0], lim[:, 1])
    best = (np.inf, q.copy(), np.inf, np.inf)  # pos+IK_W_ROT*ori, q, pos, pro
    best_k = 0
    status = "max_iters"
    hist = []
    for k in range(max_iterations + 1):
        # Loop Algorithm DLS
        if k - best_k >= IK_STALL_ITERATIONS:
            status = "stalled"
            break
        dq = 0.0
        if np.abs(dq).max() < 1e-12:
            status = "stationary"
            break
        q = np.clip(q + dq, lim[:, 0], lim[:, 1])
    _, q_b, pos_b, ori_b = best
    return q_b, False, status, pos_b, ori_b, k, hist


def inverse_kinematics(
    p_dest,
    q0=None,
    mode="position",
    pitch=None,
    R_dest=None,
    method="priority",
    limits=None,
    restarts=IK_RESTARTS,
    max_iterations=IK_MAX_ITERATIONS,
    seed=RND_SEED,
    record=False,
):
    if mode not in MODES or method not in METHODS:
        raise ValueError(f"Mode in {MODES} and method in {METHODS}")
    if mode == "pitch" and pitch is None:
        raise ValueError("pitch mode needs pitch parameter")
    if mode == "pose" and R_dest is None:
        raise ValueError("pose needs R_dest parameter")
    lim = default_limits() if limits is None else limits
    total_iterations = 0
    t0 = time.perf_counter()
    best = None
    for i, s in enumerate(_seeds(q0, p_dest, lim, restarts, seed)):
        q, ok, status, pos, ori, it, hist = solve(
            s, p_dest, mode, pitch, R_dest, method, lim, max_iterations, record
        )
        total_iterations += it
        cand = (ok, -(pos + IK_W_ROT * ori), q, status, pos, ori, i, hist)
        if best is None or cand[:2] > best[:2]:
            best = cand
        if ok:
            break
    ok, _, q, status, pos, ori, i, hist = best
    sigma = float(np.linalg.svd(tcp_jacobian(q), compute_uv=False)[-1])
    return IKResult(
        q,
        ok,
        status,
        pos,
        ori,
        total_iterations,
        i,
        sigma,
        time.perf_counter() - t0,
        hist,
    )


def ik_position(p_dest, q0=None, **kw):
    return inverse_kinematics(p_dest, q0, mode="position", **kw)


def ik_pitch(p_dest, pitch, q0=None, **kw):
    return inverse_kinematics(p_dest, q0, mode="pitch", pitch=pitch, **kw)


def pitch_of(R):
    r = R[:, 0]
    return float(np.arctan2(-r[2], np.hypot(r[0], r[1])))


def ik_pose(T_dest, q0=None, orientation="full", **kw):
    if orientation == "pitch":
        return inverse_kinematics(
            T_dest[:3, 3], q0, mode="pitch", pitch=pitch_of(T_dest[:3, :3]), **kw
        )
    return inverse_kinematics(
        T_dest[:3, 3], q0, mode="pose", R_dest=T_dest[:3, :3], **kw
    )


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    q = np.array(POSES["side_mixed"])
    T = SO_101_fk(q)
    print("target from POSES, p=", T[:3, 3])
    for name, r in [
        ("position", ik_position(T[:3, 3])),
        ("pitch", ik_pose(T, orientation="pitch")),
        ("pose", ik_pose(T)),
    ]:
        print(
            f"{name:9s} {r.status:10s} q={r.q} pos_err={r.pos_err:2e} m ori_err={r.ori_err:.2e} rad iters={r.iterations} restarts={r.restarts} t={r.time_s * 1e3:.1f} ms"
        )
