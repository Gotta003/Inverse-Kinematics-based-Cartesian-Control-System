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
    IK_BACKTRACK,
    IK_EPS_ORI,
    IK_EPS_POS,
    IK_LAMBDA_ORI,
    IK_LAMBDA_POS,
    IK_MAX_ITERATIONS,
    IK_MAX_JOINT_STEP,
    IK_MAX_ORI_ERR,
    IK_MAX_POS_ERR,
    IK_MIN_STEP,
    IK_POS_TOL,
    IK_RESTARTS,
    IK_ROT_TOL,
    IK_STALL_ITERATIONS,
    IK_STALL_REL,
    IK_W_ROT,
    N_JOINTS,
    POSES,
    POSITIONS,
    RND_SEED,
)
from src.utils.helper import damped_pseudo_inverse, skew, so3_log
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


def _clip_norm(e, max_norm):
    """Scale e so that its norm is at most max_norm"""
    n = np.linalg.norm(e)
    return e * (max_norm / n) if n > max_norm else e


def _errors(q, p_dest, mode, pitch, R_dest):
    """Task errors at q: position vector, orientation vector (0, 1 or 3 entries), their norms and R"""
    T = SO_101_fk(q)
    R = T[:3, :3]
    e_p = p_dest - T[:3, 3]
    if mode == "pitch":
        e_o = np.array([_wrap(pitch - pitch_of(R))])
    elif mode == "pose":
        e_o = so3_log(R_dest @ R.T)  # spatial frame, consistent with J_w
    else:
        e_o = np.zeros(0)
    return e_p, e_o, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_o)), R


def pitch_jacobian(R, J_w):
    """1x5 Jacobian of pitch_of(R): with r=R[:,0], r_dot=w x r=-[r]J_w q_dot,
    and d(pitch)/dr=[r_z r_x/h, r_z r_y/h, -h], h=hypot(r_x, r_y)"""
    r = R[:, 0]
    h = max(np.hypot(r[0], r[1]), 1e-9)
    g = np.array([r[2] * r[0] / h, r[2] * r[1] / h, -h])
    return (g @ (-skew(r) @ J_w))[None, :]


def _task_jacobians(q, mode, R):
    """Position Jacobian J_p (3x5) and orientation-task Jacobian J_o (0x5, 1x5 or 3x5)"""
    J = tcp_jacobian(q)
    J_w, J_p = J[:3], J[3:]
    if mode == "pitch":
        J_o = pitch_jacobian(R, J_w)
    elif mode == "pose":
        J_o = J_w
    else:
        J_o = np.zeros((0, J.shape[1]))
    return J_p, J_o


def _dls_step(J_p, e_p, J_o, e_o, method, rank2):
    """Unconstrained step on the given columns.
    - stacked: one weighted DLS on [J_p; w J_o]
    - priority: position first, orientation in the null space of J_p
    Returns dq, sigma (monitored singular value) and the damping used"""
    n = J_p.shape[1]
    r_p = min(3, n)
    has_ori = J_o.shape[0] > 0 and rank2 > 0
    if method == "stacked" and has_ori:
        J = np.vstack((J_p, IK_W_ROT * J_o))
        e = np.concatenate((e_p, IK_W_ROT * e_o))
        J_pinv, sigma, lam, _ = damped_pseudo_inverse(J, IK_EPS_POS, IK_LAMBDA_POS, rank=min(3 + rank2, n))
        return J_pinv @ e, sigma, lam
    Jp_pinv, sigma, lam, N = damped_pseudo_inverse(J_p, IK_EPS_POS, IK_LAMBDA_POS, rank=r_p)
    dq = Jp_pinv @ e_p
    r2 = min(rank2, n - r_p)
    if has_ori and r2 > 0:
        Jo_pinv, _, _, _ = damped_pseudo_inverse(J_o @ N, IK_EPS_ORI, IK_LAMBDA_ORI, rank=r2)
        dq = dq + N @ (Jo_pinv @ (e_o - J_o @ dq))
    return dq, sigma, lam


def _constrained_step(q, J_p, e_p, J_o, e_o, method, rank2, lim):
    """Active-set clamping: joints that would cross a limit are moved exactly onto it,
    their columns are removed, and the remaining joints re-solve the residual task.
    Repeats until no new joint saturates. The step is scaled so no joint moves more than IK_MAX_JOINT_STEP"""
    free = np.ones(len(q), dtype=bool)
    dq_fixed = np.zeros(len(q))
    dq = np.zeros(len(q))
    sigma, lam = 0.0, 0.0
    for _ in range(len(q)):
        fixed = ~free
        r_p = e_p - J_p[:, fixed] @ dq_fixed[fixed]
        r_o = e_o - J_o[:, fixed] @ dq_fixed[fixed]
        dq_free, sigma, lam = _dls_step(J_p[:, free], r_p, J_o[:, free], r_o, method, rank2)
        dq = dq_fixed.copy()
        dq[free] = dq_free
        m = np.abs(dq).max()
        if m > IK_MAX_JOINT_STEP:
            dq *= IK_MAX_JOINT_STEP / m
        q_try = q + dq
        hit = free & ((q_try < lim[:, 0]) | (q_try > lim[:, 1]))
        if not hit.any():
            break
        dq_fixed[hit] = np.clip(q_try[hit], lim[hit, 0], lim[hit, 1]) - q[hit]
        free &= ~hit
        if not free.any():
            dq = dq_fixed.copy()
            break
    return dq, sigma, lam


def solve(q0, p_dest, mode, pitch, R_dest, method, lim, max_iterations, record):
    """Single DLS run from seed q0. Returns q, success, status, pos_err, ori_err, iterations, history"""
    q = np.clip(np.asarray(q0, dtype=float), lim[:, 0], lim[:, 1])
    rank2 = TASK_RANK2[mode]
    e_p, e_o, pos_err, ori_err, R = _errors(q, p_dest, mode, pitch, R_dest)
    best = (np.inf, q.copy(), np.inf, np.inf)  # merit, q, pos_err, ori_err
    best_k = 0
    status = "max_iters"
    hist = []
    k = 0
    for k in range(max_iterations + 1):
        merit = pos_err + IK_W_ROT * ori_err
        if record:
            hist.append({"k": k, "q": q.copy(), "pos_err": pos_err, "ori_err": ori_err})

        # Progress bookkeeping (stall counter restarts only on a significant improvement)
        if merit < best[0]:
            if merit < best[0] * (1.0 - IK_STALL_REL):
                best_k = k
            best = (merit, q.copy(), pos_err, ori_err)

        # Stop rules
        if pos_err < IK_POS_TOL and ori_err < IK_ROT_TOL:
            return q, True, "converged", pos_err, ori_err, k, hist
        if k - best_k >= IK_STALL_ITERATIONS:
            status = "stalled"
            break
        if k == max_iterations:
            break

        # Step: clipped errors -> constrained DLS step
        J_p, J_o = _task_jacobians(q, mode, R)
        e_pc = _clip_norm(e_p, IK_MAX_POS_ERR)
        e_oc = _clip_norm(e_o, IK_MAX_ORI_ERR)
        dq, sigma, lam = _constrained_step(q, J_p, e_pc, J_o, e_oc, method, rank2, lim)

        # Backtracking: halve the step while the merit function increases
        alpha = 1.0
        for _ in range(IK_BACKTRACK):
            q_new = np.clip(q + alpha * dq, lim[:, 0], lim[:, 1])
            new = _errors(q_new, p_dest, mode, pitch, R_dest)
            if new[2] + IK_W_ROT * new[3] < merit:
                break
            alpha *= 0.5
        if record:
            hist[-1].update(sigma=sigma, lam=lam, alpha=alpha)

        if np.abs(q_new - q).max() < IK_MIN_STEP:
            status = "stationary"
            break
        q = q_new
        e_p, e_o, pos_err, ori_err, R = new

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
    # Target from a configuration that is NOT one of the seeds (side_mixed + offset)
    q = np.array(POSES["side_mixed"]) + np.array([0.15, -0.1, 0.2, -0.15, 0.3])
    T = SO_101_fk(q)
    print("target p=", T[:3, 3], " pitch=", round(pitch_of(T[:3, :3]), 4))
    for method in METHODS:
        for name, r in [
            ("position", ik_position(T[:3, 3], method=method)),
            ("pitch", ik_pose(T, orientation="pitch", method=method)),
            ("pose", ik_pose(T, method=method)),
        ]:
            print(
                f"{method:8s} {name:9s} {r.status:10s} q={r.q} pos_err={r.pos_err:.2e} m ori_err={r.ori_err:.2e} rad iters={r.iterations} restarts={r.restarts} t={r.time_s * 1e3:.1f} ms"
            )
