"""
ik_validation.py

Outputs:
    results/data/ik_validation.csv                  (target x mode x method x start)
    results/data/ik_validation_summary.csv          success rate, errors, iters, time
    results/data/ik_singularity.csv                 singularity sweep
    results/figures/ik_errors.png                   distributions error
    results/figures/ik_iterations.png               iterations and time
    results/figures/ik_failures.png                 where the failures are
    results/figures/ik_singularity.png              sweep figure
    
Run:
    python simulation/ik_validation.py 
    python simulation/ik_validation.py --n 100      Setup n# poses for quick exec
    python simulation/ik_validation.py --view 8     Move arm to 8 IK solutions
"""
import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import argparse
import csv
import platform
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from src.forward_kinematics import SO_101_fk
from src.inverse_kinematics import (
    PAN_AXIS_XY,
    default_limits,
    heading_of,
    ik_pitch,
    inverse_kinematics,
    pitch_of,
)
from src.jacobian import position_jacobian
from src.utils.constants import IK_EPS_POS, IK_LAMBDA_POS, JOINT_NAMES, N_JOINTS
from src.utils.helper import damped_pseudo_inverse, vee_so3
from src.utils.sim_checks import contacts
from utils.so101_mujoco_utils import mujoco_starting, mujoco_tcp

SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"
DATA_DIR=REPO_ROOT/"results"/"data"
FIG_DIR=REPO_ROOT/"results"/"figures"

Z_MIN=0.02
R_MIN=0.10
CASES=[
    ("position", "priority"),
    ("pitch", "priority"),
    ("pitch", "stacked"),
    ("pose", "priority"),
    ("pose", "stacked")
]
NOISE=0.1
FAIL_CLASSES=("pan_axis", "near_singular", "joint_limit", "near_miss", "local_minimum")
SHOULDER_LIFT_POINT=np.array([0.0692345, -0.0182778, 0.1166])
FLOOR=1e-10

def view_solutions(m, d, Q, seconds=2.0):
    try:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(m, d)
    except RuntimeError as e:
        if platform.system()=="Darwin":
            print(f"MuJoCo viewer needs mjpython on macOS ({e})")
            return
        raise
    from utils.so101_mujoco_utils import convert_to_dictionary, hold_sim, move_sim
    site_id=m.site("gripperframe").id
    d.qpos[:]=0.0
    d.ctrl[:]=0.0
    mujoco.mj_forward(m, d)
    q=np.zeros(N_JOINTS)
    with viewer:
        for k, q_t in enumerate(Q):
            if not viewer.is_running():
                break
            T_t=SO_101_fk(q_t)
            p=T_t[:3, 3]
            pitch=pitch_of(T_t[:3, :3])
            res=ik_pitch(p, pitch, q0=q)
            with viewer.lock():
                g=viewer.user_scn.geoms[0]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0, 0]), p, np.eye(3).flatten(), np.array([0.1, 0.8, 0.2, 0.6] if res.success else [0.9, 0.1, 0.1, 0.6], dtype=np.float32))
                viewer.user_scn.ngeom=1
            if not res.success:
                print(f"Target {k}: IK Failed ({res.status}), SKIPPED")
                continue
            move_sim(m, d, viewer, convert_to_dictionary(list(req.q)+[0.0]), seconds)
            hold_sim(m, d, viewer, 0.5)
            p_sim=d.site_xpos[site_id].copy()
            print(f"Target {k}: p={np.round(p*1000, 1)} mm pitch {np.degrees(pitch):.1f} deg  IK {res.iterations} iters, simulated arm: TCP error {np.linalg.norm(p_sim-p)*1000:.2f} mm")
            q=res.q
        print("Done, close viewer window")
        while viewer.is_running():
            time.sleep(0.1)

def radial(p):
    return float(np.hypot(*(p[:2]-PAN_AXIS_XY)))

def facing(q, p):
    return bool(np.cos(heading_of(p)+q[0])>0)

def in_workspace(m, d, q):
    p=SO_101_fk(q)[:3, 3]
    return p[2]>=Z_MIN and radial(q)>=R_MIN and facing(q, p) and not contacts(m, d, q)

def sample_targets(m, d, lim, n, seed, workspace):
    rng=np.random.default_rng(seed)
    out=[]
    drawn=0
    while len(out)<n:
        q=rng.uniform(lim[:, 0], lim[:, 1])
        drawn+=1
        if not workspace or in_workspace(m, d, q):
            out.append(q)
    return np.array(out), drawn

def rotation_angle(R_a, R_b):
    R=R_a@R_b.T
    return float(np.arctan2(np.linalg.norm(vee_so3(R-R.T))/2.0, (np.trace(R)-1.0)/2.0))

def failure_class(q, res, lim):
    if res.success:
        return ""
    p=SO_101_fk(q)[:3, 3]
    if radial(p)<0.015:
        return "pan_axis"
    if np.linalg.svd(position_jacobian(q), compute_uv=False)[-1]<0.03:
        return "near_singular"
    if np.any(np.isclose(res.q, lim[:,0]) | np.isclose(res.q, lim[:, 1])):
        return "joint_limit"
    if res.pos_err<0.001 and res.ori_err<0.01:
        return "near_miss"
    return "local_minimum"

def run_case(m, d, site_id, lim, q, mode, method, start, rng):
    T=SO_101_fk(q)
    kw={
        "method": method
    }
    if mode=="pitch":
        kw["pitch"]=pitch_of(T[:3, :3])
    if mode=="pose":
        kw["R_dest"]=T[:3, :3]
    if start=="warm":
        q0=np.clip(q+rng.normal(0, NOISE, N_JOINTS), lim[:, 0], lim[:, 1])
        res=inverse_kinematics(T[:3, 3], q0=q0, mode=mode, restarts=0, **kw)
    else:
        res=inverse_kinematics(T[:3, 3], q0=None, mode=mode, **kw)
    T_mj=mujoco_tcp(m, d, site_id, res.q)

    pos_mj=np.linalg.norm(T_mj[:3, 3]-T[:3, 3])
    if mode=="pitch":
        ori_mj=abs(pitch_of(T_mj[:3, 3]-T[:3, :3]))
    elif mode=="pose":
        ori_mj=rotation_angle(T_mj[:3, :3], T[:3, :3])
    else:
        ori_mj=0.0
    p=T[:3 ,3]
    row={
        "mode": mode,
        "method": method,
        "start": start,
        "success": int(res.success),
        "status": res.status,
        "fail_class": failure_class(q, res, lim),
        "iterations": res.iterations,
        "restarts": res.restarts,
        "time_ms": res.time_s*1000,
        "pos_err_m": pos_mj,
        "ori_err_m": ori_mj,
        "solver_pos_err_m": res.pos_err,
        "solver_ori_err_m": res.ori_err,
        "sigma": res.sigma,
        "min_limit_dist_rad": float(np.min(np.minimum(res.q-lim[:, 0], lim[:,1]-res.q))),
        "r_m": radial(p),
        "z_m": float(p[2]),
        "pitch_deg": float(np.degrees(pitch_of(T[:3, :3])))
    }
    row.update({f"q_true_{j}": float(v) for j, v in zip(JOINT_NAMES, q)})
    row.update({f"q_ik_{j}": float(v) for j, v in zip(JOINT_NAMES, res.q)})
    return row

def run_validation(m, d, site_id, lim, sets, seed):
    rows=[]
    for name, Q in sets.items():
        t0=time.perf_counter()
        for start in ("warm", "cold"):
            for mode, method in CASES:
                rng=np.random.default_rng(seed)
                for k, q in enumerate(Q):
                    rows.append({
                        "set": name,
                        "target": k,
                        **run_case(m, d, site_id, lim, q, mode, method, start, rng)
                    })
                    print(f"{name:9s} {start:4s} {mode:8s} {method:8s} done ({time.perf_counter()-t0:6.1f} s)")
    return rows

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["set"], r["mode"], r["method"], r["start"])].append(r)
    out=[]
    for (s, mode, method, start), g in groups.items():
        ok=[r for r in g if r["success"]]
        pos_err=np.array([r["pos_err_m"] for r in ok]) if ok else np.array([np.nan])
        ori_err=np.array([r["ori_err_m"] for r in ok]) if ok else np.array([np.nan])
        iteration=np.array([r["iterations"] for r in g])
        time_ms=np.array([r["time_ms"] for r in g])
        fails=Counter(r["fail_class"] for r in g if not r["success"])
        out.append({
            "set": s,
            "mode": mode,
            "start": start,
            "n": len(g),
            "success": len(ok),
            "success_pct": 100.0*len(ok)/len(g),
            "pos_err_median_m": np.median(pos_err),
            "pos_err_p95_m": np.percentile(pos_err, 95),
            "pos_err_max_m": pos_err.max(),
            "ori_err_median_rad": np.median(ori_err),
            "ori_err_max_rad": ori_err.max(),
            "iter_mean": iteration.mean(),
            "iter_median": np.median(iteration),
            "iter_p95": np.percentile(iteration, 95),
            "time_ms_mean": time_ms.mean(),
            "time_ms_median": np.median(time_ms),
            "time_ms_p95": np.percentile(time_ms, 95),
            "restarts_mean": np.mean([r["restarts"] for r in g]),
            **{f"fail_{c}": fails.get(c, 0) for c in FAIL_CLASSES}
        })

def singularity_sweep(lim, steps=400, before=0.10, beyond=0.01):
    q_s=np.array([0.0, 0.0, np.radians(-78.44), 0.0, 0.0])
    p_s=SO_101_fk(q_s)[:3, 3]
    s=np.linspace(-before, beyond, steps+1)
    u=(p_s-SHOULDER_LIFT_POINT)/np.linalg.norm(p_s-SHOULDER_LIFT_POINT)
    P=p_s+s[:,None]*u
    q_start=inverse_kinematics(P[0], q0=np.array([0, 0, -0.9, 0, 0]), mode="position").q
    inverses={
        "pseudoinverse": dict(eps=0.0, lam_max=0.0),
        "fixed_DLS": dict(eps=1e9, lam_max=IK_LAMBDA_POS),
        "adaptive_DLS": dict(eps=IK_EPS_POS, lam_max=IK_LAMBDA_POS)
    }
    rows=[]
    for name, kw in inverses.items():
        q=q_start.copy()
        for k in range(steps):
            p=SO_101_fk(q)[:3, 3]
            J=position_jacobian(q, p)
            Jp, sig, lam, _=damped_pseudo_inverse(J, **kw)
            dq=Jp@(P[k+1]-p)
            q=np.clip(q+dq, lim[:,0], lim[:,1])
            rows.append({
                "inverse": name,
                "step": k,
                "s_m": s[k+1],
                "dq_norm_rad": float(np.linalg.norm(dq)),
                "err_m": float(np.linalg.norm(P[k+1]-SO_101_fk(q)[:3, 3])),
                "sigma": sig,
                "lambda": lam
            })
    return rows

def save_csv(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        print(f"Error saving data in file {path}")
        return
    with open(path, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

def plot_errors(data, path):
    pass

def plot_failures(data, path):
    pass

def plot_iterations(data, path):
    pass

def plot_singularity(data, path):
    pass

def args_parse():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="Number of targets per set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", type=int, default=0, metavar="N", help="Show number of IK solutions in MuJoCo viewer")
    parser.add_argument("--view-only", action="store_true", help="Skip validation")
    return parser.parse_args()

def main():
    args=args_parse()
    m, d, site_id=mujoco_starting(str(SCENE_XML))
    lim=default_limits()
    if not args.view_only:
        Q_box, _=sample_targets(m, d, lim, args.n, args.seed, workspace=False)
        Q_ws, drawn=sample_targets(m, d, lim, args.n, args.seed, workspace=True)
        print(f"Targets: {args.n} box, {args.n} workspace (accepted {args.n}/{drawn}={100*args.n/drawn:.1f}% of random configs)")
        rows=run_validation(m, d, site_id, lim, {
            "box": Q_box,
            "workspace": Q_ws
        }, args.seed)
        summary=summarize(rows)
        sweep=singularity_sweep(lim)
        
        #TABLES
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_csv(rows, DATA_DIR/"ik_validation.csv")
        save_csv(summary, DATA_DIR/"ik_validation_summary.csv")
        save_csv(sweep, DATA_DIR/"ik_singularity.csv")
        #FIGURES
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plot_errors(rows, FIG_DIR/"ik_errors.png")
        plot_iterations(rows, FIG_DIR/"ik_iterations.png")
        plot_failures(rows, FIG_DIR/"ik_failures.png")
        plot_singularity(rows, FIG_DIR/"ik_singularity.png")
        
        print(f"\n{'set':9s} {'mode':8s} {'method':8s} {'start':5s} {'success':>9s} "
              f"{'pos err max':>12s} {'ori err max':>12s} {'it med':>6s} {'it p95':>6s} {'ms med':>7s} fails")
        for s in summary:
            fails = {k[5:]: v for k, v in s.items() if k.startswith("fail_") and v}
            print(f"{s['set']:9s} {s['mode']:8s} {s['method']:8s} {s['start']:5s} "
                  f"{s['success']:4d}/{s['n']:<4d} {s['pos_err_max_m']:12.2e} {s['ori_err_max_rad']:12.2e} "
                  f"{s['iter_median']:6.0f} {s['iter_p95']:6.0f} {s['time_ms_median']:7.2f} {fails or ''}")
        print("\nSingularity sweep:")
        for name in ("pseudoinverse", "fixed DLS", "adaptive DLS"):
            g=[r for r in sweep if r["inverse"]==name]
            inside=[r for r in g if r["s_m"]<=0]
            print(f"    {name:14s} max |dq| {max(r['dq_norm_rad'] for r in g):9.3f} rad "
                  f"max err inside reach {max(r['err_m'] for r in inside)*1000:7.3f} mm "
                  f"err at -5 cm {g[int(np.argmin([abs(r['s_m']+0.05) for r in g]))]['err_m']*1000:7.3f} mm "
                  f"final err {g[-1]['err_m']*1000:7.2f} mm"      
            )
        
    if args.view or args.view_only:
        Q_view, _=sample_targets(m, d, lim, args.view, args.seed, workspace=True)
        view_solutions(m, d, Q_view)
        print("Ended view")

if __name__=="__main__":
    main()