"""
ik_targets.py

IK on SPECIFIED Cartesian targets, checked in MuJoCo (Phase 2 validation).

    IK_TARGETS      reachable targets: the IK must converge, the MuJoCo TCP must be on the target
    IK_UNREACHABLE  unreachable targets: the IK must NOT converge and must stay bounded, inside the limits

Both lists are in src/utils/constants.py: position [mm] relative to the pan foot (x forward, y left,
z up from the table) and tool pitch [deg] (+ = tool down), or None for position only. Every target is
solved from q0=0 with restarts (cold start), with the priority method and the joint limits of the arm.
The errors are measured on the gripperframe site of MuJoCo, independently of our FK.

Outputs:
    results/data/ik_targets.csv     one row per target
    table on the terminal (used in the report)

Run:
    python simulation/ik_targets.py
    python simulation/ik_targets.py --robot jarvis --view     also move the simulated arm to each solution
"""
import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import time

import numpy as np

from src.forward_kinematics import SO_101_fk
from src.inverse_kinematics import inverse_kinematics, pitch_of
from src.utils.constants import IK_POS_TOL, IK_ROT_TOL, IK_TARGETS, IK_UNREACHABLE, JOINT_NAMES, POSITIONS
from src.utils.limits import inv_kin_limits
from src.utils.sim_checks import contacts
from utils.so101_mujoco_utils import mujoco_starting, mujoco_tcp

SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"
OUT_CSV=REPO_ROOT/"results"/"data"/"ik_targets.csv"
PAN_FOOT=np.array([POSITIONS[0, 0], POSITIONS[0, 1], 0.0])

def to_base(p_mm):
    """Target [mm] relative to the pan foot -> base frame [m]"""
    return np.asarray(p_mm, dtype=float)/1000.0+PAN_FOOT

def solve_target(m, d, site_id, limits, name, p_mm, pitch_deg, reachable):
    """Solve one target (cold start, priority) and measure it in MuJoCo. Returns the CSV row."""
    p_dest=to_base(p_mm)
    mode="position" if pitch_deg is None else "pitch"
    pitch=None if pitch_deg is None else np.radians(pitch_deg)
    res=inverse_kinematics(p_dest, q0=None, mode=mode, pitch=pitch, limits=limits)
    T_mj=mujoco_tcp(m, d, site_id, res.q)
    row={
        "target": name,
        "expected": "reachable" if reachable else "unreachable",
        **{f"target_{a}_mm": v for a, v in zip("xyz", p_mm)},
        "target_pitch_deg": "" if pitch_deg is None else pitch_deg,
        "mode": mode,
        "status": res.status,
        "success": int(res.success),
        "iterations": res.iterations,
        "best_seed": res.restarts,      # index of the seed that gave the result (0 = q0)
        "time_ms": res.time_s*1000,
        "solver_pos_err_mm": res.pos_err*1000,
        "mujoco_pos_err_mm": float(np.linalg.norm(T_mj[:3, 3]-p_dest))*1000,
        "pitch_err_deg": "" if pitch is None else abs(np.degrees(pitch_of(T_mj[:3, :3])-pitch)),
        "sigma_min": res.sigma,
        "contacts": ";".join("/".join(c) for c in sorted(contacts(m, d, res.q))),
        **{f"q_{j}_deg": np.degrees(v) for j, v in zip(JOINT_NAMES, res.q)},
    }
    row["as_expected"]=int(bool(res.success)==reachable)
    return row, res

def print_table(rows):
    """Terminal table of the results (same columns as the report)"""
    print(f"\n{'target':15s} {'expected':11s} {'pitch':>6s} {'status':10s} {'pos err MuJoCo':>15s} {'pitch err':>10s} "
          f"{'iters':>6s} {'seed':>5s} {'ms':>7s}  ok")
    for r in rows:
        pe=f"{r['pitch_err_deg']:8.3f} d" if r["pitch_err_deg"]!="" else "        -"
        pt=f"{r['target_pitch_deg']:4.0f}" if r["target_pitch_deg"]!="" else "free"
        print(f"{r['target']:15s} {r['expected']:11s} {pt:>6s} {r['status']:10s} {r['mujoco_pos_err_mm']:12.4f} mm {pe:>10s} "
              f"{r['iterations']:6d} {r['best_seed']:5d} {r['time_ms']:7.1f}  {'yes' if r['as_expected'] else 'NO'}")
    n_ok=sum(r["as_expected"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} targets behave as expected "
          f"(tolerances: position {IK_POS_TOL*1000:.1f} mm, orientation {np.degrees(IK_ROT_TOL):.3f} deg)")

def view(m, d, rows, results):
    """Move the simulated arm through the reachable solutions in the MuJoCo viewer"""
    try:
        import mujoco.viewer as mj_viewer
        viewer=mj_viewer.launch_passive(m, d)
    except RuntimeError as e:
        print(f"MuJoCo viewer not available ({e}); on macOS run with mjpython")
        return
    from utils.so101_mujoco_utils import convert_to_dictionary, hold_sim, move_sim
    with viewer:
        for r, res in zip(rows, results):
            if not viewer.is_running():
                break
            if r["expected"]=="reachable" and res.success:
                print(f"-> {r['target']}")
                move_sim(m, d, viewer, convert_to_dictionary(list(res.q)+[0.0]), 2.0)
                hold_sim(m, d, viewer, 0.5)
        print("Done, close the viewer window")
        while viewer.is_running():
            time.sleep(0.1)

def main():
    """Solve every reachable and unreachable target, print the table, save the CSV"""
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="jarvis", help="calibration used for the joint limits (none: MuJoCo limits)")
    p.add_argument("--view", action="store_true", help="show the solutions in the MuJoCo viewer")
    a=p.parse_args()
    limits=inv_kin_limits(None if a.robot.lower()=="none" else a.robot)
    m, d, site_id=mujoco_starting(str(SCENE_XML))
    rows, results=[], []
    for reachable, targets in ((True, IK_TARGETS), (False, IK_UNREACHABLE)):
        for name, (p_mm, pitch_deg) in targets.items():
            row, res=solve_target(m, d, site_id, limits, name, p_mm, pitch_deg, reachable)
            rows.append(row)
            results.append(res)
    print_table(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {OUT_CSV}")
    if a.view:
        view(m, d, rows, results)

if __name__=="__main__":
    main()