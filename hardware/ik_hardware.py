"""
ik_hardware.py
Inverse-Kinematics check on physical SO-101

For each Cartesian target the script solves IK, checks the full chain in MuJoCo
and only if every check passes it moves the arm. If any check fails the reasons
are printed and nothing moves. Measures are saved in results/data/ik_hardware.csv

Chain checks (before any movement):
    1) IK converged
    2) Solution inside joint limits (MuJoCo XML + calibration, with margin)
    3) MuJoCo TCP at the solution matches the target
    4) No floor/self contacts at the solution
    5) Joint-space path from the current pose (the one Arm.move follows)
       has no new contacts and keeps the TCP above the table

Measures:
    x_rel: Forward
    y_rel: Left
    z_rel: Height above table

Plans:
    named (default): list of predefined Cartesian targets relative pan foot
    random: targets drawn by rejection (random q in limits, TCP above the
    table, in front of the arm, no contacts), then solved by IK from the
    current pose

Modes:
    real (default): torque ON, the arm moves to every target that passes the checks
    dry run: only IK + MuJoCo checks, the arm never moves (no port needed)

Usage:
python hardware/ik_hardware.py --port /dev/ttyACM0 #LINUX
python hardware/ik_hardware.py --dry-run
python hardware/ik_hardware.py --port /dev/ttyACM0 --plan random --n 5
"""

import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import argparse
import csv

import numpy as np

from src.forward_kinematics import SO_101_fk
from src.inverse_kinematics import METHODS, heading_of, inverse_kinematics, pitch_of
from src.utils.constants import (
    IK_POS_TOL,
    JOINT_NAMES,
    MOVE_TIME,
    N_JOINTS,
    POSITIONS,
    RND_SEED,
    SETTLE_TIME,
)
from src.utils.limits import inv_kin_limits
from src.utils.sim_checks import contacts
from utils.so101_mujoco_utils import mujoco_starting, mujoco_tcp

OUT_CSV=REPO_ROOT/"results"/"data"/"ik_hardware.csv"
SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"
PAN_FOOT=np.array([POSITIONS[0, 0], POSITIONS[0, 1], 0.0])

Z_MIN=0.03          # m, minimum TCP height above table, at target and along the path
R_MIN=0.12          # m, minimum radial distance from pan axis (random plan)
SIM_TOL=1e-3        # m, FK/IK target vs MuJoCo TCP
PATH_STEPS=50       # samples on the joint-space path
LIMIT_TOL=1e-9

# name: ([x, y, z] mm relative pan foot, pitch deg (+ = tool down) or None for position only)
TARGETS={
    "front_mid": ([250.0, 0.0, 150.0], 30.0),
    "front_low": ([280.0, 0.0, 80.0], 60.0),
    "front_high": ([220.0, 0.0, 300.0], 0.0),
    "near_down": ([180.0, 0.0, 100.0], 70.0),
    "far": ([320.0, 0.0, 120.0], 20.0),
    "left": ([220.0, 120.0, 120.0], 45.0),
    "right": ([220.0, -120.0, 120.0], 45.0),
    "position_only": ([260.0, 60.0, 180.0], None),
}

def save_rows(rows):
    """Write the rows to OUT_CSV and print a summary: targets that passed the checks,
    poses moved, target vs FK(read q) error and target vs ruler error"""
    if not rows:
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")
    moved=[r for r in rows if r["moved"]]
    track=[r["track_err_mm"] for r in moved]
    errs=[r["pos_err_mm"] for r in moved if r["pos_err_mm"]!=""]
    print(f"Targets: {len(rows)}, passed checks: {sum(r['checks_ok'] for r in rows)}, moved: {len(moved)}")
    if track:
        print(f"Target vs FK(read q): max {max(track):.1f} mm, mean {np.mean(track):.1f} mm over {len(track)} poses")
    if errs:
        print(f"Target vs ruler: max {max(errs):.1f} mm, mean {np.mean(errs):.1f} mm over {len(errs)} measured poses")

def within_limits(q, limits):
    """True if every joint of q lies inside [limits[:,0], limits[:,1]] (tolerance LIMIT_TOL)"""
    return bool(np.all(q>=limits[:,0]-LIMIT_TOL) and np.all(q<=limits[:,1]+LIMIT_TOL))

def radial(p):
    """Horizontal distance [m] of the point p from the shoulder_pan axis"""
    return float(np.hypot(*(p[:2]-PAN_FOOT[:2])))

def facing(q, p):
    """True if the arm at q points towards p (no reach-over-the-head solutions)"""
    return bool(np.cos(heading_of(p)+q[0])>0)

def named_plan():
    """Convert TARGETS into a plan of (name, p_dest base frame [m], pitch [rad] or None)"""
    return [(name, np.array(p)/1000.0+PAN_FOOT, None if pitch is None else np.radians(pitch)) for name, (p, pitch) in TARGETS.items()]

def random_plan(m, d, limits, n, seed):
    """Rejection sampling: draw q uniformly in limits and keep it only if the TCP is
    above Z_MIN, at least R_MIN from the pan axis, in front of the arm and without contacts.
    Returns n targets as (name, p_dest [m], pitch [rad]) taken from FK(q)"""
    rng=np.random.default_rng(seed)
    plan, drawn=[], 0
    while len(plan)<n:
        q=rng.uniform(limits[:,0], limits[:,1])
        drawn+=1
        T=SO_101_fk(q)
        p=T[:3,3]
        if p[2]<Z_MIN or radial(p)<R_MIN or not facing(q, p) or contacts(m, d, q):
            continue
        plan.append((f"random_{len(plan):02d}", p, pitch_of(T[:3,:3])))
    print(f"Random plan: {n} targets accepted out of {drawn} draws")
    return plan

def path_reasons(m, d, q_start, q_goal):
    """Check the linear joint-space path q_start -> q_goal (the one Arm.move follows)
    on PATH_STEPS samples. Contacts already present at q_start are ignored.
    Returns the reason of the first failure (new contact or TCP below Z_MIN), empty list if safe"""
    reasons=[]
    start_contacts=contacts(m, d, q_start)
    for s in np.linspace(0.0, 1.0, PATH_STEPS+1)[1:]:
        q=(1-s)*q_start+s*q_goal
        new=contacts(m, d, q)-start_contacts
        if new:
            reasons.append(f"path contact at s={s:.2f}: {sorted(new)}")
            break
        z=SO_101_fk(q)[2,3]
        if z<Z_MIN:
            reasons.append(f"path TCP too low at s={s:.2f}: z={z*1000:.1f} mm < {Z_MIN*1000:.0f} mm")
            break
    return reasons

def check_chain(m, d, site_id, limits, q_start, p_dest, pitch, method):
    """Full chain before any movement: IK from q_start (position mode if pitch is None,
    pitch mode otherwise), target height, convergence, joint limits, MuJoCo TCP vs target,
    contacts at the solution and, if all pass, the joint-space path.
    Returns the IK result, the list of reasons (empty = safe to move) and the MuJoCo error [m]"""
    mode="position" if pitch is None else "pitch"
    res=inverse_kinematics(p_dest, q0=q_start, mode=mode, pitch=pitch, method=method, limits=limits)
    reasons=[]
    if p_dest[2]<Z_MIN:
        reasons.append(f"target too low: z={p_dest[2]*1000:.1f} mm < {Z_MIN*1000:.0f} mm")
    if not res.success:
        reasons.append(f"IK {res.status}: pos_err {res.pos_err*1000:.2f} mm, ori_err {np.degrees(res.ori_err):.2f} deg")
    if not within_limits(res.q, limits):
        reasons.append(f"solution outside joint limits: {np.round(np.degrees(res.q), 1)} deg")
    p_sim=mujoco_tcp(m, d, site_id, res.q)[:3,3]
    sim_err=np.linalg.norm(p_sim-p_dest)
    if sim_err>max(SIM_TOL, res.pos_err+IK_POS_TOL):
        reasons.append(f"MuJoCo TCP off target by {sim_err*1000:.2f} mm")
    final=contacts(m, d, res.q)
    if final:
        reasons.append(f"contacts at solution: {sorted(final)}")
    if not reasons:
        reasons+=path_reasons(m, d, q_start, res.q)
    return res, reasons, sim_err

def ask_measure():
    """Ask the measured TCP [x, y, z] in mm relative pan foot.
    Returns it in m, or None if skipped or not parsable"""
    raw=input("Measured TCP [x, y, z] mm relative pan foot (Enter to skip):").strip()
    if not raw:
        return None
    try:
        vals=np.array([float(v) for v in raw.replace(",", " ").split()])
        if vals.size!=3:
            raise ValueError
        return vals/1000.0
    except ValueError:
        print("Could not parse, measure skip")
        return None

def report_target(name, p_dest, pitch, res, reasons, sim_err, q_recv):
    """Build the CSV row of a target: target, IK result, check outcome and commanded q.
    If the arm moved (q_recv not None) also print the reached pose, the target vs FK(q_recv)
    error and ask for the ruler measure"""
    row={"pose": name}
    for i, a in enumerate("xyz"):
        row[f"target_rel_{a}_mm"]=(p_dest-PAN_FOOT)[i]*1000
    row["target_pitch_deg"]="" if pitch is None else np.degrees(pitch)
    row["ik_status"]=res.status
    row["ik_iterations"]=res.iterations
    row["ik_pos_err_mm"]=res.pos_err*1000
    row["ik_ori_err_deg"]=np.degrees(res.ori_err)
    row["sigma_min"]=res.sigma
    row["sim_err_mm"]=sim_err*1000
    row["checks_ok"]=not reasons
    row["reasons"]=" | ".join(reasons)
    row["moved"]=q_recv is not None
    for i, j in enumerate(JOINT_NAMES):
        row[f"q_cmd_{j}"]=res.q[i]
        row[f"q_recv_{j}"]="" if q_recv is None else q_recv[i]
    row["track_err_mm"]=""
    for a in "xyz":
        row[f"fk_rel_{a}_mm"]=""
        row[f"recv_rel_{a}_mm"]=""
    row["pos_err_mm"]=""
    if q_recv is None:
        return row
    p_base=SO_101_fk(q_recv)[:3,3]
    p_rel=p_base-PAN_FOOT
    track=np.linalg.norm(p_base-p_dest)
    print(f"q measured [deg]: {np.round(np.degrees(q_recv), 1)}")
    print(f"q measured-send [deg]: {np.round(np.degrees(q_recv-res.q), 2)}")
    print(f"FK TCP, relative pan foot [mm]: {np.round(p_rel*1000, 1)}")
    print(f"Target vs FK(read q) error: {track*1000:.1f} mm")
    row["track_err_mm"]=track*1000
    for i, a in enumerate("xyz"):
        row[f"fk_rel_{a}_mm"]=p_rel[i]*1000
    p_meas=ask_measure()
    if p_meas is not None:
        for i, a in enumerate("xyz"):
            row[f"recv_rel_{a}_mm"]=p_meas[i]*1000
        err=np.linalg.norm((p_dest-PAN_FOOT)-p_meas)
        row["pos_err_mm"]=err*1000
        print(f"Target vs ruler error: {err*1000:.1f} mm")
    return row

def run_plan(arm, m, d, site_id, limits, plan, method, q_start):
    """For each target of the plan: solve IK from the current pose, run check_chain and
    move only if every check passes and the user confirms. With arm=None (dry run) nothing
    moves and the next target starts from the previous IK solution. Returns the CSV rows"""
    rows=[]
    q_now=q_start
    for name, p_dest, pitch in plan:
        if arm is not None:
            q_now=arm.read_arm()
        p_txt=np.round((p_dest-PAN_FOOT)*1000, 1)
        pitch_txt="free" if pitch is None else f"{np.degrees(pitch):.1f} deg"
        print(f"\nTarget '{name}' {p_txt} mm rel pan foot, pitch {pitch_txt}")
        res, reasons, sim_err=check_chain(m, d, site_id, limits, q_now, p_dest, pitch, method)
        print(f"IK {res.status}: q {np.round(np.degrees(res.q), 1)} deg, pos_err {res.pos_err*1000:.3f} mm, "
              f"{res.iterations} iters, {res.restarts} restarts, sigma_min {res.sigma:.4f}, MuJoCo err {sim_err*1000:.3f} mm")
        if reasons:
            print("CHECKS FAILED, nothing moves:")
            for r in reasons:
                print(f"  - {r}")
            rows.append(report_target(name, p_dest, pitch, res, reasons, sim_err, None))
            continue
        print("All checks passed")
        if arm is None:
            rows.append(report_target(name, p_dest, pitch, res, reasons, sim_err, None))
            q_now=res.q
            continue
        cmd=input("Enter=move, s=skip, q=quit: ").strip().lower()
        if cmd=="q":
            break
        if cmd=="s":
            rows.append(report_target(name, p_dest, pitch, res, reasons, sim_err, None))
            continue
        arm.move(dict(zip(JOINT_NAMES, res.q)), MOVE_TIME)
        arm.hold(SETTLE_TIME)
        rows.append(report_target(name, p_dest, pitch, res, reasons, sim_err, arm.read_arm()))
    return rows

def args_parse():
    """Command-line arguments, --port is required unless --dry-run"""
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="Serial Port (required unless --dry-run)")
    parser.add_argument("--robot", default="jarvis", help="name of calibration in calibration_files/<robot>.json")
    parser.add_argument("--dry-run", action="store_true", help="IK + MuJoCo checks only, the arm never moves")
    parser.add_argument("--plan", choices=("named", "random"), default="named", help="target plan")
    parser.add_argument("--n", type=int, default=5, help="number of random targets")
    parser.add_argument("--seed", type=int, default=RND_SEED, help="seed of the random plan")
    parser.add_argument("--method", choices=METHODS, default="priority", help="IK method")
    args=parser.parse_args()
    if not args.dry_run and not args.port:
        parser.error("--port is required unless --dry-run")
    return args

def main():
    """Load limits and MuJoCo, build the plan, run it (dry run or real arm), return to
    the start pose, save the CSV and release torque"""
    args=args_parse()
    np.set_printoptions(precision=4, suppress=True)
    limits=inv_kin_limits(args.robot)
    m, d, site_id=mujoco_starting(str(SCENE_XML))
    plan=named_plan() if args.plan=="named" else random_plan(m, d, limits, args.n, args.seed)
    if args.dry_run and not args.port:
        q_start=np.zeros(N_JOINTS)
        print(f"Dry run from zero configuration [deg]: {np.round(np.degrees(q_start), 1)}")
        save_rows(run_plan(None, m, d, site_id, limits, plan, args.method, q_start))
        return
    from src.arm import Arm
    arm=Arm.connect(args.robot, args.port)
    start=arm.read()
    q_start=np.array([start[j] for j in JOINT_NAMES])
    print(f"Start configuration [deg]: {np.round(np.degrees(q_start), 1)}")
    rows=[]
    try:
        rows=run_plan(None if args.dry_run else arm, m, d, site_id, limits, plan, args.method, q_start)
        if not args.dry_run:
            input("\nEnter to return to start pose...")
            arm.move({j: start[j] for j in JOINT_NAMES}, MOVE_TIME)
            arm.hold(SETTLE_TIME)
    except KeyboardInterrupt:
        if not args.dry_run:
            arm.freeze()
        print("\nInterrupted: arm frozen")
    finally:
        save_rows(rows)
        input("\nHOLD ARM, then press Enter to release torque")
        arm.close()

if __name__=="__main__":
    main()
