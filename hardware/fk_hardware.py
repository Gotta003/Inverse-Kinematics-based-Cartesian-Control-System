"""
fk_hardware.py
Forward-Kinematics check on physical SO-101

For each pose the script will print the TCP positions, to enable measures and same those in results/data/fk_hardware.csv

Measures:
    x_rel: Forward
    y_rel: Left
    z_rel: Height above table

Modes:
    commanded (default): The robot moves, with torque ON, to a list of
    predefined joint poses (checked in MuJoCo: no collisions, TCP above the
    table)
    passive: Torque OFF. You move the arm by hand and press Enter to
    read it.

Usage:
python hardware/fk_hardware.py --port /dev/ttyACM0 #LINUX
"""

import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import argparse
import csv
import os

import numpy as np

from src.arm import Arm
from src.forward_kinematics import SO_101_fk
from src.utils.constants import (
    JOINT_NAMES,
    LIMIT_MARGIN,
    MOVE_TIME,
    N_JOINTS,
    POSES,
    POSITIONS,
    SETTLE_TIME,
)
from src.utils.helper import derive_limits

OUT_CSV=REPO_ROOT/"results"/"data"/"fk_hardware.csv"
SCENE_XML = REPO_ROOT / "model_files" / "scene.xml"
PAN_FOOT=np.array([POSITIONS[0, 0], POSITIONS[0, 1], 0.0])

def save_rows(rows):
    if not rows:
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    errs=[r["pos_err_mm"] for r in rows if r["pos_err_mm"]!=""]
    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")
    if errs:
        print(f"FK vs ruler: max {max(errs):.1f} mm, mean {np.mean(errs):.1f} mm over {len(errs)} measured poses")
    
def load_limits():
    from utils.so101_mujoco_utils import mujoco_starting
    m,_,_=mujoco_starting(str(SCENE_XML))
    limits=derive_limits(m, N_JOINTS)
    return np.array([limits[j] for j in JOINT_NAMES])
        
def within_limits(q, limits):
    return bool(np.all(q>=limits[:,0]+LIMIT_MARGIN) and np.all(q<=limits[:,1]-LIMIT_MARGIN))

def ask_measure():
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

def report_pose(name, q_send, q_recv):
    p_base=SO_101_fk(q_recv)[:3,3]
    p_rel=p_base-PAN_FOOT
    print(f"q measured [def]: {np.round(np.degrees(q_recv), 1)}")
    if q_send is not None:
        print(f"q measured send [def]: {np.round(np.degrees(q_recv-q_send), 2)}")
    print(f"FK TCP, base frame [mm]: {np.round(p_base*1000, 1)}")
    print(f"FK TCP, relative pan foot [mm]: {np.round(p_rel*1000, 1)}")
    p_recv=ask_measure()
    row={"pose": name}
    for i, j in enumerate(JOINT_NAMES):
        row[f"q_cmd_{j}"]="" if q_send is None else q_send[i]
        row[f"q_recv_{j}"]=q_recv[i]
    for i, a in enumerate("xyz"):
        row[f"fk_rel_{a}_mm"]=p_rel[i]*1000
        row[f"recv_rel_{a}_mm"]="" if p_recv is None else p_recv[i]*1000
    if p_recv is not None:
        err=np.linalg.norm(p_rel-p_recv)
        row["pos_err_mm"]=err*1000
        print(f"FK vs ruler error: {err*1000:.1f} mm")
    else:
        row["pos_err_mm"]=""
    return row
        

def run_commanded(arm, limits):
    rows=[]
    for name, q in POSES.items():
        q=np.array(q, dtype=float)
        if not within_limits(q, limits):
            print(f"Skipping {name}: too close to limits")
            continue
        cmd=input(f"\nNext pose '{name}' {np.round(np.degrees(q), 1)} deg. Enter=move, s=skip, q=quit: ").strip().lower()
        if cmd=="q":
            break
        if cmd=="s":
            continue
        arm.move(dict(zip(JOINT_NAMES, q)), MOVE_TIME)
        arm.hold(SETTLE_TIME)
        rows.append(report_pose(name, q, arm.read_arm()))
    return rows

def run_passive(arm):
    input("\nHOLD ARM WITH HAND, then press Enter to torque OFF...")
    arm.close()
    print("Tip: first put the arm in the 'L' (zero) pose with q close to 0")
    rows, k=[], 0
    while True:
        cmd=input("\nPose the arm by hand. Enter=read, q=quit: ").strip().lower()
        if cmd=="q":
            break
        rows.append(report_pose(f"manual_{k:02d}", None, arm.read_arm()))
        k+=1
    return rows

def args_parse():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Serial Port")
    parser.add_argument("--robot", default="jarvis", help="name of calibration in calibration_files/<robot>.json")
    parser.add_argument("--passive", action="store_true", help="torque OFF, move free by hand")
    return parser.parse_args()

def main():
    args=args_parse()
    limits=load_limits()
    arm=Arm.connect(args.robot, args.port)
    start=arm.read()
    print(f"Start configuration [deg]: {np.round(np.degrees([start[j] for j in JOINT_NAMES]), 1)}")
    rows=[]
    try:
        if args.passive:
            rows=run_passive(arm)
        else:
            rows=run_commanded(arm, limits)
        if not args.passive:
            input("\nEnter to return to start pose...")
            arm.move({j: start[j] for j in JOINT_NAMES}, MOVE_TIME)
            arm.hold(SETTLE_TIME)
    except KeyboardInterrupt:
        if not args.passive:
            arm.freeze()
        print("\nInterrupted: arm frozen")
    finally:
        save_rows(rows)
        input("\nHOLD ARM, then press Enter to release torque")
        arm.close()
    
if __name__=="__main__":
    main()