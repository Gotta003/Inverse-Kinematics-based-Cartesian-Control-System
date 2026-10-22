"""
fk_validation.py
Forward-Kinematics against Mujoco

Outputs:
    results/data/fk_val.csv         per-config errors
    results/figures/fk_err.png      position/orientation graph
    results/figures/fk_poses.png    3D graph showing poses
    
Run:
    python fk_validation.py         Run validation without showing anything
    python fk_validation.py --show  Shows also steps in Mujoco
"""
import sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from src.forward_kinematics import SO_101_fk
from src.utils.constants import POSES
from utils.so101_mujoco_utils import mujoco_starting, mujoco_tcp
from src.utils.helper import derive_limits, position_error, rotation_error
from src.utils.constants import N_JOINTS, JOINT_NAMES

SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"
DATA_DIR=REPO_ROOT/"results"/"data"
FIG_DIR=REPO_ROOT/"results"/"figures"

def build_dataset(m, d, site_id, n_random, seed):
    rng=np.random.default_rng(seed)
    configs=[(name, np.array(q)) for name, q in POSES.items()]
    JOINT_LIMITS=derive_limits(m, N_JOINTS)
    lowers=np.array([JOINT_LIMITS[name][0] for name in JOINT_NAMES])
    uppers=np.array([JOINT_LIMITS[name][0] for name in JOINT_NAMES])
    configs+=[
        (f"random_{k:03d}", rng.uniform(lowers, uppers)) for k in range(n_random)
    ]
    rows=[]
    for name, q in configs:
        T_ours=SO_101_fk(q)
        T_ref=mujoco_tcp(m, d, site_id, q)
        rows.append({
            "name": name,
            **{f"q_{j}": q[i] for i, j in enumerate(JOINT_NAMES)},
            **{f"fk_{a}": T_ours[i, 3] for i, a in enumerate("xyz")},
            **{f"mj_{a}": T_ref[i, 3] for i, a in enumerate("xyz")},
            "pos_err_m": position_error(T_ours, T_ref),
            "rot_err_rad": rotation_error(T_ours[:3, :3], T_ref[:3, :3]),
        })
    return rows

def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def view_poses(m, d, site_id, seconds_per_pose=2.5):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for name, q, in POSES.items():
            if not viewer.is_running():
                break
            q=np.array(q)
            T_ref=mujoco_tcp(m, d, site_id, q)
            p_ours=SO_101_fk(q)[:3, 3]
            print(f"{name:13s} FK {np.round(p_ours*1000, 2)} mm | MuJoco {np.round(T_ref[:3, 3], 2)} mm")
            viewer.sync()
            time.sleep(seconds_per_pose)

def parse_args():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_random", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true", help="Show poses in Mujoco viewer")
    return parser.parse_args()

def main():
    args=parse_args()
    m, d, site_id=mujoco_starting(str(SCENE_XML))
    rows=build_dataset(m, d, site_id, args.n_random, args.seed)
    save_csv(rows, DATA_DIR/"fk_validation.csv")
    
    if args.view:
        view_poses(m, d, site_id)
    
if __name__=="__main__":
    main()