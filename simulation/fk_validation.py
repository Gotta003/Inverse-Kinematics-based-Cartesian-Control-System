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

from src.forward_kinematics import SO_101_fk, matrix_exp
from src.utils.constants import JOINT_NAMES, N_JOINTS, OMEGA, POSES, POSITIONS, M, V
from src.utils.helper import derive_limits, position_error, rotation_error
from utils.so101_mujoco_utils import mujoco_starting, mujoco_tcp

SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"
DATA_DIR=REPO_ROOT/"results"/"data"
FIG_DIR=REPO_ROOT/"results"/"figures"

def sample_configs(limits, n, seed):
    rng=np.random.default_rng(seed)
    lowers=np.array([limits[name][0] for name in JOINT_NAMES])
    uppers=np.array([limits[name][1] for name in JOINT_NAMES])
    return rng.uniform(lowers, uppers, size=(n, N_JOINTS))

def compare(m, d, site_id, q):
    T_ours=SO_101_fk(q)
    T_ref=mujoco_tcp(m, d, site_id, q)
    return (T_ours, T_ref, position_error(T_ours, T_ref), rotation_error(T_ours[:3, :3], T_ref[:3, :3]))

def named_pose_rows(m, d, site_id):
    rows=[]
    for name, q in POSES.items():
        T_ours, T_ref, e_p, e_r=compare(m, d, site_id, np.array(q, dtype=float))
        rows.append({
            "pose": name,
            **{f"q_{j}_deg": round(float(np.degrees(v)), 1) for j, v in zip(JOINT_NAMES, q)},
            **{f"fk_{a}_mm": round(T_ours[i, 3]*1000, 2) for i, a in enumerate("xyz")},
            **{f"mj_{a}_mm": round(T_ref[i, 3]*1000, 2) for i, a in enumerate("xyz")},
            "pos_err_um": e_p*1e6,
            "rot_err_urad": e_r*1e6,
        })
    return rows

def random_rows(m, d, site_id, Q):
    rows=[]
    for k, q in enumerate(Q):
        T_ours, T_ref, e_p, e_r=compare(m, d, site_id, q)
        rows.append({
            "sample": k,
            **{f"q_{j}": v for j, v in zip(JOINT_NAMES, q)},
            **{f"fk_{a}": T_ours[i, 3] for i, a in enumerate("xyz")},
            **{f"mj_{a}": T_ref[i, 3] for i, a in enumerate("xyz")},
            "pos_err_m": e_p,
            "rot_err_rad": e_r
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
            print(f"{name:13s} FK {np.round(p_ours*1000, 2)} mm | MuJoco {np.round(T_ref[:3, 3]*1000, 2)} mm")
            viewer.sync()
            time.sleep(seconds_per_pose)

def joint_points(q):
    pts=[np.array([POSITIONS[0, 0], POSITIONS[0, 1], 0.0])]
    T=np.eye(4)
    for i in range(N_JOINTS):
        pts.append((T@np.append(POSITIONS[i], 1.0))[:3])
        T=T@matrix_exp(OMEGA[i], V[i], q[i])
    pts.append((T@M)[:3, 3])
    return np.array(pts)

def plot_named_poses(m, d, site_id, path):
    fig, ax=plt.subplots(figsize=(7, 5))
    ax.axhspan(-0.05, 0.0, color="0.85", zorder=0)
    ax.axhline(0.0, color="0.4", lw=1)
    ax.text(0.47, -0.03, "table", ha="right", fontsize=8, color="0.3")
    k=0
    for name, q in POSES.items():
        if q[0]!=0.0:
            continue
        q=np.array(q, dtype=float)
        pts=joint_points(q)
        p_ref=mujoco_tcp(m, d, site_id, q)[:3, 3]
        ax.plot(pts[:,0], pts[:,2], "-o", color=f"C{k}", ms=3, label=name)
        ax.plot(p_ref[0], p_ref[2], "x", color=f"C{k}", ms=10, mew=2)
        k+=1
    ax.plot([], [], "kx", ms=8, mew=2, label="MuJoCo TCP")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Forward kinematics (lines) vs MuJoCo TCP (×), side view")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 0.48)
    ax.set_ylim(-0.05, 0.48)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_workspace(Q, path):
    P=np.array([SO_101_fk(q)[:3, 3] for q in Q])
    r=np.hypot(P[:,0]-POSITIONS[0,0], P[:,1]-POSITIONS[0,1])
    fig, axes=plt.subplots(1,2, figsize=(11, 4.6))
    axes[0].scatter(r, P[:,2], s=3, c="tab:blue", alpha=0.4)
    axes[0].axhline(0.0, color="k", lw=1)
    axes[0].text(0.01, -0.02, "table(z=0)", fontsize=8, va="top")
    axes[0].set_xlabel("radial distance from pan axis [m]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("Reachable Positions")
    
    sc=axes[1].scatter(P[:,0], P[:,1], s=3, c=P[:,2], cmap="viridis")
    fig.colorbar(sc, ax=axes[1], label="TCP height z [m]")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title("Top view")
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle(f"TCP positions of {len(Q)} random configs in joint limits")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def parse_args():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_random", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true", help="Show poses in Mujoco viewer")
    return parser.parse_args()

def main():
    args=parse_args()
    m, d, site_id=mujoco_starting(str(SCENE_XML))
    limits=derive_limits(m, N_JOINTS)
    Q=sample_configs(limits, args.n_random, args.seed)
    named=named_pose_rows(m, d, site_id)
    rand=random_rows(m, d, site_id, Q)
    
    save_csv(named, DATA_DIR/"fk_named_poses.csv")
    save_csv(rand, DATA_DIR/"fk_validation.csv")
    plot_named_poses(m, d, site_id, FIG_DIR/"fk_named_poses.png")
    plot_workspace(Q, FIG_DIR/"fk_workspace.png")
    
    pos = np.array([row["pos_err_m"] for row in rand])
    rot = np.array([row["rot_err_rad"] for row in rand])
    print("Named poses (FK vs MuJoCo):")
    for row in named:
        print(f"  {row['pose']:14s} FK [{row['fk_x_mm']:7.2f} {row['fk_y_mm']:7.2f} "
              f"{row['fk_z_mm']:7.2f}] mm   err {row['pos_err_um']:.3f} um")
    print(f"\nRandom configurations : {len(rand)}")
    print(f"  position error  max {pos.max():.2e} m    mean {pos.mean():.2e} m")
    print(f"  rotation error  max {rot.max():.2e} rad  mean {rot.mean():.2e} rad")
    print(f"\nSaved data to {DATA_DIR} and figures to {FIG_DIR}")
    
    if args.view:
        view_poses(m, d, site_id)
    
if __name__=="__main__":
    main()