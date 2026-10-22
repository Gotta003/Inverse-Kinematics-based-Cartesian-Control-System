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
REPO_ROOT=Path(__file__).resolve().parent.parent)
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
from utils.so101_mujoco_utils import mujoco_starting

SCENE_XML=REPO_ROOT/"model_files"/"scheme.xml"
DATA_DIR=REPO_ROOT/"results"/"data"
FIG_DIR=REPO_ROOT/"results"/"figures"

def build_dataset(m, d, site_id, n_random, seed):
    return

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
    
if __name__=="__main__":
    main()