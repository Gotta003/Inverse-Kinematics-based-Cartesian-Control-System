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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))