import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from src.utils.helper import deg_to_rad, vectorial_product

JOINT_NAMES=(
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ALL_JOINT_NAMES=JOINT_NAMES+("gripper",)
N_JOINTS=len(JOINT_NAMES)

# ------------------------------------
# FORWARD KINEMATICS
# ------------------------------------

theta_cal=np.arctan2(0.048660291, 0.998815386)
#theta_cal = 2.789
#theta_cal = deg_to_rad(theta_cal)
M = np.array(
    [
        [1, 0, 0, 0.3913619],
        [0, np.cos(theta_cal), -np.sin(theta_cal), -0.0000113],
        [0, np.sin(theta_cal), np.cos(theta_cal), 0.2264687],
        [0, 0, 0, 1],
    ]
)

OMEGA = np.array(
    [
        [0, 0, -1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [-1, 0, 0],
    ]
)

POSITIONS = np.array(
    [
        [0.0388353, 0, 0],
        [0.0692345, -0.0182778, 0.1166],
        [0.0972345, -0.0182778, 0.2291700],
        [0.2321345, -0.0001778, 0.23437],
        [0.2932345, -0.0001778, 0.23437],
    ]
)

V = vectorial_product(POSITIONS, OMEGA)

# ------------------------------------
# TEST, VAL, DEPLOYMENT
# ------------------------------------
POS_TOL = 1e-5
ROT_TOL = 1e-5
RND_SEED = 0
N_RANDOMS = 200
LOOP_HZ=50 # Use between 50-100, bus can't handle much higher
MOVE_TIME=3.0
SETTLE_TIME=1.0
LIMIT_MARGIN=0.05
PLOT_PERIOD=0.15

MATH_POSES = {
    "zero": [0, 0, 0, 0, 0],
    "pan_plus": [1.0, 0, 0, 0, 0],
    "pan_minus": [-1.0, 0, 0, 0, 0],
    "lift_forward": [0, 0.8, 0, 0, 0],
    "elbow_up": [0, 0, -1.0, 0, 0],
    "wrist_down": [0, 0, 0, 1.2, 0],
    "roll": [0, 0, 0, 0, 1.5],
    "folded": [0, -1.5, -1.5, 1.0, 0],
    "mixed": [0.5, 0.4, -0.6, 0.9, -1.0],
}

POSES = {
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0],
    "elbow_up": [0.0, 0.0, -0.6, 0.3, 0.0],
    "high_reach": [0.0, -0.5, -0.4, 0.4, 0.0],
    "reach_forward": [0.0, 0.4, -0.3, 0.6, 0.0],
    "wrist_down": [0.0, 0.0, 0.0, 1.2, 0.0],
    "compact": [0.0, -0.6, 0.9, 0.5, 0.0],
    "side_mixed": [0.4, 0.3, -0.3, 0.8, 0.5],
}
