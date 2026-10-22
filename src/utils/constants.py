import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from src.utils.helper import deg_to_rad, vectorial_product

# ------------------------------------
# FORWARD KINEMATICS
# ------------------------------------

theta_cal = 2.789
theta_cal = deg_to_rad(theta_cal)

JOINT_NAMES={
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
}
N_JOINTS=len(JOINT_NAMES)

M = np.array(
    [
        [1, 0, 0, 391.3619],
        [0, np.cos(theta_cal), -np.sin(theta_cal), -0.0113],
        [0, np.sin(theta_cal), np.cos(theta_cal), 226.4687],
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
        [38.8353, 0, 0],
        [69.2345, -18.2778, 116.6],
        [97.2345, -18.2778, 229.1700],
        [232.1345, -0.1778, 234.37],
        [293.2345, -0.1778, 234.37],
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

POSES = {
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
