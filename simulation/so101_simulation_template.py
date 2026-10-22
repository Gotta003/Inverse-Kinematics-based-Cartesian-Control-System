import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer

from utils.so101_mujoco_utils import hold_sim, move_sim, set_initial_pose

# ========== INITIALIZATION ===========
m = mujoco.MjModel.from_xml_path("model_files/scene.xml")
d = mujoco.MjData(m)
starting_position = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 30,
}
set_initial_pose(d, starting_position)
mujoco.mj_forward(m, d)

# ============= MIDPOINTS =============
P1 = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.5,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 30,
}

# ============ TRAJECTORY =============
with mujoco.viewer.launch_passive(m, d) as viewer:
    move_sim(m, d, viewer, P1, 2.0)
    hold_sim(m, d, viewer, 3.0)
