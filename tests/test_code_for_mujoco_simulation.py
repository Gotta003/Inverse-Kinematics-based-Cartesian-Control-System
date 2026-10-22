import mujoco
import mujoco.viewer

from so101_mujoco_utils import (
    set_initial_pose,
    move_sim,
    hold_sim
)

# Load SO101 MuJoCo model

m = mujoco.MjModel.from_xml_path('model_files/scene.xml')
d = mujoco.MjData(m)

# Starting position

starting_position = {
    'shoulder_pan': -0.0192,
    'shoulder_lift': -1.7453,
    'elbow_flex': 1.5500,
    'wrist_flex': 1.1599,
    'wrist_roll': 0.0766,
    'gripper': 0.0
}

# Desired position

desired_position = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0.0
}

# Put robot directly at starting position

set_initial_pose(d, starting_position)

# Run simulation

with mujoco.viewer.launch_passive(m, d) as viewer:

    # 1. Move from starting to extreme position
    move_sim(
        m,
        d,
        viewer,
        desired_position,
        duration=3.0
    )

    # 2. Hold the extreme pose for 2 seconds
    hold_sim(
        m,
        d,
        viewer,
        duration=2.0
    )

    # 3. Return to starting pose
    move_sim(
        m,
        d,
        viewer,
        starting_position,
        duration=3.0
    )