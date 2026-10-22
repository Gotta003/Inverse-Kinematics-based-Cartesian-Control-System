import time

import mujoco
import numpy as np

N_JOINTS = 5


def convert_to_dictionary(qpos):
    """
    Converts a list to a joint angle dictionary.

    NOTE. If you are using degrees for computation / in hardware
        you need to make a rad-to-deg conversion.

    Args:
        qpos: List of joint angles in rad.

    Returns:
        A joint angle dictionary. (See SO-101 Robot Arm Setup Section 5.3)
    """
    return {
        "shoulder_pan": qpos[0],
        "shoulder_lift": qpos[1],
        "elbow_flex": qpos[2],
        "wrist_flex": qpos[3],
        "wrist_roll": qpos[4],
        "gripper": qpos[5] * 100.0 / 3.14159,  # Converts to 0-100 range
    }


def convert_to_list(dictionary):
    """
    Converts a joint angle dictionary to a list for MuJoCo use.

    NOTE. If you are using degrees for computation / in hardware
        you need to make a rad-to-deg conversion.

    Args:
        dictionary: Joint angle dictionary

    Returns:
        Joint angle list
    """
    return [
        dictionary["shoulder_pan"],
        dictionary["shoulder_lift"],
        dictionary["elbow_flex"],
        dictionary["wrist_flex"],
        dictionary["wrist_roll"],
        dictionary["gripper"] * 3.14159 / 100.0,  # Converts to radians for MuJoCo
    ]


def set_initial_pose(d, position_dict):
    """
    Sets the simulation's joint configuration directly, bypassing dynamics.

    Writes position_dict straight into d.qpos, teleporting the model to
    that configuration for the next step/render — this does not command
    an actuator (unlike send_position_command) and does not respect
    joint dynamics. Intended for initializing a scene before a
    simulation loop starts, not for use during control.

    Args:
        d: MuJoCo MjData object for the model instance.
        position_dict: Joint angle dictionary (See SO-101 Robot Arm
            Setup Section 5.3).

    Returns:
        None.
    """
    pos = convert_to_list(position_dict)
    d.qpos[:] = pos


def send_position_command(d, position_dict):
    """
    Sends a joint-space position command to the simulated actuators.

    Writes position_dict into d.ctrl, the actuator control input, so the
    change takes effect through the model's actuator/dynamics on the
    next mujoco.mj_step call (unlike set_initial_pose, which sets state
    directly).

    Args:
        d: MuJoCo MjData object for the model instance.
        position_dict: Goal joint configuration as a joint angle
            dictionary (See SO-101 Robot Arm Setup Section 5.3).

    Returns:
        None.
    """
    pos = convert_to_list(position_dict)
    d.ctrl[:] = pos


def move_sim(m, d, viewer, desired_position, duration):
    """
    Trajectory generator for joint-space straight line path with linear
    time scaling, MuJoCo simulation counterpart to move().

    NOTE. Moving time not exactly duration seconds since this doesn't
        take in to account the time spent in computing the midpoints
        and stepping the physics. This might have a significant effect
        on computationally heavier trajectory generators.

    Args:
        m: MuJoCo MjModel object for the model instance.
        d: MuJoCo MjData object for the model instance.
        viewer: Active MuJoCo viewer to sync after each physics step.
        desired_position: Goal joint configuration as a joint angle
            dictionary (See SO-101 Robot Arm Setup Section 5.3).
        duration: Duration of the movement in seconds. Must be >= 0.

    Returns:
        None.
    """
    if duration <= 0:
        send_position_command(d, desired_position)
        mujoco.mj_step(m, d)
        viewer.sync()
        return

    start_time = time.time()
    starting_pose = d.qpos.copy()
    starting_pose = convert_to_dictionary(starting_pose)

    while viewer.is_running():
        step_start = time.time()
        t = time.time() - start_time
        if t > duration:
            break

        # Linear time scaling, s ∈ [0,1], must make sure s doesn't exceed 1
        s = min(t / duration, 1)

        # Interpolation step
        position_dict = {}
        for joint in desired_position:
            p0 = starting_pose[joint]
            pf = desired_position[joint]
            position_dict[joint] = (1 - s) * p0 + s * pf

        # Send command
        send_position_command(d, position_dict)
        mujoco.mj_step(m, d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Real-time pacing, matches simulation timestep to wall-clock time
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def hold_sim(m, d, viewer, duration):
    """
    Holds the simulated arm in place for duration seconds, MuJoCo
    simulation counterpart to hold().

    Unlike hold(), which relies on the physical motors' own
    position-control loop to maintain the last commanded position,
    hold_sim() actively re-sends the current joint configuration as the
    control input on every step. This is because a MuJoCo actuator
    won't hold position on its own once mj_step is called with no
    control input refreshed.

    Args:
        m: MuJoCo MjModel object for the model instance.
        d: MuJoCo MjData object for the model instance.
        viewer: Active MuJoCo viewer to sync after each physics step.
        duration: Time to hold in seconds. Must be >= 0.

    Returns:
        None.
    """
    current_pos = d.qpos.copy()
    current_pos_dict = convert_to_dictionary(current_pos)

    start_time = time.time()
    while viewer.is_running():
        step_start = time.time()
        t = time.time() - start_time
        if t > duration:
            break
        send_position_command(d, current_pos_dict)
        mujoco.mj_step(m, d)
        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def mujoco_starting(path_xml):
    m = mujoco.MjModel.from_xml_path(path_xml)
    d = mujoco.MjData(m)
    site_id = m.site("gripperframe").id
    return m, d, site_id


def mujoco_tcp(m, d, site_id, q, gripper=0.0):
    d.qpos[:N_JOINTS] = q
    d.qpos[N_JOINTS] = gripper
    mujoco.mj_forward(m, d)
    T = np.eye(4)
    T[:3, :3] = d.site_xmat[site_id].reshape(3, 3)
    T[:3, 3] = d.site_xpos[site_id]
    return T
