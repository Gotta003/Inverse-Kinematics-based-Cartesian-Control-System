import math
import time
from pathlib import Path

import draccus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from src.utils.constants import LOOP_HZ


def load_calibration(ROBOT_NAME):
    """
    Loads calibration data from calibration_files/<ROBOT_NAME>.json.

    Args:
        ROBOT_NAME: Name of the robot, used to locate the calibration file
            at calibration_files/{ROBOT_NAME}.json.

    Returns:
        dict[str, MotorCalibration]: Mapping from motor name (e.g.
        "shoulder_pan") to its MotorCalibration.

    Raises:
        FileNotFoundError: If no calibration file exists for ROBOT_NAME.
    """
    fpath = Path(f"calibration_files/{ROBOT_NAME}.json")
    with open(fpath) as f, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], f)
        return calibration

_POSITION_REGISTERS=("Present_Position", "Goal_Position")
_ARM_JOINTS=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

class RadianBus:
    def __init__(self, bus):
        self._bus=bus
    
    def __getattr__(self, name):
        return getattr(self._bus, name)
    
    @staticmethod
    def _convert(data_name, values, factor, normalize):
        if not normalize or data_name not in _POSITION_REGISTERS or not isinstance(values, dict):
            return values
        return {
            j: (v * factor if j in _ARM_JOINTS else v) for j,v in values.items()
        }
        
    def sync_read(self, data_name, motors=None, *args, normalize=True, **kwargs):
        values=self._bus.sync_read(data_name, motors, *args, normalize=normalize, **kwargs)
        return self._convert(data_name, values, math.pi/180.0, normalize)
    
    def sync_write(self, data_name, values, *args, normalize=True, **kwargs):
        values=self._convert(data_name, values, 180.0/math.pi, normalize)
        return self._bus.sync_write(data_name, values, *args, normalize=normalize, **kwargs)
    
    def read(self, data_name, motor, *args, normalize=True, **kwargs):
        value=self._bus.read(data_name, motor, *args, normalize=normalize, **kwargs)
        if normalize and data_name in _POSITION_REGISTERS and motor in _ARM_JOINTS:
            value=value*math.pi/180.0
        return value
    
    def write(self, data_name, motor, value, *args, normalize=True, **kwargs):
        if normalize and data_name in _POSITION_REGISTERS and motor in _ARM_JOINTS:
            value=value*180.0/math.pi
        return self._bus.write(data_name, motor, value, *args, normalize=normalize, **kwargs)

def setup_motors(calibration, PORT_ID):
    """
    Initializes and configures the SO-101 arm's Feetech STS3215 motor bus.

    Args:
        calibration: dict[str, MotorCalibration] as returned by
            load_calibration(), applied to the bus so motor readings/
            commands are mapped through each joint's homing offset and
            limits.
        PORT_ID: Serial port identifier for the arm's USB adapter
            (e.g. "/dev/ttyACM0" or "COM3").

    Returns:
        FeetechMotorsBus: Connected and configured motor bus, ready for
        read/write calls. Torque is re-enabled on exiting the
        torque_disabled() context.
    """
    has_radians=hasattr(MotorNormMode, "RADIANS")
    norm_mode_body=MotorNormMode.RADIANS if has_radians else MotorNormMode.DEGREES
    bus = FeetechMotorsBus(
        port=PORT_ID,
        motors={
            "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=calibration,
    )
    bus.connect(True)

    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", motor, 32)  # Default 32
            bus.write("I_Coefficient", motor, 0)  # Default 0
            bus.write("D_Coefficient", motor, 32)  # Default 32
    return bus if has_radians else RadianBus(bus)


def move(bus, desired_position, duration):
    """
    Trajectory generator for joint-space straight line path with linear time scaling.

    NOTE. Moving time not exactly duration seconds since this doesn't take in to account
        the time spent in computing the midpoints. This might have a significant effect on
        computationally heavier trajectory generators.

    Args:
        bus: Connected motor bus.
        desired_position: Goal joint configuration as a joint angle dictionary
            (See SO-101 Robot Arm Setup Section 5.3).
        duration: Duration of the movement in seconds. Must be >= 0.

    Returns:
        None.
    """
    f_loop = LOOP_HZ  # Interpolation loop frequency in Hz.
    # Use between 50-100, bus can't handle much higher

    if duration <= 0:
        bus.sync_write("Goal_Position", desired_position, normalize=True)
        return

    start_time = time.time()
    starting_pose = bus.sync_read("Present_Position")

    while True:
        t = time.time() - start_time
        if t > duration:
            break

        # Linear tine scaling, s ∈ [0,1], must make sure s doesn't exceed 1
        s = min(t / duration, 1)

        # Interpolation step
        position_dict = {}
        for joint in desired_position:
            p0 = starting_pose[joint]
            pf = desired_position[joint]
            position_dict[joint] = (1 - s) * p0 + s * pf
        # Send Command
        bus.sync_write("Goal_Position", position_dict, normalize=True)
        time.sleep(1 / f_loop)
    # Send Command
    bus.sync_write("Goal_Position", desired_position, normalize=True)


def hold(duration):
    """
    Holds the arm in place for duration seconds.

    Args:
        duration: Time to hold in seconds. Must be >= 0.

    Returns:
        None.
    """
    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t > duration:
            break
        time.sleep(0.02)  # 50 Hz loop
