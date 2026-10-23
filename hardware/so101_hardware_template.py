import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.so101_utils import hold, load_calibration, move, setup_motors

# ========== INITIALIZATION ===========
PORT_ID = "/dev/ttyACM0"  # USB port you are using
ROBOT_NAME = "jarvis"  # ID you assigned for the robot

calibration = load_calibration(ROBOT_NAME)
bus = setup_motors(calibration, PORT_ID)

# ============= MIDPOINTS =============
P1 = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 30,
}

# ============ TRAJECTORY =============
move(bus, P1, 2.0)
hold(3.0)
bus.disable_torque()