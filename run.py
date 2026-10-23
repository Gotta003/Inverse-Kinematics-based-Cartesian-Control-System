"""
Menu
1) FK tests (pytest)
2) FK validation vs MuJoCo  simulation/fk_validation.py
3) FK check on real arm     hardware/fk_hardware.py
4) Simulation vs real arm   hardware/sim_vs_real.py
5) Check robot connection   bus type + joint readings in radians
6) Detect the port again
7) Change robot (calibration file)
q) Quit
"""
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent
CONFIG_FILE=REPO_ROOT/".robot_config.json"
CALIB_DIR=REPO_ROOT/"calibration_files"
PYTHON=sys.executable

CYAN="\033[0;36m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BOLD="\033[1m"
RESET="\033[0m"

def info(msg):
    print(f"{CYAN}[INFO]{RESET} {msg}")
    
def ok(msg):
    print(f"{GREEN}[OK]{RESET} {msg}")
    
def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")
    
def err(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")

def load_config():
    if CONFIG_FILE.is_file():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except json.JSONDecodeError:
            warn(f"{CONFIG_FILE.name} is corrupted, starting fresh.")
    return {}

def menu(title, options):
    print(f"\n{BOLD}{title}{RESET}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}   {opt}")
    raw=input("> ").strip()
    if raw.isdigit() and 1<=int(raw)<=len(options):
        return int(raw)-1
    return None

def run(cmd):
    print(f"\n{CYAN}$ {' '.join(cmd)}{RESET}\n")
    try:
        return subprocess.call(cmd, cwd=REPO_ROOT)
    except KeyboardInterrupt:
        return 130

def ask(prompt, default):
    raw=input(f"{prompt} [{default}]: ").strip()
    return raw or str(default)

def ask_yes(prompt, default=False):
    d="Y/n" if default else "y/N"
    raw=input(f"{prompt} [{d}]: ").strip().lower()
    return default if not raw else raw.startswith("y")

def list_ports():
    try:
        from serial.tools import list_ports as lp
        return sorted(p.device for p in lp.comports())
    except ImportError:
        pass
    system=platform.system()
    if system=="Linux":
        patterns=["/dev/ttyACM*", "/dev/ttyUSB*"]
    elif system=="Darwin":
        patterns=["/dev/tty.usbmodem*", "/dev/tty.usbserial*", "/dev/cu.usbmodem*"]
    else:
        patterns=[]
    return sorted(p for pat in patterns for p in glob.glob(pat))

def port_is_valid(port):
    if not port or port not in list_ports():
        return False
    try:
        import serial    
        with serial.Serial(port, baudrate=1000000, timeout=0.1):
            pass
    except ImportError:
        return os.access(port, os.R_OK | os.W_OK) if os.name=="posix" else True
    except Exception as e:
        warn(f"{port} is present but cannot be opened: {e}")
        if "ermission" in str(e) and platform.system()=="Linux":
            warn("Fix once with: sudo usermod -aG dialout $USER (then log out and in)")
        return False
    return True

def detect_port():
    print(f"\n{BOLD}Finding the robot's serial port{RESET}")
    input("Make sure the arm's USB cable is CONNECTED, then press Enter...")
    before=set(list_ports())
    if not before:
        err("No serial ports found. Is the cable connected and the driver installed?")
        return None
    info(f"Ports now: {sorted(before)}")
    input("UNPLUG the arm's USB cable, then press Enter...")
    time.sleep(0.5)
    after=set(list_ports())
    gone=sorted(before-after)
    if len(gone)!=1:
        err(f"Expected exactly one port to disappear, got {gone or 'none'}.")
        return None
    port=gone[0]
    ok(f"The robot is on {port}")
    input("PLUG the cable back in, then press Enter...")
    for _ in range(20):
        if port in list_ports():
            break
        time.sleep(0.5)
    else:
        warn(f"{port} has not reappeared yet; it may get a different name")
        return None
    return port

def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

def get_port(cfg, force_detect=False):
    port=cfg.get("port")
    if not force_detect and port_is_valid(port):
        return port
    if port and not force_detect:
        warn(f"Saved port {port} is not valid any more.")
    while True:
        port=detect_port()
        if port and port_is_valid(port):
            cfg["port"]=port
            save_config(cfg)
            ok(f"Saved port {port} in {CONFIG_FILE.name}")
            return port
        if input("Try the detection again? [Y/n]: ").strip().lower()=="n":
            return None

def choose_robot(cfg):
    names=sorted(p.stem for p in CALIB_DIR.glob("*.json"))
    if not names:
        err(f"No calibration files in {CALIB_DIR}")
        return cfg.get("robot")
    if len(names)==1:
        cfg["robot"]=names[0]
    else:
        idx=menu("Choose the robot (calibration file)", names)
        if idx is not None:
            cfg["robot"]=names[idx]
    save_config(cfg)
    ok(f"Robot: {cfg['robot']}")
    return cfg["robot"]

def run_hardware(cfg, args):
    port=get_port(cfg)
    if not port:
        warn("No valid port: hardware script not started.")
        return
    robot=cfg.get("robot") or choose_robot(cfg)
    code=run([PYTHON]+args+["--port", port, "--robot", robot])
    if code not in (0, 130):
        err(f"Script exited with code {code}")
        if ask_yes("Was it a connection problem? Detect the port again", default=False):
            get_port(cfg, force_detect=True)

def pose_names():
    import re
    try:
        out=subprocess.run([PYTHON, "hardware/sim_vs_real.py", "-h"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60).stdout
        match=re.search(r"--poses \{([^}]*)\}", out)
        return match.group(1).split(",") if match else []
    except Exception:
        return []

CHECK_SNIPPET="""
import numpy as np
from utils.so101_utils import setup_motors, load_calibration
b = setup_motors(load_calibration('{robot}'), '{port}')
try:
    q = b.sync_read('Present_Position')
    print('Bus type :', type(b).__name__)
    for j, v in q.items():
        print(f'  {{j:<14}} {{v: .4f}}')
    arm = [v for j, v in q.items() if j != 'gripper']
    if any(abs(v) > 2 * np.pi for v in arm):
        print('NOT RADIANS: the bus is in degrees without conversion (check utils/so101_utils.py)')
    else:
        print('Readings look like radians.')
finally:
    b.disconnect()
"""

def check_connection(cfg):
    port=get_port(cfg)
    if not port:
        return
    robot=cfg.get("robot") or choose_robot(cfg)
    run([PYTHON, "-c", CHECK_SNIPPET.format(robot=robot, port=port)])

def main():
    os.chdir(REPO_ROOT)
    cfg=load_config()
    print(f"{BOLD}SO-101 project launcher{RESET}    (repo: {REPO_ROOT.name})")
    if cfg.get("port"):
        info(f"Saved port: {cfg['port']} robot: {cfg.get('robot', '-')}")
        
    items=[
        "FK tests (pytest)",
        "FK validation vs MuJoCo",
        "FK check on the real arm",
        "Simulation vs real arm",
        "Check robot connection",
        "Detect the port again",
        "Change robot (calibration file)",
        "Quit",
    ]
    while True:
        idx=menu("Choose what to run!", items)
        if idx is None:
            warn("Type the number of an option.")
            continue
        choice=items[idx]
        if choice=="Quit":
            break
        elif choice=="FK tests (pytest)":
            run([PYTHON, "-m", "pytest", "tests", "-v"])
        elif choice=="FK validation vs MuJoCo":
            n=ask("Number of random configurations", 2000)
            seed=ask("Random seed", 0)
            args=["simulation/fk_validation.py", "--n_random", n, "--seed", seed]
            if ask_yes("Stop through the poses in the MuJoCo viewer"):
                args.append("--view")
            run([PYTHON]+args)
        elif choice=="FK check on the real arm":
            mode=menu("Mode", [
                "commanded: the arm moves to the safe poses (torque ON)",
                "passive: move the arm by hand (torque OFF, HOLD THE ARM)"
            ])
            if mode is None:
                continue
            args=["hardware/fk_hardware.py"]+(["--passive"] if mode==1 else [])
            run_hardware(cfg, args)
        elif choice=="Simulation vs real arm":
            names=pose_names()
            if names:
                info(f"Available poses: {' '.join(names)}")
            raw=input("Poses to visit, space separated (Enter=default sequence): ").split()
            unknown=[p for p in raw if names and p not in names]
            if unknown:
                err(f"Unknown pose(s): {unknown}")
                continue
            args=["hardware/sim_vs_real.py"]
            if raw:
                args+=["--poses"]+raw
            if not ask_yes("Show the MuJoCo 3D viewer", default=True):
                args.append("--no-viewer")
            run_hardware(cfg, args)
        elif choice=="Check robot connection":
            check_connection(cfg)
        elif choice=="Detect the port again":
            get_port(cfg, force_detect=True)
        elif choice=="Change robot (calibration file)":
            choose_robot(cfg)

    print("Quitting...")
if __name__=="__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nClose Menu")