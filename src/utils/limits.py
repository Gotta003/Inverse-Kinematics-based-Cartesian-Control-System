import sys
from pathlib import Path
import json
import xml.etree.ElementTree as ET

REPO_ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.utils.constants import JOINT_NAMES, LIMIT_MARGIN

MODEL_XML=REPO_ROOT/"model_files"/"so101_new_calib.xml"
CALIB_DIR=REPO_ROOT/"calibration_files"
TICKS_PER_REVOLUTION=4096

def model_limits(xml_path=MODEL_XML):
    ranges={}
    for j in ET.parse(xml_path).getroot().iter("joint"):
        if j.get("name") in JOINT_NAMES and j.get("range"):
            ranges[j.get("name")]=[float(v) for v in j.get("range").split()]
    return np.array([ranges[n] for n in JOINT_NAMES])

def calibration_limits(robot="jarvis", calib_dir=CALIB_DIR):
    calib=json.loads((Path(calib_dir)/f"{robot}.json").read_text())
    rows=[]
    for n in JOINT_NAMES:
        half=(calib[n]["range_max"]-calib[n]["range_min"])/2*2*np.pi/TICKS_PER_REVOLUTION
        rows.append([-half, half])
    return np.array(rows)

def inv_kin_limits(robot=None, margin=LIMIT_MARGIN):
    lim=model_limits()
    if robot is not None:
        cal=calibration_limits(robot)
        max_lower=np.maximum(lim[:,0], cal[:,0])
        min_higher=np.minimum(lim[:,1], cal[:,1])
        lim=np.column_stack((max_lower, min_higher))
    lim+=np.array([margin, -margin])
    if np.any(lim[:,0]>=lim[:,1]):
        raise ValueError("Empty joint range after margin")
    return lim

if __name__=="__main__":
    model=model_limits()
    cal=calibration_limits("jarvis")
    inv_kin=inv_kin_limits("jarvis")
    for n, a, b, c in zip(JOINT_NAMES, np.degrees(model), np.degrees(cal), np.degrees(inv_kin)):
        print(f"{n:14s} [{a[0]:7.1f},{a[1]:7.1f}] [{b[0]:7.1f},{b[1]:7.1f}] [{c[0]:7.1f},{c[1]:7.1f}]")