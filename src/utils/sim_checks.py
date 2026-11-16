import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import mujoco
from src.forward_kinematics import SO_101_fk
from src.utils.constants import N_JOINTS

def contacts(m, d, q, gripper=0.0):
    d.qpos[:]=0.0
    d.qpos[:N_JOINTS]=q
    if m.nq>N_JOINTS:
        d.qpos[N_JOINTS]=gripper
    mujoco.mj_forward(m, d)
    out=set()
    for c in d.contact[:d.ncon]:
        n1=m.body(m.geom_bodyid[c.geom1]).name
        n2=m.body(m.geom_bodyid[c.geom2]).name
        if m.geom(c.geom1).name=="floor" or m.geom(c.geom2).name=="floor":
            out.add(("floor", n2 if m.geom(c.geom1).name=="floor" else n1, "floor"))
        else:
            out.add(("self", *sorted((n1, n2))))
    return out