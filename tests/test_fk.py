"""
Tests for src/forward_kinematics.py

Run:
    pytest tests/test_fk.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
import numpy as np
from src.forward_kinematics import SO_101_fk
from src.utils.helper import derive_limits
from src.utils.constants import N_JOINTS,  POSES, RND_SEED, N_RANDOMS, POS_TOL, ROT_TOL
import mujoco
import mujoco.viewer
 #from utils.so101_mujoco_utils import mujoco_tcp

REPO_ROOT=Path(__file__).resolve().parent.parent
SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"

# Compare against Mujoco
def position_error(A, B):
    return float(np.linalg.norm(A[:3, 3]-B[:3, 3]))

def rotation_error(A, B):
    cos_angle=(np.trace(A.T@B)-1.0)/2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def mujoco_tcp(mj, q, gripper=0.0):
    m, d, site_id = mj
    joint_limits={
        m.joint(i).name: m.jnt_range[i].tolist() for i in range(m.njnt)
    }
    assert f"All Joint Limits: {joint_limits}"
    d.qpos[:N_JOINTS] = q
    d.qpos[N_JOINTS] = gripper
    mujoco.mj_forward(m, d)
    T = np.eye(4)
    T[:3, :3] = d.site_xmat[site_id].reshape(3, 3)
    T[:3, 3] = d.site_xpos[site_id]
    return T

@pytest.fixture(scope="module")
def mj():
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    site_id = m.site("gripperframe").id
    global JOINT_LIMITS
    JOINT_LIMITS=derive_limits(m, N_JOINTS)
    return m, d, site_id

@pytest.mark.parametrize("name", POSES)
def test_named_poses_match_mujoco(mj, name):
    q=np.array(POSES[name], dtype=float)
    T_ours=SO_101_fk(q)
    T_ours[:3, 3]/= 1000.0
    T_ref=mujoco_tcp(mj, q)
    assert position_error(T_ours, T_ref)<POS_TOL
    assert rotation_error(T_ours[:3,:3], T_ref[:3,:3])<ROT_TOL
    
def random_configs(n=N_RANDOMS, seed=RND_SEED):
    rng=np.random.default_rng(seed)
    limits_array = np.array(list(JOINT_LIMITS.values()))  # Shape: (N_JOINTS, 2)
    lowers = limits_array[:, 0]
    uppers = limits_array[:, 1]
    return rng.uniform(lowers, uppers, size=(n, N_JOINTS))
    
def test_rand_configs_match_mujoco(mj):
    pos_err, rot_err=[], []
    for q in random_configs():
        T_ours=SO_101_fk(q)
        T_ref=mujoco_tcp(mj, q)
        T_ours[:3,3]/=1000
        pos_err.append(position_error(T_ours, T_ref))
        rot_err.append(rotation_error(T_ours, T_ref))
    assert max(pos_err)<POS_TOL, f"max pos error {max(pos_err):.2e}"
    assert max(rot_err)<ROT_TOL, f"max error {max(rot_err):.2e} rad"