"""
Tests for src/forward_kinematics.py

Run:
    pytest tests/test_fk.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pytest

from src.forward_kinematics import SO_101_fk
from src.utils.constants import N_JOINTS, N_RANDOMS, POS_TOL, POSES, RND_SEED, ROT_TOL
from src.utils.helper import derive_limits, position_error, rotation_error
from utils.so101_mujoco_utils import mujoco_starting, mujoco_tcp

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = REPO_ROOT / "model_files" / "scene.xml"


@pytest.fixture(scope="module")
def mj():
    m, d, site_id=mujoco_starting(str(SCENE_XML))
    limits=derive_limits(m, N_JOINTS)
    return m, d, site_id, limits

@pytest.mark.parametrize("name", POSES)
def test_named_poses_match_mujoco(mj, name):
    q = np.array(POSES[name], dtype=float)
    T_ours = SO_101_fk(q)
    m, d, site_id, _= mj
    T_ref = mujoco_tcp(m, d, site_id, q)
    assert position_error(T_ours, T_ref) < POS_TOL
    assert rotation_error(T_ours[:3, :3], T_ref[:3, :3]) < ROT_TOL


def random_configs(limits, n=N_RANDOMS, seed=RND_SEED):
    rng = np.random.default_rng(seed)
    bounds=np.array(list(limits.values()))
    return rng.uniform(bounds[:,0], bounds[:,1], size=(n, N_JOINTS))


def test_rand_configs_match_mujoco(mj):
    pos_err, rot_err = [], []
    m,d,site_id,limits=mj
    for q in random_configs(limits):
        T_ours = SO_101_fk(q)
        T_ref = mujoco_tcp(m, d, site_id, q)
        pos_err.append(position_error(T_ours, T_ref))
        rot_err.append(rotation_error(T_ours[:3, :3], T_ref[:3, :3]))
    assert max(pos_err) < POS_TOL, f"max pos error {max(pos_err):.2e}"
    assert max(rot_err) < ROT_TOL, f"max error {max(rot_err):.2e} rad"
