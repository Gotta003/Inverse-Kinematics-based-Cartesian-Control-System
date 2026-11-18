"""
Tests for src/jacobian.py

    1. Structure: J_s(0) equals the screw axes [omega; v]
    2. Consistency with FK: finite differences of SO_101_fk match J_s and J_TCP (vee check)
    3. MuJoCo: J_TCP matches mj_jacSite on the gripper frame after mj_forward

Run:
    pytest tests/test_jacobian.py -v
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
import pytest

from src.forward_kinematics import SO_101_fk
from src.jacobian import (
    SCREW_AXES,
    SO_101_space_jacobian,
    jacobian_metrics,
    position_jacobian,
    tcp_jacobian,
)
from src.utils.constants import N_JOINTS, N_RANDOMS, OMEGA, POSES, RND_SEED, V
from src.utils.helper import derive_limits, so3_log, vee_so3
from utils.so101_mujoco_utils import mujoco_starting, mujoco_tcp

SCENE_XML = REPO_ROOT / "model_files" / "scene.xml"

FD_STEP = 1e-6   # rad, central-difference step
FD_TOL = 1e-6    # FD truncation error is O(h^2), round-off O(eps/h)
MJ_TOL = 1e-5    # same order as POS_TOL/ROT_TOL used for FK vs MuJoCo


# ------------------------------------
# FIXTURES / HELPERS
# ------------------------------------
@pytest.fixture(scope="module")
def mj():
    m, d, site_id = mujoco_starting(str(SCENE_XML))
    limits = derive_limits(m, N_JOINTS)
    return m, d, site_id, limits


def random_configs(limits, n=N_RANDOMS, seed=RND_SEED):
    rng = np.random.default_rng(seed)
    bounds = np.array(list(limits.values()))
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, N_JOINTS))


def fd_space_jacobian(q, h=FD_STEP):
    """Column i = vee( dT/dq_i T^-1 ) via central differences (spatial twist [omega; v])"""
    J = np.zeros((6, N_JOINTS))
    T_inv = np.linalg.inv(SO_101_fk(q))
    for i in range(N_JOINTS):
        dq = np.zeros(N_JOINTS)
        dq[i] = h
        dT = (SO_101_fk(q + dq) - SO_101_fk(q - dq)) / (2 * h)
        X = dT @ T_inv                       # element of se(3)
        J[:3, i] = vee_so3(X[:3, :3])
        J[3:, i] = X[:3, 3]
    return J


def fd_tcp_jacobian(q, h=FD_STEP):
    """Column i = [ log(R+ R-^T)/2h ; (p+ - p-)/2h ] (angular and linear TCP velocity in base frame)"""
    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        dq = np.zeros(N_JOINTS)
        dq[i] = h
        T_p, T_m = SO_101_fk(q + dq), SO_101_fk(q - dq)
        J[:3, i] = so3_log(T_p[:3, :3] @ T_m[:3, :3].T) / (2 * h)
        J[3:, i] = (T_p[:3, 3] - T_m[:3, 3]) / (2 * h)
    return J


def mujoco_tcp_jacobian(m, d, site_id, q):
    """[jacr; jacp] of the gripper site restricted to the arm dofs, after mj_forward"""
    mujoco_tcp(m, d, site_id, q)             # sets qpos and runs mj_forward
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
    dofs = [m.jnt_dofadr[m.joint(i).id] for i in range(N_JOINTS)]
    return np.vstack((jacr[:, dofs], jacp[:, dofs]))


# ------------------------------------
# 1. STRUCTURE
# ------------------------------------
def test_space_jacobian_at_zero_equals_screw_axes():
    J = SO_101_space_jacobian(np.zeros(N_JOINTS))
    assert J.shape == (6, N_JOINTS)
    np.testing.assert_allclose(J, SCREW_AXES.T, atol=1e-12)
    np.testing.assert_allclose(J[:3], OMEGA.T, atol=1e-12)
    np.testing.assert_allclose(J[3:], V.T, atol=1e-12)


@pytest.mark.parametrize("name", POSES)
def test_first_column_is_constant(name):
    """J_s,1 = S_1 for every q (nothing precedes joint 1)"""
    J = SO_101_space_jacobian(POSES[name])
    np.testing.assert_allclose(J[:, 0], SCREW_AXES[0], atol=1e-12)


@pytest.mark.parametrize("name", POSES)
def test_angular_columns_are_unit_and_shapes(name):
    q = np.array(POSES[name], dtype=float)
    J_s, J_tcp, J_p = SO_101_space_jacobian(q), tcp_jacobian(q), position_jacobian(q)
    assert J_tcp.shape == (6, N_JOINTS) and J_p.shape == (3, N_JOINTS)
    # revolute joints: unit rotation axes, identical in J_s and J_TCP
    np.testing.assert_allclose(np.linalg.norm(J_s[:3], axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(J_tcp[:3], J_s[:3], atol=1e-12)
    np.testing.assert_allclose(J_p, J_tcp[3:], atol=1e-12)


def test_explicit_tcp_point_matches_default():
    q = np.array(POSES["side_mixed"])
    p = SO_101_fk(q)[:3, 3]
    np.testing.assert_allclose(tcp_jacobian(q, p), tcp_jacobian(q), atol=1e-12)


@pytest.mark.parametrize("bad_q", [np.zeros(N_JOINTS - 1), np.zeros(N_JOINTS + 1)])
def test_wrong_dimension_raises(bad_q):
    with pytest.raises(ValueError):
        SO_101_space_jacobian(bad_q)
    with pytest.raises(ValueError):
        tcp_jacobian(bad_q)


def test_metrics_consistency():
    J = position_jacobian(POSES["reach_forward"])
    met = jacobian_metrics(J)
    s = np.linalg.svd(J, compute_uv=False)
    np.testing.assert_allclose(met["sigma"], s)
    assert met["sigma_min"] == pytest.approx(s[-1])
    assert met["manipulability"] == pytest.approx(np.sqrt(np.linalg.det(J @ J.T)))
    assert met["cond"] == pytest.approx(s[0] / s[-1])
    assert met["cond"] >= 1.0


def test_metrics_singular_matrix():
    met = jacobian_metrics(np.zeros((3, N_JOINTS)))
    assert met["sigma_min"] == 0.0
    assert met["manipulability"] == 0.0
    assert met["cond"] == np.inf


# ------------------------------------
# 2. CONSISTENCY WITH FK (finite differences)
# ------------------------------------
@pytest.mark.parametrize("name", POSES)
def test_space_jacobian_matches_fk_named(name):
    q = np.array(POSES[name], dtype=float)
    np.testing.assert_allclose(SO_101_space_jacobian(q), fd_space_jacobian(q), atol=FD_TOL)


@pytest.mark.parametrize("name", POSES)
def test_tcp_jacobian_matches_fk_named(name):
    q = np.array(POSES[name], dtype=float)
    np.testing.assert_allclose(tcp_jacobian(q), fd_tcp_jacobian(q), atol=FD_TOL)


def test_jacobians_match_fk_random(mj):
    *_, limits = mj
    err_s, err_tcp = [], []
    for q in random_configs(limits):
        err_s.append(np.max(np.abs(SO_101_space_jacobian(q) - fd_space_jacobian(q))))
        err_tcp.append(np.max(np.abs(tcp_jacobian(q) - fd_tcp_jacobian(q))))
    assert max(err_s) < FD_TOL, f"max J_s FD error {max(err_s):.2e}"
    assert max(err_tcp) < FD_TOL, f"max J_TCP FD error {max(err_tcp):.2e}"


# ------------------------------------
# 3. MUJOCO mj_jacSite
# ------------------------------------
@pytest.mark.parametrize("name", POSES)
def test_tcp_jacobian_matches_mujoco_named(mj, name):
    m, d, site_id, _ = mj
    q = np.array(POSES[name], dtype=float)
    np.testing.assert_allclose(tcp_jacobian(q), mujoco_tcp_jacobian(m, d, site_id, q), atol=MJ_TOL)


def test_tcp_jacobian_matches_mujoco_random(mj):
    m, d, site_id, limits = mj
    err_w, err_p = [], []
    for q in random_configs(limits):
        J_ours, J_ref = tcp_jacobian(q), mujoco_tcp_jacobian(m, d, site_id, q)
        err_w.append(np.max(np.abs(J_ours[:3] - J_ref[:3])))
        err_p.append(np.max(np.abs(J_ours[3:] - J_ref[3:])))
    assert max(err_w) < MJ_TOL, f"max J_w error {max(err_w):.2e}"
    assert max(err_p) < MJ_TOL, f"max J_p error {max(err_p):.2e}"
