"""
Tests for src/inverse_kinematics.py

Run:
    pytest tests/test_ik.py -v
"""

import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
from src.forward_kinematics import SO_101_fk
from src.inverse_kinematics import MODES, METHODS, PAN_AXIS_XY, ik_position, inverse_kinematics, pitch_of
from src.jacobian import position_jacobian, tcp_jacobian
from src.utils.constants import IK_POS_TOL, IK_ROT_TOL, N_JOINTS, IK_MAX_JOINT_STEP, RND_SEED
from src.utils.limits import calibration_limits, inv_kin_limits, model_limits
from utils.so101_mujoco_utils import mujoco_starting, mujoco_tcp
from src.utils.helper import damped_pseudo_inverse, derive_limits, rotation_error

SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"
RANDOM_IK=100
LIMITS=inv_kin_limits()
SHOULDER_LIFT_POINT=np.array([0.062345, -0.0182778, 0.1166])
Q_STRETCHED=np.array([0.0, 0.0, np.radians(-78.44), 0.0, 0.0]) #Stretched arm

@pytest.fixture(scope="module")
def mj():
    m, d, site_id=mujoco_starting(str(SCENE_XML))
    return m, d, site_id

def test_damped_pseudoinverse_without_damping():
    J=np.random.default_rng(RND_SEED).normal(size=(3,5))
    Jp, _, lam, _=damped_pseudo_inverse(J)
    assert lam==0.0
    assert np.allclose(Jp, np.linalg.pinv(J), atol=1e-12)
    
def test_damped_pseudoinverse_bounded():
    U, _, Vt=np.linalg.svd(np.random.default_rng(RND_SEED).normal(size=(3,5)))
    J=U@np.diag([0.3, 0.1, 1e-6])@Vt[:3]
    eps=0.02
    lam_max=0.02
    Jp, sig, lam, N=damped_pseudo_inverse(J, eps, lam_max)
    assert sig==pytest.approx(1e-6)
    assert lam==pytest.approx(lam_max, rel=1e-6)
    assert np.linalg.norm(Jp, 2)<=1/(2*lam)+1e-9
    assert np.abs(J@N).max()<1e-12
    assert np.allclose(N@N, N, atol=1e-12)
    
def test_xml_limits_match_mujoco(mj):
    m=mj[0]
    assert np.allclose(model_limits(), np.array(list(derive_limits(m, N_JOINTS).values())))
    
def test_ik_limits_inside_model_and_calibration():
    lim=inv_kin_limits("jarvis")
    model=model_limits()
    cal=calibration_limits("jarvis")
    assert np.all(lim[:,0]>np.maximum(model[:,0], cal[:,0]))
    assert np.all(lim[:,1]<np.minimum(model[:,1], cal[:,1]))
    
def targets(n=RANDOM_IK, seed=RND_SEED):
    rng=np.random.default_rng(seed)
    return rng.uniform(LIMITS[:,0], LIMITS[:,1], size=(n, N_JOINTS))
    
def solve(q, mode, q0=None, **kw):
    T=SO_101_fk(q)
    if mode=="pitch":
        kw["pitch"]=pitch_of(T[:3, :3])
    if mode=="pose":
        kw["R_dest"]=T[:3, :3]
    return T, inverse_kinematics(T[:3, 3], q0=q0, mode=mode, **kw)
    
def check_solution(T, res, mode, mj=None):
    T_s=SO_101_fk(res.q)
    assert np.linalg.norm(T_s[:3, 3]-T[:3, 3])<IK_POS_TOL
    if mode=="pitch":
        assert abs(pitch_of(T_s[:3, :3])-pitch_of(T[:3, :3]))<IK_ROT_TOL
    if mode=="pose":
        assert rotation_error(T_s[:3, :3], T[:3, :3])<IK_ROT_TOL
    assert np.all(res.q>=LIMITS[:, 0]) and np.all(res.q<=LIMITS[:, 1])
    if mj is not None:
        T_mj=mujoco_tcp(*mj, res.q)
        assert np.linalg.norm(T_mj[:3,3]-T[:3,3])<IK_POS_TOL+1e-6
        
def failure_class(q, res):
    p=SO_101_fk(q)[:3, 3]
    if np.hypot(*(p[:2]-PAN_AXIS_XY))<0.015:
        return "pan_axis"
    if np.linalg.svd(position_jacobian(q), compute_uv=False)[-1]<0.03:
        return "near_singular"
    if np.any(np.isclose(res.q, LIMITS[:, 0]) | np.isclose(res.q, LIMITS[:, 1])):
        return "joint_limit"
    return None
        
@pytest.mark.parametrize("mode", MODES)
def test_round_trip_warm_start(mj, mode):
    rng=np.random.default_rng(RND_SEED)
    ok=0
    unexplained=[]
    for q in targets():
        q0=np.clip(q+rng.normal(0, 0.1, N_JOINTS), LIMITS[:,0], LIMITS[:,1])
        T, res=solve(q, mode, q0=q0, restarts=0)
        assert np.all(res.q>=LIMITS[:,0]) and np.all(res.q<=LIMITS[:,1])
        if res.success:
            ok+=1
            check_solution(T, res, mode, mj)
        elif failure_class(q, res) is None:
            unexplained.append(np.round(q, 3).tolist())
    assert not unexplained, f"unexplained failures at q={unexplained}"
    assert ok>=0.95*RANDOM_IK, f"{ok}/{RANDOM_IK}"
    
@pytest.mark.parametrize("mode", MODES)
def test_round_trip_cold_start(mode):
    ok=0
    for q in targets(seed=RND_SEED):
        T, res=solve(q, mode)
        assert np.all(res.q>=LIMITS[:, 0]) and np.all(res.q<=LIMITS[:, 1])
        if res.success:
            ok+=1
            check_solution(T, res, mode)
    assert ok>={
        "position": 1.0,
        "pitch": 0.9,
        "pose": 0.85
    }[mode]*RANDOM_IK, f"{ok}/{RANDOM_IK}"
    
@pytest.mark.parametrize("method", METHODS)
def test_methods_converge_on_consistent_pose(method):
    rng=np.random.default_rng(RND_SEED)
    for q in targets(20, seed=RND_SEED):
        q0=np.clip(q+rng.normal(0, 0.05, N_JOINTS), LIMITS[:, 0], LIMITS[:, 1])
        T, res=solve(q, "pose", q0=q0, restarts=0, method=method)
        assert res.success
        check_solution(T, res, "pose")
        
def test_warm_start_stays_on_branch():
    rng=np.random.default_rng(RND_SEED)
    for q in targets(30, seed=RND_SEED):
        if np.linalg.svd(tcp_jacobian(q), compute_uv=False)[-1]<0.05:
            continue
        q0=np.clip(q+rng.normal(0, 0.03, N_JOINTS), LIMITS[:, 0], LIMITS[:, 1])
        _, res=solve(q, "pose", q0=q0, restarts=0)
        assert res.success
        assert np.abs(res.q-q).max()<0.01
        
def test_target_beyond_limit():
    for j in range(4):
        q=np.zeros(N_JOINTS)
        q[j]=LIMITS[j, 1]+0.1
        T, res=solve(q, "pose")
        assert np.all(res.q>=LIMITS[:,0]) and np.all(res.q<=LIMITS[:,1])
        if res.success:
            check_solution(T, res, "pose")
            
def test_stretched_arm_near_singularity():
    s=np.linalg.svd(position_jacobian(Q_STRETCHED), compute_uv=False)
    assert s[-1]<0.02*s[0]
    
@pytest.mark.parametrize("q0", [np.zeros(N_JOINTS), Q_STRETCHED+0.05])
def test_exact_singular_target_converges(q0):
    res=ik_position(SO_101_fk(Q_STRETCHED)[:3, 3], q0=q0, restarts=0)
    assert res.success
    
@pytest.mark.parametrize("beyond", [0.005, 0.02, 0.1, 0.5])
def test_target_beyond_reach(beyond):
    p_s=SO_101_fk(Q_STRETCHED)[:3, 3]
    u=(p_s-SHOULDER_LIFT_POINT)/np.linalg.norm(p_s-SHOULDER_LIFT_POINT)
    res=ik_position(p_s+beyond*u, q0=Q_STRETCHED+0.03, restarts=0, record=True)
    q_hist=np.array([h["q"] for h in res.history])
    assert not res.success
    assert np.all(np.isfinite(q_hist))
    assert np.abs(np.diff(q_hist, axis=0)).max()<=IK_MAX_JOINT_STEP+1e-12
    assert np.all(res.q>=LIMITS[:, 0]) and np.all(res.q<=LIMITS[:, 1])
    assert beyond-1e-3<res.pos_err<beyond+1e-3