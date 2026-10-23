"""
extract_poe_params.py
Extracts the PoE parameters of the SO-101 from the MuJoCo model (model_files/so101_new_calib.xml) at configuration q=0:
- OMEGA         unit rotation axis of each joint
- POSITIONS     point on each joint axis
- M             pose of TCP 
- theta_cal     roll of TCP frame about x, from two entries of M's rotation

Run:
    python tools/extract_poe_params.py
"""
import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import mujoco
import numpy as np
from scipy.linalg import expm

SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_SITE = "gripperframe"

def extract(m, d):
    d.qpos[:]=0.0
    mujoco.mj_forward(m, d)
    omega, points=[], []
    for name in JOINT_NAMES:
        j=m.joint(name).id
        omega.append(d.xaxis[j].copy())
        points.append(d.xanchor[j].copy())
    s=m.site(TCP_SITE).id
    R=d.site_xmat[s].reshape(3, 3).copy()
    p=d.site_xpos[s].copy()
    return np.array(omega), np.array(points), R, p

def poe_fk(omega, points, M, q):
    T=np.eye(4)
    for w, p, th in zip(omega, points, q):
        S=np.zeros((4, 4))
        S[:3, :3]=[
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ]
        S[:3, 3]=-np.cross(w, p)
        T=T@expm(S*th)
    return T@M

def main():
    m=mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d=mujoco.MjData(m)
    omega, points, R, p=extract(m, d)
    omega=np.where(np.abs(omega)<1e-9, 0.0, omega)
    sin_t=R[2, 1]
    cos_t=R[1, 1]
    theta_cal=np.arctan2(sin_t, cos_t)
    Rx=np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])
    print(np.array2string(R, precision=9, suppress_small=True))
    print(f"  R[2,1] = sin(theta_cal) = {sin_t:.9f}")
    print(f"  R[1,1] = cos(theta_cal) = {cos_t:.9f}")
    print(f"  theta_cal = arctan2({sin_t:.9f}, {cos_t:.9f}) = {theta_cal:.9f} rad "
          f"= {np.degrees(theta_cal):.4f} deg")
    
    print("\n# ---- paste into src/utils/constants.py (metres, radians) ----")
    print(f"theta_cal = np.arctan2({sin_t:.9f}, {cos_t:.9f})")
    print("M = np.array([")
    print(f"    [1, 0, 0, {p[0]:.7f}],")
    print(f"    [0, np.cos(theta_cal), -np.sin(theta_cal), {p[1]:.7f}],")
    print(f"    [0, np.sin(theta_cal), np.cos(theta_cal), {p[2]:.7f}],")
    print("    [0, 0, 0, 1],")
    print("])")
    print("OMEGA = np.array([")
    for name, w in zip(JOINT_NAMES, omega):
        print(f"    [{w[0]:g}, {w[1]:g}, {w[2]:g}],   # {name}")
    print("])")
    print("POSITIONS = np.array([")
    for name, q in zip(JOINT_NAMES, points):
        print(f"    [{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}],   # {name}")
    print("])")
 
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    s = m.site(TCP_SITE).id
    rng = np.random.default_rng(0)
    lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
    err = 0.0
    for q in rng.uniform(lo, hi, size=(200, 5)):
        d.qpos[:] = 0.0
        d.qpos[:5] = q
        mujoco.mj_forward(m, d)
        err = max(err, np.linalg.norm(poe_fk(omega, points, M, q)[:3, 3] - d.site_xpos[s]))
    print(f"\nCheck: PoE from these parameters vs MuJoCo, 200 random q: max error {err:.1e} m")
 
    try:
        from src.utils import constants as C
        print("\nDifference with src/utils/constants.py:")
        print(f"  theta_cal : {abs(C.theta_cal - theta_cal):.1e} rad")
        print(f"  M position: {np.abs(C.M[:3, 3] - p).max():.1e} m")
        print(f"  OMEGA     : {np.abs(np.asarray(C.OMEGA) - omega).max():.1e}")
        v_c = -np.cross(np.asarray(C.OMEGA, float), np.asarray(C.POSITIONS, float))
        v_x = -np.cross(omega, points)
        print(f"  POSITIONS : {np.abs(np.asarray(C.POSITIONS) - points).max():.1e} m "
              f"(effect on v = -w x p: {np.abs(v_c - v_x).max():.1e} m)")
    except Exception as e:
        print(f"\n(could not compare with constants.py: {e})")
        
if __name__=="__main__":
    main()