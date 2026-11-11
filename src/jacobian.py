"""
jacobian.py

Manipulator Jacobians of SO101 using PoE screw axes from Phase 1

    - SO_101_space_jacobian(q) 6x5 rows [omega; v] with a column i being the twist of joint i expressed in base frame
    - tcp_jacobian(q) 6x5 rows [J_w; J_p] angular velocity of the tool and linear velocity of TCP point both in base frame (quantities to match mj_jacSite outputs [jacr, jacp])
    - position_jacobian(q) 3x5 block J_p, used by the position task of IK 
    - jacobian_metrics(J) singular values, sigma_min, manipulability, condition
    
Run:
    python src/jacobian.py
"""
import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.forward_kinematics import SO_101_fk, matrix_exp, print_matrix
from src.utils.constants import N_JOINTS, OMEGA, V
from src.utils.helper import adjoint, skew

SCREW_AXES=np.hstack((OMEGA, V))

def check_dimension(q):
    """Return q as a float array of length N_JOINTS or raise"""
    q=np.asanyarray(q, dtype=float).flatten()
    if q.size!=N_JOINTS:
        raise ValueError("Shape of position array is wrong")
    return q

def SO_101_space_jacobian(q):
    """Space Jacobian with J_s,1=S1 and J_s,i=Ad(e^[S_1]q_1 ... e^[S_{i-1}]q_{i-1}) S_i"""
    q=check_dimension(q)
    J=np.zeros((6, N_JOINTS))
    T=np.eye(4)
    for i in range(N_JOINTS):
        J[:, i]=adjoint(T)@SCREW_AXES[i]     #Twist joint i
        T=T@matrix_exp(OMEGA[i], V[i], q[i]) #Concatenation
    return J

def tcp_jacobian(q, p=None):
    """Geometric Jacobian of TCP (6x5) with linear part v of space Jacobian is velocity of the body, coinciding with base origin
    With p_dot=v+w x p J_p=J_v-[p] x J_w
    """
    q=check_dimension(q)
    J_s=SO_101_space_jacobian(q)
    if p is None:
        p=SO_101_fk(q)[:3, 3]
    J_w=J_s[:3]                #Angular part
    J_p=J_s[3:]-skew(p)@J_w    #Move linear part of TCP
    return np.vstack((J_w, J_p))

def position_jacobian(q, p=None):
    """3x5 block d(p of TCP)/dq of tcp_jacobian"""
    return tcp_jacobian(q, p)[3:]

def jacobian_metrics(J):
    """Conditioning of J:
        - Singular valeus
        - sigma_min (distance from a singularity for damping of IK)
        - Yoshikawa manipulability (product of singular values=sqrt(def(J*J^T)))
        - Condition number
    """
    s=np.linalg.svd(J, compute_uv=False)
    manipulability=float(np.prod(s))
    condition=float(s[0]/s[-1]) if s[-1]>0 else np.inf
    return {
        "sigma": s,
        "sigma_min": float(s[-1]),
        "manipulability": manipulability,
        "cond": condition
    }

def main():
    np.set_printoptions(precision=4, suppress=True)
    q0=np.zeros(N_JOINTS)
    print_matrix(SO_101_space_jacobian(q0), name="J_s(0)=screw axes")
    print_matrix(tcp_jacobian(q0), name="TCP jacobian at q=0 [J_w; J_p]")
    m=jacobian_metrics(position_jacobian(q0))
    print("J_p(0) singular values:", m["sigma"], " manipulability:" f"{m['manipulability']:.3e}")

if __name__=="__main__":
    main()