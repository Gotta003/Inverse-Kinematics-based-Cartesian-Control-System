import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

def vec_to_matrix(V):
    return np.array([
        [0,   -V[2],  V[1]],
        [ V[2],   0,   -V[0]],
        [-V[1],  V[0],   0],
    ])

def matrix_exp(S, theta):
    omega = S[0:3]
    v = S[3:6]
    w_mat = vec_to_matrix(omega)
    
    # rodrigues formula
    R = np.eye(3) + np.sin(theta) * w_mat + (1 - np.cos(theta)) * (w_mat @ w_mat)
    
    # Translation vector p
    V = np.eye(3) * theta + (1 - np.cos(theta)) * w_mat + (theta - np.sin(theta)) * (w_mat @ w_mat)
    p = V @ v
    
    # Assemble 4x4 transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    
    return T

def UR3_fk(q_list):
    if len(np.asarray(q_list).flatten()) != 6:
        raise ValueError("Shape of the position array is wrong")
    
    M = np.array([
        [1, 0, 0, -192],
        [0, 1, 0, 0],
        [0, 0, 1, 692],
        [0, 0, 0, 1]
    ])
    
    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([1, 0, 0, 0, 152, 0])
    S3 = np.array([1, 0, 0, 0, 396, 0])
    S4 = np.array([1, 0, 0, 0, 609, 0])
    S5 = np.array([0, 0, 1, 0, -110, 0])
    S6 = np.array([1, 0, 0, 0, 692, 0])
    
    exp1 = matrix_exp(S1, q_list[0])
    exp2 = matrix_exp(S2, q_list[1])
    exp3 = matrix_exp(S3, q_list[2])
    exp4 = matrix_exp(S4, q_list[3])
    exp5 = matrix_exp(S5, q_list[4])
    exp6 = matrix_exp(S6, q_list[5])
    
    T = exp1 @ exp2 @ exp3 @ exp4 @ exp5 @ exp6 @ M
    
    return T

if __name__=="__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("Screw Axes:")
    print("M")
    print("T_0E")
    print("Links Lengths [m]:")
    