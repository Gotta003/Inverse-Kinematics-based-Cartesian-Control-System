import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from src.utils.constants import OMEGA, POSITIONS, M, V


def print_matrix(m, name="Matrix", decimals=4, width=11):
    m = np.atleast_2d(np.squeeze(np.array(m, dtype=float)))
    m = np.round(m, decimals) + 0.0
    total = width * m.shape[1]
    print("=" * total)
    print(name.center(total))
    print("=" * total)
    for row in m:
        print("".join(f"{x:>{width}.{decimals}f}" for x in row))


def vec_to_matrix(V):
    return np.array(
        [
            [0, -V[2], V[1]],
            [V[2], 0, -V[0]],
            [-V[1], V[0], 0],
        ]
    )


def matrix_exp(omega, v, theta):
    w_mat = vec_to_matrix(omega)
    # rodrigues formula
    R = np.eye(3) + np.sin(theta) * w_mat + (1 - np.cos(theta)) * (w_mat @ w_mat)
    # Translation vector p
    V = (
        np.eye(3) * theta
        + (1 - np.cos(theta)) * w_mat
        + (theta - np.sin(theta)) * (w_mat @ w_mat)
    )
    p = V @ v
    # Assemble 4x4 transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


def SO_101_fk(q_list):
    if len(np.asarray(q_list).flatten()) != 5:
        raise ValueError("Shape of the position array is wrong")
    T = np.eye(4)
    for i, _ in enumerate(V):
        T = T @ matrix_exp(OMEGA[i], V[i], q_list[i])
    T = T @ M
    return T


def model_link_lengths():
    p = POSITIONS
    xz = [0, 2]
    return {
        "base_to_shoulder_lift": float(p[1, 2]),
        "upper_arm": float(np.linalg.norm(p[2, xz] - p[1, xz])),
        "forearm": float(np.linalg.norm(p[3, xz] - p[2, xz])),
        "wrist_to_tcp": float(np.linalg.norm(M[[0, 2], 3] - p[3, xz])),
    }


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print_matrix(OMEGA.T, name="Screw Axes")
    print_matrix(POSITIONS.T, name="Positions")
    print_matrix(V.T, name="Velocities")
    print_matrix(M, name="M")
    print_matrix(SO_101_fk([0, 0, 0, 0, 0]), name="T_0E")
    print_matrix(SO_101_fk([-np.pi / 2, 0, 0, 0, 0]), name="Move to left")
    print("Links Lengths [m]:", model_link_lengths())
