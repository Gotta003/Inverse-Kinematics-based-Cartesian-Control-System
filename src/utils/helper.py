import numpy as np

def vectorial_product(A, B):
    results = []
    for i, _ in enumerate(A):
        prod = []
        prod.append(A[i][1] * B[i][2] - A[i][2] * B[i][1])
        prod.append(A[i][2] * B[i][0] - A[i][0] * B[i][2])
        prod.append(A[i][0] * B[i][1] - A[i][1] * B[i][0])
        results.append(prod)
    return np.array(results)


def position_error(A, B):
    return float(np.linalg.norm(A[:3, 3]-B[:3, 3]))

def rotation_error(A, B):
    cos_angle=(np.trace(A.T@B)-1.0)/2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def deg_to_rad(theta_cal):
    return theta_cal * np.pi / 180


def derive_limits(m, njts):
    limits = {m.joint(i).name: m.jnt_range[i].tolist() for i in range(njts)}
    return limits

def interp(A, B, s, joints_list):
    return {j: (1-s)*A[j]+s*B[j] for j in joints_list}