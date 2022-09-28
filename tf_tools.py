import numpy as np

from consts import RADIUS_EARTH


def polar2cart(phi, lambda_, r):
    xc = r * np.cos(phi) * np.cos(lambda_)
    yc = r * np.cos(phi) * np.sin(lambda_)
    zc = r * np.sin(phi)
    return np.array([xc, yc, zc])


def get_r(p_c):  # 地心距
    if p_c.ndim == 1:
        return np.sqrt(p_c[0] ** 2 + p_c[1] ** 2 + p_c[2] ** 2)
    else:
        return np.linalg.norm(p_c, axis=0)


def get_phi(p_c):  # 地心纬度 [-pi/2, pi.2]
    return np.arcsin(p_c[2] / get_r(p_c))


def get_lamda(p_c):  # 地心经度 (-pi, pi]
    return np.arctan2(p_c[1], p_c[0])


def get_H(p_c):  # 地心高度
    return get_r(p_c) - RADIUS_EARTH


def build_M_c2f(phi_0, lambda_0, A_T):
    s, c = np.sin, np.cos
    p, l, a = phi_0, lambda_0, A_T  # 天文纬度、天文经度、发射瞄准角
    return np.array([
        [-s(l) * s(a) - c(l) * s(p) * c(a), c(l) * c(p),
         c(l) * s(p) * s(a) - s(l) * c(a)],
        [c(l) * s(a) - s(l) * s(p) * c(a), s(l) * c(p),
         c(l) * c(a) + s(l) * s(p) * s(a)],
        [c(p) * c(a), s(p), -c(p) * s(a)],
    ])


def build_M_f2c(phi_0, lambda_0, A_T):
    return np.linalg.inv(build_M_c2f(phi_0, lambda_0, A_T))


def build_M_s2c(p_c: np.ndarray):
    s, c = np.sin, np.cos
    l, p = get_lamda(p_c), get_phi(p_c)  # 地心纬度、地心经度
    return np.array([
        [-s(l), c(l) * c(p), c(l) * s(p)],
        [c(l), s(l) * c(p), s(l) * s(p)],
        [0, s(p), -c(p)]
    ])


def build_M_c2s(p_c: np.ndarray):
    return np.linalg.inv(build_M_s2c(p_c))


def build_M_n2b():  # 以后再弄
    pass


def Pf_2_Pc(M_f2c, x, y, z, xc_0, yc_0, zc_0):
    return np.matmul(M_f2c, np.array([x, y, z]).T) \
           + np.array([xc_0, yc_0, zc_0])


def Vf_2_Vc(M_f2c, vx, vy, vz):
    np.matmul(M_f2c, np.array([vx, vy, vz]).T)
