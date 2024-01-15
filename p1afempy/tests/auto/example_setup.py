import numpy as np


OMEGA = 7./4. * np.pi


def u(r: np.ndarray) -> float:
    """analytical solution"""
    return np.sin(OMEGA*2.*r[:, 0])*np.sin(OMEGA*r[:, 1])


def f(r: np.ndarray) -> float:
    """volume force corresponding to analytical solution"""
    return 5. * OMEGA**2 * np.sin(OMEGA*2.*r[:, 0]) * np.sin(OMEGA*r[:, 1])


def uD(r: np.ndarray) -> float:
    """solution value on the Dirichlet boundary"""
    return u(r)


def g_right(r: np.ndarray) -> float:
    return -2.*OMEGA*np.sin(OMEGA*r[:, 1])*np.cos(OMEGA*2.*r[:, 0])


def g_upper(r: np.ndarray) -> float:
    return OMEGA*np.sin(OMEGA*2.*r[:, 0]) * np.cos(OMEGA*r[:, 1])


def g(r: np.ndarray) -> float:
    out = np.zeros(r.shape[0])
    right_indices = r[:, 0] == 1
    upper_indices = r[:, 1] == 1
    out[right_indices] = g_right(r[right_indices])
    out[upper_indices] = g_upper(r[upper_indices])
    return out
