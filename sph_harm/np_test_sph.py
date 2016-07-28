import numpy as np
import math

nsamps = 102400000

theta = np.linspace(0, 2*math.pi, nsamps)
phi = np.linspace(0, 2*math.pi, nsamps)

selected_modes = [(2,-2),(2,0),(2,2)]

_a2m2 = math.sqrt(5.0 / (64.0 * math.pi))
_a2m1 = math.sqrt(5.0 / (16.0 * math.pi))
_a20  = math.sqrt(15.0 /(32.0 * math.pi))
_a21  = math.sqrt(5.0 / (16.0 * math.pi))
_a22  = math.sqrt(5.0 / (64.0 * math.pi))

def vector_compute_spherical_harmonics(theta, phi, selected_modes=None):
    """
    Compute spherical harmonics for various m with l=2.
    """
    # Only copies if input is scalar, else returns original array
    theta   = np.array(theta, copy=False, ndmin=1)
    theta   = np.array(phi, copy=False, ndmin=1)

    Ylms = {}
    one_m_theta = 1.0 - np.cos(theta)
    one_p_theta = 1.0 + np.cos(theta)
    snTheta = np.sin(theta)
    for m in range(-2, 3):
        if selected_modes is not None and (2,m) not in selected_modes:
            continue
        if m == -2:
            Ylms[(2,m)] = _a2m2 * one_m_theta * one_m_theta * np.exp(1j*m*phi)
        elif m == -1:
            Ylms[(2,m)] = _a2m1 * snTheta * one_m_theta * np.exp(1j*m*phi)
        elif m == 0:
            Ylms[(2,m)] = _a20 * snTheta * snTheta
        elif m == 1:
            Ylms[(2,m)] = _a21 * snTheta * one_p_theta * np.exp(1j*m*phi)
        elif m == 2:
            Ylms[(2,m)] = _a22 * one_p_theta * one_p_theta * np.exp(1j*m*phi)

    return Ylms

vector_compute_spherical_harmonics(theta, phi, selected_modes)
