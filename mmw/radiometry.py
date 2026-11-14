"""Long-wave infrared (8-14 um) band radiance <-> temperature.

A microbolometer pixel integrates radiance, not temperature. When a pixel sees half a warm head
and half the cool floor, the reading is the radiance average converted back to temperature,
which is warmer than the temperature average. Rendering and pixel integration therefore work
in radiance and only the final reading is converted to degrees.
"""
from __future__ import annotations

import numpy as np

_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
_T = np.arange(-60.0, 400.0, 0.02)          # degC table
_LAM = np.linspace(8e-6, 14e-6, 301)         # LWIR band


def _band_radiance(t_c: np.ndarray) -> np.ndarray:
    T = t_c[:, None] + 273.15
    lam = _LAM[None, :]
    planck = 2 * _H * _C ** 2 / lam ** 5 / np.expm1(_H * _C / (lam * _KB * T))
    return np.trapezoid(planck, _LAM, axis=1)  # W / (m^2 sr)


_L = _band_radiance(_T)


def radiance(t_c):
    return np.interp(t_c, _T, _L)


def temperature(L):
    return np.interp(L, _L, _T)


def apparent_radiance(t_surface, emissivity, t_reflected):
    """What reaches the camera from an opaque surface: emitted + reflected surroundings."""
    return emissivity * radiance(t_surface) + (1.0 - emissivity) * radiance(t_reflected)
