"""ISA atmosphere, effective sound speed along a slant path, ISO 9613-1 absorption."""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
R_AIR = 287.05287          # J/(kg K)
T0 = 288.15                # K
P0 = 101325.0              # Pa
LAPSE = -0.0065            # K/m (troposphere)


def isa_temperature(h_m, dT=0.0):
    """ISA temperature [K] at geopotential-ish height h (m), with ISA+dT offset."""
    h = np.asarray(h_m, dtype=float)
    t = np.where(h < 11000.0, T0 + LAPSE * h, T0 + LAPSE * 11000.0)
    return t + dT


def isa_pressure(h_m):
    h = np.asarray(h_m, dtype=float)
    g0 = 9.80665
    t11 = T0 + LAPSE * 11000.0
    p11 = P0 * (t11 / T0) ** (-g0 / (LAPSE * R_AIR))
    tro = P0 * ((T0 + LAPSE * np.minimum(h, 11000.0)) / T0) ** (-g0 / (LAPSE * R_AIR))
    strat = p11 * np.exp(-g0 * (h - 11000.0) / (R_AIR * t11))
    return np.where(h < 11000.0, tro, strat)


def speed_of_sound(T_kelvin):
    return np.sqrt(GAMMA * R_AIR * np.asarray(T_kelvin, dtype=float))


def sound_speed_at(h_m, dT=0.0, temp_profile=None):
    """c(h). temp_profile: optional callable h->T[K] (e.g. from a radiosonde)."""
    T = temp_profile(h_m) if temp_profile is not None else isa_temperature(h_m, dT)
    return speed_of_sound(T)


def effective_sound_speed(h_obs, h_src, dT=0.0, temp_profile=None, n=8):
    """Path-averaged sound speed for a straight ray between two heights.

    Travel time along a straight ray is T = (R/dz) * integral dz / c(z), so the
    correct average is the *harmonic* mean of c over height:
        c_eff = dz / integral_{h_obs}^{h_src} dz / c(z)
    (ray bending is ignored). Vectorised over h_src. Gauss-Legendre, n nodes.
    """
    h_obs = np.asarray(h_obs, dtype=float)
    h_src = np.asarray(h_src, dtype=float)
    x, w = np.polynomial.legendre.leggauss(n)
    lo, hi = np.broadcast_arrays(h_obs, h_src)
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    hs = mid[..., None] + half[..., None] * x
    inv_c = 1.0 / sound_speed_at(hs, dT, temp_profile)
    mean_inv = (inv_c * w).sum(-1) / 2.0
    return 1.0 / mean_inv


def iso9613_absorption_db_per_m(f_hz, T_kelvin=293.15, rh_percent=60.0, p_pa=P0):
    """Pure-tone atmospheric absorption coefficient alpha [dB/m], ISO 9613-1:1993."""
    f = np.asarray(f_hz, dtype=float)
    T = np.asarray(T_kelvin, dtype=float)
    pr = 101325.0
    T01 = 273.16
    Tref = 293.15
    psat_ratio = 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = rh_percent * psat_ratio * (pr / p_pa)  # molar concentration of water vapour [%]
    pa_r = p_pa / pr
    frO = pa_r * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = pa_r * (T / Tref) ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * ((T / Tref) ** (-1 / 3) - 1.0)))
    alpha = 8.686 * f ** 2 * (
        1.84e-11 / pa_r * (T / Tref) ** 0.5
        + (T / Tref) ** -2.5 * (
            0.01275 * np.exp(-2239.1 / T) / (frO + f ** 2 / frO)
            + 0.1068 * np.exp(-3352.0 / T) / (frN + f ** 2 / frN)
        )
    )
    return alpha


def received_amplitude(R_m, f_hz, ref_dist=1.0, T_kelvin=288.15, rh=60.0):
    """Relative pressure amplitude: spherical spreading (1/R) x ISO absorption."""
    R = np.maximum(np.asarray(R_m, dtype=float), ref_dist)
    a_db = iso9613_absorption_db_per_m(f_hz, T_kelvin, rh) * R
    return (ref_dist / R) * 10.0 ** (-a_db / 20.0)
