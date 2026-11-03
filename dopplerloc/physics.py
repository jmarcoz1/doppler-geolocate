r"""Forward model: what frequency does a static observer hear from a moving aircraft?

Conventions
-----------
* All geometry in WGS84 ECEF metres; time in unix seconds.
* u = (P_src - P_obs)/R is the unit vector from observer to source.
* v_r = V_src . u is the *radial* (range-rate) velocity: **positive = receding**.
* n = -u is the propagation direction (source -> observer); wind w (ENU, m/s,
  vector the air moves *towards*) changes the propagation speed along the ray:
  c_w = c + w . n.
* Moving source, static observer, moving medium:

      f_obs(t_obs) = f_src * c_w / (c_w + v_r(t_e))
                   = f_src * c_w / (c_w - v_approach),   v_approach = -v_r

  evaluated at the *emission* (retarded) time t_e which solves

      t_obs = t_e + R(t_e) / c_w(t_e)

  (Newton iteration; g'(t_e) = 1 + v_r/c_w > 0 for subsonic sources, so it is
  well-posed and converges in a few steps).
* c is the effective (harmonic-mean) ISA sound speed between observer and
  source height (see atmosphere.effective_sound_speed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import atmosphere, geodesy


@dataclass
class Observer:
    lat: float
    lon: float
    h: float  # ellipsoidal height of the microphone [m]

    @property
    def ecef(self):
        return geodesy.geodetic_to_ecef(self.lat, self.lon, self.h)


@dataclass
class Propagation:
    dT: float = 0.0                 # ISA temperature offset [K]
    wind_enu: tuple | None = None   # (we, wn) m/s, uniform along path
    temp_profile: object = None     # callable h -> T [K]
    h_ground: float | None = None   # height datum for the ISA profile (default: observer h)
    fixed_c: float | None = None    # override everything with a constant c


def _sound_speed(obs: Observer, src_h, prop: Propagation):
    if prop.fixed_c is not None:
        return np.full_like(np.asarray(src_h, dtype=float), prop.fixed_c)
    h0 = obs.h if prop.h_ground is None else prop.h_ground
    # ISA uses height above MSL; ellipsoidal vs MSL (~50 m) is negligible for c.
    return atmosphere.effective_sound_speed(np.full_like(src_h, h0, dtype=float), src_h,
                                            prop.dT, prop.temp_profile)


def _wind_ecef(obs: Observer, prop: Propagation):
    if prop.wind_enu is None:
        return None
    we, wn = prop.wind_enu[:2]
    return geodesy.enu_velocity_to_ecef(np.array([we, wn, 0.0]), obs.lat, obs.lon)


def emission_time(model, obs: Observer, t_obs, prop: Propagation | None = None, iters: int = 6):
    """Solve t_obs = t_e + R(t_e)/c_w for t_e (vectorised). Returns (t_e, geom dict)."""
    prop = prop or Propagation()
    t_obs = np.asarray(t_obs, dtype=float)
    O = obs.ecef
    w = _wind_ecef(obs, prop)
    c_guess = 340.0
    te = t_obs - np.linalg.norm(model.position(t_obs) - O, axis=-1) / c_guess
    for _ in range(iters):
        P, V = model.position(te), model.velocity(te)
        d = P - O
        R = np.linalg.norm(d, axis=-1)
        u = d / R[..., None]
        _, _, hs = geodesy.ecef_to_geodetic(P)
        c = _sound_speed(obs, hs, prop)
        cw = c - (u @ w if w is not None else 0.0)   # n = -u
        vr = np.einsum("...i,...i->...", V, u)
        g = te + R / cw - t_obs
        te = te - g / (1.0 + vr / cw)
    P, V = model.position(te), model.velocity(te)
    d = P - O
    R = np.linalg.norm(d, axis=-1)
    u = d / R[..., None]
    _, _, hs = geodesy.ecef_to_geodetic(P)
    c = _sound_speed(obs, hs, prop)
    cw = c - (u @ w if w is not None else 0.0)
    vr = np.einsum("...i,...i->...", V, u)
    return te, dict(R=R, u=u, vr=vr, c=cw, h_src=hs, P=P, V=V)


def doppler_factor(vr, c):
    """Moving source / static observer: f_obs/f_src = c/(c + v_r), v_r>0 receding."""
    return c / (c + vr)


def predict_fobs(model, obs: Observer, t_obs, f_src, prop: Propagation | None = None):
    te, g = emission_time(model, obs, t_obs, prop)
    return f_src * doppler_factor(g["vr"], g["c"]), te, g


def cpa(model, obs: Observer, t_grid):
    """Closest point of approach (time, slant range) on a sampled time grid."""
    R = np.linalg.norm(model.position(t_grid) - obs.ecef, axis=-1)
    i = int(np.argmin(R))
    return float(t_grid[i]), float(R[i])


# --------------------------------------------------------------------- Lloyd's mirror
def lloyd_delay(src_ecef, obs: Observer, h_receiver: float, c=340.0):
    """Delay tau between reflected and direct path for flat ground.

    The ground plane is at ellipsoidal height obs.h - h_receiver. With source
    height above ground z_s, horizontal distance d:
        Delta = sqrt(d^2 + (z_s + h_r)^2) - sqrt(d^2 + (z_s - h_r)^2) ~= 2 h_r sin(elev)
        tau   = Delta / c
    """
    ground = Observer(obs.lat, obs.lon, obs.h - h_receiver)
    enu = geodesy.ecef_to_enu(src_ecef, ground.lat, ground.lon, ground.h)
    d = np.hypot(enu[..., 0], enu[..., 1])
    zs = enu[..., 2]
    delta = np.hypot(d, zs + h_receiver) - np.hypot(d, zs - h_receiver)
    return delta / c


def lloyd_notches(tau, f_max, hard_ground=True):
    """Notch frequencies of the comb filter 1 + Q e^{-i 2 pi f tau}.

    Hard ground (Q ~ +1): destructive interference at f_n = (n + 1/2)/tau.
    Soft/pressure-release (Q ~ -1): f_n = n/tau (n >= 1).
    Returns array (..., n_max) padded with NaN beyond f_max.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    n_max = int(np.ceil(np.nanmax(f_max * tau))) + 1
    n = np.arange(n_max) + (0.5 if hard_ground else 1.0)
    f = n[None, :] / tau[:, None]
    return np.where(f <= f_max, f, np.nan)
