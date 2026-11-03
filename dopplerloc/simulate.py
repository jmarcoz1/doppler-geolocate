"""Synthetic scenario near Valencia airport (LEVC): flight tracks + flyover audio.

Audio is synthesised *physically*: for every audio sample time t_obs we solve
the retarded time t_e for the direct path and for the ground-reflected path
(image receiver), and evaluate the source signal at t_e:
  tones  : A_k sin(2 pi f_k t_e + phi_k)          -> Doppler emerges naturally
  jet/fan: broadband noise n(t_e)                 -> Lloyd's-mirror fringes
scaled by 1/R spherical spreading and ISO 9613-1 absorption, plus 50 Hz mains
hum with harmonics, background noise and wind rumble.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from . import geodesy, tracks
from .atmosphere import iso9613_absorption_db_per_m, effective_sound_speed
from .physics import Observer

LEVC = (39.4893, -0.4816, 69.0)          # ARP lat, lon, elevation ~ 69 m MSL (approx)
RWY12_THR = (39.4985, -0.4999)           # approx. threshold RWY 12 (illustrative)
RWY12_HDG = 124.0
GEOID_N = 50.0


@dataclass
class Aircraft:
    icao24: str
    callsign: str
    enu_fn: object            # t -> (N,3) ENU position wrt LEVC origin (ellipsoidal)
    t0: float
    t1: float
    tones: tuple = ()         # ((f_src, amplitude), ...)
    noise_level: float = 0.0  # broadband source level
    audible: bool = True


def _straight(p0, heading_deg, speed, vs, t_ref, accel=0.0):
    hd = np.radians(heading_deg)
    dirv = np.array([np.sin(hd), np.cos(hd), 0.0])

    def f(t):
        tau = np.asarray(t, dtype=float) - t_ref
        s = speed * tau + 0.5 * accel * tau ** 2
        p = p0[None, :] + s[:, None] * dirv[None, :]
        p[:, 2] = p0[2] + vs * tau
        return p
    return f


def _bank_turn(f_lin, t_turn, rate_deg_s, speed, heading_deg):
    """Straight then constant-rate turn starting at t_turn (adds realism/geometry)."""
    def f(t):
        t = np.asarray(t, dtype=float)
        p = f_lin(t)
        after = t > t_turn
        if after.any():
            p_turn = f_lin(np.array([t_turn]))[0]
            om = np.radians(rate_deg_s)
            r = speed / om
            hd0 = np.radians(heading_deg)
            tau = t[after] - t_turn
            ang = hd0 + om * tau
            # centre of turn to the right of heading
            cx = p_turn[0] + r * np.cos(hd0)
            cy = p_turn[1] - r * np.sin(hd0)
            p[after, 0] = cx - r * np.cos(ang)
            p[after, 1] = cy + r * np.sin(ang)
        return p
    return f


def scenario(t_rec0=1_793_750_400.0):
    """Hidden observer + 2 audible aircraft + 1 inaudible distractor. Returns dict."""
    lat0, lon0, elev = LEVC
    h0 = elev + GEOID_N
    o = dict(lat0=lat0, lon0=lon0, h0=h0)
    # threshold in ENU
    thr = geodesy.geodetic_to_enu(RWY12_THR[0], RWY12_THR[1], h0, lat0, lon0, h0)
    gs = np.tan(np.radians(3.0))
    # A: Arrival RWY12 on 3 deg glide path, 70 m/s, passes ~7 km from threshold at t_rec0+45 s
    v_a = 72.0
    back = np.array([-np.sin(np.radians(RWY12_HDG)), -np.cos(np.radians(RWY12_HDG)), 0.0])
    t_cpa_a = t_rec0 + 45.0
    d_cpa = 7000.0
    p_cpa = thr + back * d_cpa
    p_cpa[2] = 15.0 + d_cpa * gs
    A = Aircraft("3451c2", "VLG8KP", _straight(p_cpa, RWY12_HDG, v_a, -v_a * gs, t_cpa_a, accel=-0.15),
                 t_rec0 - 120, t_rec0 + 150, tones=((612.0, 1.0), (1224.0, 0.45), (1538.0, 0.6)),
                 noise_level=0.8)
    # C: distractor high cruise 25 km away (in ADS-B data, not audible)
    C = Aircraft("39d2a7", "AFR706", _straight(np.array([22000.0, 15000.0, 11500.0]), 180.0, 235.0, 0.0,
                                               t_rec0 + 60), t_rec0 - 120, t_rec0 + 200, audible=False)
    # hidden observer: ~6.3 km NW of LEVC, ~900 m south-west of the RWY12 centreline
    e_obs = p_cpa[:2] + np.array([np.sin(np.radians(RWY12_HDG + 90)), np.cos(np.radians(RWY12_HDG + 90))]) * 900.0
    obs_lat, obs_lon, _ = geodesy.enu_to_geodetic(np.array([e_obs[0], e_obs[1], 0.0]), lat0, lon0, h0)
    obs = Observer(float(obs_lat), float(obs_lon), h0 + 1.5)   # mic 1.5 m above ground
    # B: departure/climb heading 035 at ~1500 m, 150 m/s, passing ~2 km south-east of the
    #    observer at t_rec0+100 s, then a 1.5 deg/s right turn (non-parallel track breaks the
    #    left/right mirror ambiguity of a single straight pass)
    hb = np.radians(35.0)
    along, right = np.array([np.sin(hb), np.cos(hb)]), np.array([np.cos(hb), -np.sin(hb)])
    p_b = np.array([*(e_obs + right * 2000.0), 1500.0])
    t_cpa_b = t_rec0 + 100.0
    fB = _straight(p_b, 35.0, 150.0, 7.0, t_cpa_b)
    B = Aircraft("4ca8b1", "RYR3TF", _bank_turn(fB, t_cpa_b + 25, 1.5, 150.0, 35.0), t_rec0 - 120, t_rec0 + 200,
                 tones=((455.0, 1.6), (910.0, 0.7)), noise_level=1.0)
    return dict(origin=o, aircraft=[A, B, C], observer=obs, h_receiver=1.5, t_rec0=t_rec0)


def adsb_samples(sc, period=5.0, pos_noise_m=4.0, seed=1) -> pd.DataFrame:
    """ADS-B-like reports: `period` s sampling, position noise, 25 ft altitude quantisation."""
    rng = np.random.default_rng(seed)
    o = sc["origin"]
    rows = []
    for ac in sc["aircraft"]:
        t = np.arange(ac.t0, ac.t1, period) + rng.uniform(0, period)
        p = ac.enu_fn(t)
        dtv = 0.5
        v = (ac.enu_fn(t + dtv) - ac.enu_fn(t - dtv)) / (2 * dtv)
        pn = p + rng.normal(0, pos_noise_m, p.shape) * np.array([1, 1, 0])
        lat, lon, h = geodesy.enu_to_geodetic(pn, o["lat0"], o["lon0"], o["h0"])
        hq = np.round(h / (25 * tracks.FT)) * 25 * tracks.FT
        rows.append(pd.DataFrame(dict(
            time=t, icao24=ac.icao24, callsign=ac.callsign, lat=lat, lon=lon, alt_geom=hq,
            alt_baro=hq - 30.0, gs=np.hypot(v[:, 0], v[:, 1]), track=np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360,
            vrate=v[:, 2], on_ground=False, source="synthetic")))
    return tracks.normalize(pd.concat(rows))


def _retarded(ac, o, rx_ecef, t_obs, h_rx, c_fn, iters=30):
    tt = np.arange(ac.t0, ac.t1, 0.05)
    P_enu = ac.enu_fn(tt)
    P = geodesy.enu_to_ecef(P_enu, o["lat0"], o["lon0"], o["h0"])
    c_s = c_fn(P_enu[:, 2] + o["h0"])

    def interp(te):
        x = np.clip((te - tt[0]) / 0.05, 0, len(tt) - 1.0001)
        i = x.astype(int)
        a = (x - i)[:, None]
        return P[i] * (1 - a) + P[i + 1] * a, c_s[i] * (1 - a[:, 0]) + c_s[i + 1] * a[:, 0]

    Pe, c = interp(t_obs)
    te = t_obs - np.linalg.norm(Pe - rx_ecef, axis=1) / c
    for _ in range(iters):  # fixed point; contraction factor v/c < 0.5 -> 30 iters ~ 1e-10
        Pe, c = interp(te)
        R = np.linalg.norm(Pe - rx_ecef, axis=1)
        te = t_obs - R / c
    return te, R


def synthesize(sc, duration=150.0, sr=8000, clock_error=1.3, seed=7, snr_scale=1.0, ground_q=0.9,
               hum_level=0.02, background=0.004):
    """Render the recording heard by the hidden observer. Returns (x, sr, nominal_start)."""
    rng = np.random.default_rng(seed)
    o, obs, hr = sc["origin"], sc["observer"], sc["h_receiver"]
    t_obs = sc["t_rec0"] + np.arange(int(duration * sr)) / sr
    rx = obs.ecef
    img = geodesy.geodetic_to_ecef(obs.lat, obs.lon, obs.h - 2 * hr)
    c_fn = lambda hs: effective_sound_speed(np.full_like(hs, obs.h), hs)
    x = np.zeros_like(t_obs)
    for ac in sc["aircraft"]:
        if not ac.audible:
            continue
        # broadband source noise on a uniform emission-time grid (shaped: jet/fan hump)
        tn = np.arange(ac.t0 - 30, ac.t1, 1 / sr)
        nz = rng.normal(0, 1, len(tn))
        sos = signal.butter(2, [150, 2500], "bandpass", fs=sr, output="sos")
        nz = signal.sosfilt(sos, nz) * ac.noise_level
        phases = rng.uniform(0, 2 * np.pi, len(ac.tones))
        for path_rx, q in ((rx, 1.0), (img, ground_q)):
            te, R = _retarded(ac, o, path_rx, t_obs, hr, c_fn)
            amp = 400.0 / np.maximum(R, 50.0)
            # slow engine-power wobble on amplitude only
            wob = 1 + 0.15 * np.sin(2 * np.pi * 0.05 * te)
            sig = np.interp(te, tn, nz) * 0.6
            for (f, a), ph in zip(ac.tones, phases):
                absorb = 10 ** (-iso9613_absorption_db_per_m(f, 288.15, 60) * R / 20)
                sig = sig + a * absorb * np.sin(2 * np.pi * f * te + ph)
            x += q * amp * wob * sig
    # mains hum (slightly off-nominal grid frequency) + harmonics
    f_mains = 50.03
    for k, a in zip(range(1, 12), [1, .6, .5, .3, .25, .2, .12, .1, .08, .06, .05]):
        x += hum_level * a * np.sin(2 * np.pi * k * f_mains * (t_obs - t_obs[0]) + k)
    # background: pink-ish noise + wind rumble
    w = rng.normal(0, 1, len(x))
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    x += background * signal.lfilter(b, a, w) * 10
    x += 0.01 * signal.sosfilt(signal.butter(2, 30, fs=sr, output="sos"), rng.normal(0, 1, len(x))) * 10
    x = x / (np.max(np.abs(x)) * 1.1) * snr_scale
    nominal_start = sc["t_rec0"] + clock_error   # what the (wrong) phone clock says
    return x, sr, nominal_start
