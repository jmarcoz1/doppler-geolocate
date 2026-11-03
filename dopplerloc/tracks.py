"""Aircraft track containers, cleaning, uniform-grid interpolation and disk cache.

Canonical long-format columns (SI units, unix seconds):
  time, icao24, callsign, lat, lon, alt_geom, alt_baro, gs, track, vrate, on_ground, source
  alt_* [m], gs [m/s], track [deg true], vrate [m/s]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline, make_smoothing_spline

from . import geodesy

COLUMNS = ["time", "icao24", "callsign", "lat", "lon", "alt_geom", "alt_baro",
           "gs", "track", "vrate", "on_ground", "source"]

FT = 0.3048
KT = 0.514444
FPM = 0.3048 / 60.0


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}).astype(
        {"icao24": "object", "callsign": "object", "on_ground": "bool", "source": "object"})


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    df["icao24"] = df["icao24"].astype(str).str.lower().str.strip()
    df["callsign"] = df["callsign"].fillna("").astype(str).str.strip()
    df["on_ground"] = df["on_ground"].fillna(False).astype(bool)
    df = df.dropna(subset=["time", "lat", "lon"])
    df = df.drop_duplicates(subset=["icao24", "time"]).sort_values(["icao24", "time"])
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- cache
def save_tracks(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:  # pyarrow missing -> csv fallback
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def load_tracks(path: str | os.PathLike) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return normalize(pd.read_parquet(path))
    return normalize(pd.read_csv(path, dtype={"icao24": str, "callsign": str}))


# ----------------------------------------------------------------------------- model
@dataclass
class TrackModel:
    """Continuous-time ECEF trajectory of one aircraft (position + velocity)."""
    icao24: str
    callsign: str
    t0: float
    t1: float
    _pos: object = field(repr=False)
    _vel: object = field(repr=False)
    altitude_kind: str = "geom"

    def position(self, t):
        t = np.clip(np.asarray(t, dtype=float), self.t0, self.t1)
        return np.stack([s(t) for s in self._pos], axis=-1)

    def velocity(self, t):
        t = np.clip(np.asarray(t, dtype=float), self.t0, self.t1)
        return np.stack([s(t) for s in self._vel], axis=-1)

    def geodetic(self, t):
        return geodesy.ecef_to_geodetic(self.position(t))


def _altitude(g: pd.DataFrame, prefer="geom"):
    geo, baro = g["alt_geom"].to_numpy(float), g["alt_baro"].to_numpy(float)
    if prefer == "geom":
        alt = np.where(np.isfinite(geo), geo, baro)
        kind = "geom" if np.isfinite(geo).any() else "baro"
    else:
        alt = np.where(np.isfinite(baro), baro, geo)
        kind = "baro"
    return alt, kind


def build_model(g: pd.DataFrame, method: str = "smooth", prefer_alt: str = "geom",
                geoid_n: float = 0.0, lam: float | None = None) -> TrackModel | None:
    """Fit a continuous trajectory to one aircraft's samples.

    method="smooth": GCV smoothing splines on ECEF x/y/z (robust to 25 ft
        altitude quantisation & position noise); velocity = spline derivative.
    method="hermite": cubic Hermite through positions using the *reported*
        ADS-B velocities (gs/track/vrate) as derivatives.
    geoid_n: added to altitude to turn orthometric/MSL into ellipsoidal height.
    """
    g = g.sort_values("time").drop_duplicates("time")
    alt, kind = _altitude(g, prefer_alt)
    ok = np.isfinite(alt) & np.isfinite(g["lat"].to_numpy(float))
    g, alt = g[ok], alt[ok]
    if len(g) < 4:
        return None
    t = g["time"].to_numpy(float)
    xyz = geodesy.geodetic_to_ecef(g["lat"].to_numpy(float), g["lon"].to_numpy(float), alt + geoid_n)
    if method == "hermite" and g[["gs", "track"]].notna().all().all():
        trk = np.radians(g["track"].to_numpy(float))
        gs = g["gs"].to_numpy(float)
        vr = g["vrate"].fillna(0.0).to_numpy(float)
        venu = np.stack([gs * np.sin(trk), gs * np.cos(trk), vr], -1)
        vxyz = geodesy.enu_velocity_to_ecef(venu, g["lat"].to_numpy(float), g["lon"].to_numpy(float))
        pos = [CubicHermiteSpline(t, xyz[:, k], vxyz[:, k]) for k in range(3)]
    else:
        pos = [make_smoothing_spline(t, xyz[:, k], lam=lam) for k in range(3)]
    vel = [p.derivative() for p in pos]
    return TrackModel(str(g["icao24"].iloc[0]), str(g["callsign"].iloc[0]), float(t[0]), float(t[-1]),
                      pos, vel, kind)


def build_models(df: pd.DataFrame, t_min=None, t_max=None, airborne_only=True, **kw) -> list[TrackModel]:
    models = []
    for _, g in df.groupby("icao24"):
        if airborne_only:
            g = g[~g["on_ground"].astype(bool)]
        if t_min is not None:
            g = g[g["time"] >= t_min]
        if t_max is not None:
            g = g[g["time"] <= t_max]
        m = build_model(g, **kw)
        if m is not None:
            models.append(m)
    return models


def resample_uniform(df: pd.DataFrame, dt: float = 1.0, **kw) -> pd.DataFrame:
    """Interpolate every aircraft to a uniform time grid (step dt seconds)."""
    out = []
    for m in build_models(df, **kw):
        t = np.arange(np.ceil(m.t0 / dt) * dt, m.t1 + 1e-9, dt)
        if len(t) == 0:
            continue
        lat, lon, h = m.geodetic(t)
        v = m.velocity(t)
        R = np.stack([geodesy.enu_matrix(a, b) for a, b in zip(lat, lon)])
        venu = np.einsum("nij,nj->ni", R, v)
        out.append(pd.DataFrame({
            "time": t, "icao24": m.icao24, "callsign": m.callsign, "lat": lat, "lon": lon,
            "alt_geom": h, "alt_baro": np.nan, "gs": np.hypot(venu[:, 0], venu[:, 1]),
            "track": np.degrees(np.arctan2(venu[:, 0], venu[:, 1])) % 360, "vrate": venu[:, 2],
            "on_ground": False, "source": "interp"}))
    return normalize(pd.concat(out)) if out else empty_frame()
