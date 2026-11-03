"""WGS84 geodesy: geodetic <-> ECEF <-> local ENU, implemented in numpy.

pyproj is used (if installed) only in the tests as an independent cross-check.
Heights are *ellipsoidal* (WGS84) heights. ADS-B geometric altitude is
nominally HAE (height above ellipsoid) while barometric altitude is pressure
altitude; terrain/observer heights from maps are usually orthometric (above
geoid, ~ +50 m offset near Valencia: N ~= 50 m). Use `geoid_undulation` to
convert if you care about tens of metres vertically.
"""
from __future__ import annotations

import numpy as np

A = 6378137.0                    # semi-major axis [m]
F = 1.0 / 298.257223563          # flattening
B = A * (1.0 - F)                # semi-minor axis
E2 = F * (2.0 - F)               # first eccentricity squared
EP2 = E2 / (1.0 - E2)            # second eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, h):
    """Geodetic (deg, deg, m ellipsoidal) -> ECEF (m). Broadcasts."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    h = np.asarray(h, dtype=float)
    s, c = np.sin(lat), np.cos(lat)
    N = A / np.sqrt(1.0 - E2 * s * s)
    x = (N + h) * c * np.cos(lon)
    y = (N + h) * c * np.sin(lon)
    z = (N * (1.0 - E2) + h) * s
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(xyz):
    """ECEF -> geodetic using Bowring's method + 3 Newton-ish refinements."""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    theta = np.arctan2(z * A, p * B)
    lat = np.arctan2(z + EP2 * B * np.sin(theta) ** 3, p - E2 * A * np.cos(theta) ** 3)
    for _ in range(3):
        N = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - E2 * N / (N + h)))
    N = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h


def enu_matrix(lat0_deg, lon0_deg):
    """Rotation matrix R so that enu = R @ (ecef - ecef0)."""
    la, lo = np.radians(lat0_deg), np.radians(lon0_deg)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def ecef_to_enu(xyz, lat0, lon0, h0):
    o = geodetic_to_ecef(lat0, lon0, h0)
    R = enu_matrix(lat0, lon0)
    return (np.asarray(xyz) - o) @ R.T


def enu_to_ecef(enu, lat0, lon0, h0):
    o = geodetic_to_ecef(lat0, lon0, h0)
    R = enu_matrix(lat0, lon0)
    return np.asarray(enu) @ R + o


def geodetic_to_enu(lat, lon, h, lat0, lon0, h0):
    return ecef_to_enu(geodetic_to_ecef(lat, lon, h), lat0, lon0, h0)


def enu_to_geodetic(enu, lat0, lon0, h0):
    return ecef_to_geodetic(enu_to_ecef(enu, lat0, lon0, h0))


def enu_velocity_to_ecef(v_enu, lat_deg, lon_deg):
    """Rotate a local ENU velocity (at each point) into ECEF."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    ve, vn, vu = v_enu[..., 0], v_enu[..., 1], v_enu[..., 2]
    vx = -so * ve - sl * co * vn + cl * co * vu
    vy = co * ve - sl * so * vn + cl * so * vu
    vz = cl * vn + sl * vu
    return np.stack([vx, vy, vz], axis=-1)


def haversine_m(lat1, lon1, lat2, lon2, r=6371008.8):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def offset_latlon(lat, lon, east_m, north_m):
    """Small-offset local move (good to <1 m for offsets of tens of km)."""
    lat_r = np.radians(lat)
    N = A / np.sqrt(1 - E2 * np.sin(lat_r) ** 2)
    M = A * (1 - E2) / (1 - E2 * np.sin(lat_r) ** 2) ** 1.5
    return lat + np.degrees(north_m / M), lon + np.degrees(east_m / (N * np.cos(lat_r)))


def destination_point(lat, lon, bearing_deg, dist_m):
    """Spherical great-circle destination (fine for building synthetic tracks)."""
    R = 6371008.8
    br, d = np.radians(bearing_deg), dist_m / R
    la1, lo1 = np.radians(lat), np.radians(lon)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(br))
    lo2 = lo1 + np.arctan2(np.sin(br) * np.sin(d) * np.cos(la1), np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(la2), np.degrees(lo2)


def geoid_undulation(lat, lon):
    """Very rough EGM96 geoid undulation for the Valencia region (~49-51 m).

    Replace with pyproj/EGM2008 grids for real work. Orthometric H = h - N.
    """
    return 50.0 + 0.0 * np.asarray(lat)
