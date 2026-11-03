import numpy as np
import pytest

from dopplerloc import geodesy


def test_roundtrip_ecef():
    lat = np.array([39.4893, -33.9, 89.9, 0.0, 51.5])
    lon = np.array([-0.4816, 151.2, 10.0, 0.0, -0.12])
    h = np.array([119.0, 10.0, 3000.0, -50.0, 12000.0])
    la, lo, hh = geodesy.ecef_to_geodetic(geodesy.geodetic_to_ecef(lat, lon, h))
    assert np.allclose(la, lat, atol=1e-10)
    assert np.allclose(lo, lon, atol=1e-10)
    assert np.allclose(hh, h, atol=1e-5)


def test_known_ecef_equator():
    assert np.allclose(geodesy.geodetic_to_ecef(0, 0, 0), [geodesy.A, 0, 0])
    assert np.allclose(geodesy.geodetic_to_ecef(90, 0, 0), [0, 0, geodesy.B], atol=1e-6)


def test_against_pyproj():
    pyproj = pytest.importorskip("pyproj")
    tr = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    lat, lon, h = 39.4893, -0.4816, 1234.5
    x, y, z = tr.transform(lon, lat, h)
    assert np.allclose(geodesy.geodetic_to_ecef(lat, lon, h), [x, y, z], atol=1e-6)


def test_enu_roundtrip_and_axes():
    lat0, lon0, h0 = 39.4893, -0.4816, 100.0
    enu = np.array([[1000.0, 0, 0], [0, 1000.0, 0], [0, 0, 1000.0], [-5000.0, 3000.0, 400.0]])
    la, lo, h = geodesy.enu_to_geodetic(enu, lat0, lon0, h0)
    back = geodesy.geodetic_to_enu(la, lo, h, lat0, lon0, h0)
    assert np.allclose(back, enu, atol=1e-6)
    assert lo[0] > lon0 and abs(la[0] - lat0) < 1e-4          # east
    assert la[1] > lat0 and abs(lo[1] - lon0) < 1e-9          # north
    assert np.isclose(h[2], h0 + 1000.0, atol=1e-6)            # up


def test_slant_range_curvature():
    # 50 km away at same ellipsoidal height: ENU "up" is negative by ~ d^2/(2R) ~ 196 m
    lat0, lon0 = 39.5, -0.5
    la, lo = geodesy.offset_latlon(lat0, lon0, 0.0, 50000.0)
    enu = geodesy.geodetic_to_enu(la, lo, 0.0, lat0, lon0, 0.0)
    assert -200 < enu[2] < -190
    assert np.isclose(np.hypot(enu[0], enu[1]), 50000.0, rtol=2e-3)


def test_haversine_vs_offset():
    la, lo = geodesy.offset_latlon(39.5, -0.5, 3000.0, 4000.0)
    assert np.isclose(geodesy.haversine_m(39.5, -0.5, la, lo), 5000.0, rtol=3e-3)
