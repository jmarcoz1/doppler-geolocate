import numpy as np
import pytest

from dopplerloc import atmosphere, geodesy, tracks
from dopplerloc.physics import Observer, Propagation, doppler_factor, emission_time, lloyd_delay, lloyd_notches, predict_fobs
from conftest import LAT0, LON0, H0, straight_track_df

C = 340.0


def test_isa_speed_of_sound():
    assert np.isclose(atmosphere.sound_speed_at(0.0), 340.29, atol=0.01)
    assert np.isclose(atmosphere.sound_speed_at(11000.0), 295.07, atol=0.05)
    assert np.isclose(atmosphere.effective_sound_speed(500.0, 500.0), atmosphere.sound_speed_at(500.0))
    ce = atmosphere.effective_sound_speed(0.0, 10000.0)
    assert atmosphere.sound_speed_at(10000.0) < ce < atmosphere.sound_speed_at(0.0)


def test_iso9613_magnitude():
    a = atmosphere.iso9613_absorption_db_per_m(1000.0, 293.15, 70.0) * 1000  # dB/km
    assert 4.0 < a < 6.0          # ISO 9613-1 table: ~4.7-5 dB/km
    a4 = atmosphere.iso9613_absorption_db_per_m(4000.0, 293.15, 70.0) * 1000
    assert a4 > 3 * a


def test_doppler_factor_signs():
    assert np.isclose(doppler_factor(-100.0, C), C / (C - 100.0))  # approaching -> higher
    assert np.isclose(doppler_factor(+100.0, C), C / (C + 100.0))  # receding -> lower


@pytest.fixture
def flyover():
    # straight & level, 1000 m up, 100 m/s eastwards, passing directly over the origin
    df = straight_track_df((0.0, 0.0, 1000.0), 90.0, 100.0, dur=200.0, period=0.5)
    m = tracks.build_model(df, method="hermite")
    return m, df["time"].iloc[0] + 100.0


def test_retarded_time_analytic(flyover):
    m, t_cpa = flyover
    obs = Observer(LAT0, LON0, H0)
    prop = Propagation(fixed_c=C)
    t_obs = t_cpa + np.array([-60.0, -10.0, 0.0, 3.0, 40.0])
    te, g = emission_time(m, obs, t_obs, prop)
    # analytic: x = v tau, z = 1000 (flat-earth approx valid to <1e-3 here)
    v, d = 100.0, 1000.0
    T = t_obs - t_cpa
    # (T - tau)^2 c^2 = v^2 tau^2 + d^2  -> quadratic in tau, physical root tau < T
    a = C ** 2 - v ** 2
    b = -2 * C ** 2 * T
    cc = C ** 2 * T ** 2 - d ** 2
    tau = (-b - np.sqrt(b * b - 4 * a * cc)) / (2 * a)
    assert np.allclose(te - t_cpa, tau, atol=5e-3)
    # residual of the defining equation
    assert np.allclose(te + g["R"] / g["c"], t_obs, atol=1e-6)


def test_doppler_s_curve(flyover):
    m, t_cpa = flyover
    obs = Observer(LAT0, LON0, H0)
    prop = Propagation(fixed_c=C)
    f, te, g = predict_fobs(m, obs, t_cpa + np.array([-90.0, 1000.0 / C, 90.0]), 1000.0, prop)
    # far before: nearly radial approach -> ~f c/(c-v); heard-at-CPA (t_cpa + d/c) ~ f_src
    assert f[0] > 1000.0 * C / (C - 100.0) * 0.98
    assert abs(f[1] - 1000.0) < 1.0
    assert f[2] < 1000.0 * C / (C + 100.0) * 1.02


def test_wind_changes_propagation_speed(flyover):
    m, t_cpa = flyover
    obs = Observer(LAT0, LON0, H0)
    t = np.array([t_cpa + 30.0])           # source is east of the observer
    te0, _ = emission_time(m, obs, t, Propagation(fixed_c=C))
    # wind towards the east = blowing from observer to source: sound towards us is slower
    te1, g1 = emission_time(m, obs, t, Propagation(fixed_c=C, wind_enu=(20.0, 0.0)))
    assert g1["c"][0] < C
    assert te1[0] < te0[0]


def test_lloyd_overhead_and_notches():
    obs = Observer(LAT0, LON0, H0)
    src = geodesy.enu_to_ecef(np.array([0.0, 0.0, 1000.0]), LAT0, LON0, H0)
    tau = lloyd_delay(src, obs, 1.5, C)
    assert np.isclose(tau, 2 * 1.5 / C, rtol=1e-3)
    f = lloyd_notches(tau, 1000.0)[0]
    f = f[np.isfinite(f)]
    assert np.isclose(f[0], 0.5 / tau, rtol=1e-6) and np.allclose(np.diff(f), 1 / tau)
    src2 = geodesy.enu_to_ecef(np.array([5000.0, 0.0, 500.0]), LAT0, LON0, H0)
    assert lloyd_delay(src2, obs, 1.5, C) < tau / 5   # low elevation -> shorter delay
