"""Forward-model inversion without audio - fast regression test of the solver."""
import numpy as np
import pandas as pd

from dopplerloc import geodesy, locate, tracks
from dopplerloc.physics import Observer, Propagation, predict_fobs
from dopplerloc.ridge import Ridge
from conftest import LAT0, LON0, H0, straight_track_df


def test_locate_two_tracks():
    t0 = 1_700_000_000.0
    a = straight_track_df((0.0, 0.0, 600.0), 124.0, 75.0, vs=-3.0, t0=t0, dur=400, period=5.0, icao="aaa111", cs="A")
    b = straight_track_df((1500.0, -2500.0, 1800.0), 35.0, 150.0, vs=5.0, t0=t0 + 40, dur=400, period=5.0,
                          icao="bbb222", cs="B")
    models = tracks.build_models(tracks.normalize(pd.concat([a, b])))
    la, lo = geodesy.offset_latlon(LAT0, LON0, -700.0, -500.0)
    obs = Observer(float(la), float(lo), H0 + 1.5)
    prop = Propagation()
    t_rec = t0 + 140.0
    rng = np.random.default_rng(1)
    ridges = []
    for m, f0, (s, e) in ((models[0], 800.0, (20, 110)), (models[1], 500.0, (60, 150))):
        tt = np.arange(s, e, 0.25)
        f, _, _ = predict_fobs(m, obs, t_rec + tt, f0, prop)
        ridges.append(Ridge(tt, f + rng.normal(0, 0.1, len(tt)), np.full(len(tt), 20.0), np.ones(len(tt)),
                            np.full(len(tt), 0.1), f"R{len(ridges)}"))
    nominal = t_rec + 0.8   # clock error: true start = nominal - 0.8 s
    res = locate.locate(ridges, models, nominal, (LAT0, LON0), half_width_m=4000, step_m=500, h_obs=obs.h,
                        dt_sigma=2.0, prop=prop, verbose=False)
    err = geodesy.haversine_m(res.lat, res.lon, obs.lat, obs.lon)
    assert err < 15.0, err
    assert abs(res.dt + 0.8) < 0.1
    assert np.allclose(sorted(res.f_src), [500.0, 800.0], atol=0.5)
