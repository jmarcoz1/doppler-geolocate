import json
from pathlib import Path

import numpy as np

from dopplerloc import tracks
from dopplerloc.sources import BBox
from dopplerloc.sources.opensky import OpenSkyTrino, parse_states
from dopplerloc.sources.readsb import parse_v2
from conftest import straight_track_df

FIX = Path(__file__).parent / "fixtures"


def test_parse_opensky_fixture():
    df = tracks.normalize(parse_states(json.loads((FIX / "opensky_states_valencia.json").read_text())))
    assert len(df) == 6
    r = df[df.callsign == "VLG2AM"].iloc[0]
    assert r.icao24 == "3444d0" and np.isclose(r.lat, 39.5222) and np.isclose(r.alt_baro, 403.86)


def test_parse_adsblol_fixture():
    df = tracks.normalize(parse_v2(json.loads((FIX / "adsblol_point_valencia.json").read_text()), "adsblol"))
    assert len(df) > 0
    assert df["alt_baro"].dropna().between(-500, 15000).all()   # ft -> m
    assert df["gs"].dropna().between(0, 350).all()               # kt -> m/s


def test_smoothing_spline_velocity_and_resample(tmp_path):
    df = straight_track_df((0.0, 0.0, 1000.0), 45.0, 120.0, vs=-3.0, period=5.0)
    rng = np.random.default_rng(0)
    df["lat"] += rng.normal(0, 3e-5, len(df))   # ~3 m noise
    m = tracks.build_model(df)
    t = df["time"].to_numpy()[10:-10]
    speed = np.linalg.norm(m.velocity(t), axis=1)
    assert np.allclose(speed, np.hypot(120.0, 3.0), atol=1.0)
    u = tracks.resample_uniform(df, dt=1.0)
    assert np.allclose(np.diff(u["time"]), 1.0)
    assert np.allclose(u["gs"].iloc[20:-20], 120.0, atol=1.0)
    p = tracks.save_tracks(u, tmp_path / "x.parquet")
    assert len(tracks.load_tracks(p)) == len(u)


def test_trino_sql():
    q = OpenSkyTrino.sql(BBox.around(39.49, -0.48, 20), 1_700_000_000, 1_700_000_600)
    assert "state_vectors_data4" in q and "hour BETWEEN" in q
