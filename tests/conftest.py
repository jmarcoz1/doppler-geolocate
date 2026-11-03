import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dopplerloc import geodesy, tracks  # noqa: E402

LAT0, LON0, H0 = 39.4893, -0.4816, 119.0


def straight_track_df(p0_enu, heading_deg, speed, vs=0.0, t0=1_700_000_000.0, dur=300.0, period=1.0,
                      icao="abc123", cs="TEST1"):
    t = np.arange(t0, t0 + dur, period)
    hd = np.radians(heading_deg)
    tau = t - (t0 + dur / 2)
    p = np.stack([p0_enu[0] + speed * np.sin(hd) * tau, p0_enu[1] + speed * np.cos(hd) * tau,
                  p0_enu[2] + vs * tau], -1)
    lat, lon, h = geodesy.enu_to_geodetic(p, LAT0, LON0, H0)
    return tracks.normalize(pd.DataFrame(dict(time=t, icao24=icao, callsign=cs, lat=lat, lon=lon, alt_geom=h,
                                              gs=speed, track=heading_deg, vrate=vs, on_ground=False,
                                              source="test")))


@pytest.fixture
def origin():
    return LAT0, LON0, H0
