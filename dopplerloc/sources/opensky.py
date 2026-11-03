"""OpenSky Network adapters.

REST (https://opensky-network.org/api, docs: openskynetwork.github.io/opensky-api/rest.html)
  * anonymous: only the *current* state vectors (`time` ignored), 10 s resolution,
    400 credits/day; a bbox <= 25 sq deg costs 1 credit per call.
  * authenticated (OAuth2 client-credentials, mandatory since 2026-03-18; basic
    auth no longer works): state vectors up to 1 h in the past, 5 s resolution,
    4000 credits/day (8000 for active feeders).
  Set OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET to authenticate.

Trino (historical database, `state_vectors_data4`, ~1 s samples, years back)
  requires an OpenSky account *approved for research/non-commercial historical
  access*. We use `pyopensky` (pip install pyopensky) if available.
"""
from __future__ import annotations

import logging
import os
import time as _time

import numpy as np
import pandas as pd
import requests

from .base import BBox, DataSource, SourceError

log = logging.getLogger(__name__)

API = "https://opensky-network.org/api"
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


def parse_states(js: dict) -> pd.DataFrame:
    """Parse an /states/all JSON payload into the canonical frame."""
    rows = js.get("states") or []
    t_resp = js.get("time")
    recs = []
    for s in rows:
        t = s[3] if s[3] is not None else (s[4] if s[4] is not None else t_resp)
        recs.append(dict(time=float(t), icao24=s[0], callsign=(s[1] or "").strip(),
                         lon=s[5], lat=s[6], alt_baro=s[7], on_ground=bool(s[8]), gs=s[9],
                         track=s[10], vrate=s[11], alt_geom=s[13], source="opensky"))
    df = pd.DataFrame(recs)
    if df.empty:
        from .. import tracks
        return tracks.empty_frame()
    return df.astype({c: float for c in ["lat", "lon", "alt_baro", "alt_geom", "gs", "track", "vrate"]})


class OpenSkyRest(DataSource):
    name = "opensky"
    supports_history = True  # last hour only, authenticated

    def __init__(self, client_id=None, client_secret=None, **kw):
        super().__init__(**kw)
        self.client_id = client_id or os.environ.get("OPENSKY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("OPENSKY_CLIENT_SECRET")
        self._token, self._token_exp = None, 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "dopplerloc/0.1 (research)"
        self.min_poll_interval = 5.0 if self.authenticated else 10.0
        self.remaining_credits = None

    @property
    def authenticated(self):
        return bool(self.client_id and self.client_secret)

    def _headers(self):
        if not self.authenticated:
            return {}
        if _time.time() > self._token_exp - 60:
            r = self.session.post(TOKEN_URL, data=dict(grant_type="client_credentials",
                                  client_id=self.client_id, client_secret=self.client_secret),
                                  timeout=self.timeout)
            if r.status_code != 200:
                raise SourceError(f"OpenSky token request failed: {r.status_code} {r.text[:200]}")
            js = r.json()
            self._token = js["access_token"]
            self._token_exp = _time.time() + float(js.get("expires_in", 1800))
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, path, params):
        try:
            r = self.session.get(API + path, params=params, headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as e:
            raise SourceError(f"OpenSky request failed: {e}") from e
        self.remaining_credits = r.headers.get("X-Rate-Limit-Remaining", self.remaining_credits)
        if r.status_code == 429:
            raise SourceError("OpenSky rate limit / credits exhausted (429). Retry-After: "
                              f"{r.headers.get('X-Rate-Limit-Retry-After-Seconds')}")
        if r.status_code != 200:
            raise SourceError(f"OpenSky HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def snapshot(self, bbox: BBox, t: float | None = None) -> pd.DataFrame:
        params = dict(lamin=bbox.lat_min, lomin=bbox.lon_min, lamax=bbox.lat_max, lomax=bbox.lon_max)
        if t is not None:
            params["time"] = int(t)
        return parse_states(self._get("/states/all", params))

    def history(self, bbox: BBox, t0: float, t1: float, step: float = 10.0) -> pd.DataFrame:
        if not self.authenticated:
            raise SourceError("OpenSky anonymous access returns only *current* states. Use `poll` "
                              "while recording, authenticate (last 1 h), or use opensky-trino.")
        if t0 < _time.time() - 3600:
            raise SourceError("OpenSky REST keeps only the last hour; use the Trino source.")
        frames = [self.snapshot(bbox, t) for t in np.arange(t0, t1 + 1e-9, max(step, 5.0))]
        return pd.concat(frames, ignore_index=True)

    def track(self, icao24: str, t: float = 0) -> pd.DataFrame:
        """/tracks/all (experimental; sparse waypoints, no speed)."""
        js = self._get("/tracks/all", dict(icao24=icao24.lower(), time=int(t)))
        recs = [dict(time=p[0], lat=p[1], lon=p[2], alt_baro=p[3], track=p[4], on_ground=p[5],
                     icao24=js.get("icao24"), callsign=(js.get("callsign") or "").strip(), source="opensky-track")
                for p in js.get("path", [])]
        return pd.DataFrame(recs)


class OpenSkyTrino(DataSource):
    """Historical state vectors via the OpenSky Trino database (approved accounts only)."""
    name = "opensky-trino"
    supports_history = True

    @staticmethod
    def sql(bbox: BBox, t0: float, t1: float) -> str:
        """Raw SQL you can run yourself (Trino CLI / pyopensky)."""
        h0, h1 = int(t0 // 3600 * 3600), int(t1 // 3600 * 3600)
        return (
            "SELECT time, icao24, callsign, lat, lon, geoaltitude, baroaltitude, velocity, heading, "
            "vertrate, onground FROM minio.osky.state_vectors_data4 "
            f"WHERE hour BETWEEN {h0} AND {h1} AND time BETWEEN {int(t0)} AND {int(t1)} "
            f"AND lat BETWEEN {bbox.lat_min} AND {bbox.lat_max} "
            f"AND lon BETWEEN {bbox.lon_min} AND {bbox.lon_max} ORDER BY icao24, time")

    def history(self, bbox: BBox, t0: float, t1: float) -> pd.DataFrame:
        try:
            from pyopensky.trino import Trino  # type: ignore
        except ImportError as e:
            raise SourceError("pip install pyopensky (and configure credentials; needs an OpenSky "
                              "account approved for historical data). SQL:\n" + self.sql(bbox, t0, t1)) from e
        df = Trino().history(pd.Timestamp(t0, unit="s", tz="UTC"), pd.Timestamp(t1, unit="s", tz="UTC"),
                             bounds=(bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max))
        if df is None or len(df) == 0:
            from .. import tracks
            return tracks.empty_frame()
        t = df["time"]
        t = (pd.to_datetime(t, utc=True).astype("int64") / 1e9) if not np.issubdtype(t.dtype, np.number) else t
        return pd.DataFrame(dict(
            time=t.astype(float), icao24=df["icao24"], callsign=df.get("callsign"),
            lat=df["lat"], lon=df["lon"], alt_geom=df.get("geoaltitude"), alt_baro=df.get("baroaltitude"),
            gs=df.get("velocity"), track=df.get("heading"), vrate=df.get("vertrate"),
            on_ground=df.get("onground", False), source="opensky-trino"))
