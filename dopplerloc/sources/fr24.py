"""Stub adapter for the *official, paid* Flightradar24 API (https://fr24api.flightradar24.com).

We deliberately do NOT scrape flightradar24.com (its Terms of Service forbid it).
Requires a subscription token in FR24_API_TOKEN. Endpoint names/fields follow the
public FR24 API docs at the time of writing - untested here (no key); verify
against https://fr24api.flightradar24.com/docs before relying on it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

from .base import BBox, DataSource, SourceError
from ..tracks import FT, KT, FPM

BASE = "https://fr24api.flightradar24.com/api"


class Flightradar24Api(DataSource):
    name = "fr24"
    supports_history = True

    def __init__(self, token=None, **kw):
        super().__init__(**kw)
        self.token = token or os.environ.get("FR24_API_TOKEN")

    def _get(self, path, params):
        if not self.token:
            raise SourceError("fr24: set FR24_API_TOKEN (official paid API). Scraping FR24 is not supported.")
        r = requests.get(BASE + path, params=params, timeout=self.timeout, headers={
            "Accept": "application/json", "Accept-Version": "v1", "Authorization": f"Bearer {self.token}"})
        if r.status_code != 200:
            raise SourceError(f"fr24 HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    @staticmethod
    def _bounds(b: BBox):
        return f"{b.lat_max:.3f},{b.lat_min:.3f},{b.lon_min:.3f},{b.lon_max:.3f}"  # N,S,W,E

    @staticmethod
    def parse(js) -> pd.DataFrame:
        recs = []
        for a in js.get("data", []):
            ts = pd.Timestamp(a.get("timestamp")).timestamp() if a.get("timestamp") else np.nan
            recs.append(dict(time=ts, icao24=(a.get("hex") or "").lower(), callsign=a.get("callsign", ""),
                             lat=a.get("lat"), lon=a.get("lon"), alt_baro=(a.get("alt") or np.nan) * FT,
                             gs=(a.get("gspeed") or np.nan) * KT, track=a.get("track"),
                             vrate=(a.get("vspeed") or np.nan) * FPM, source="fr24"))
        return pd.DataFrame(recs)

    def snapshot(self, bbox: BBox) -> pd.DataFrame:
        return self.parse(self._get("/live/flight-positions/full", {"bounds": self._bounds(bbox)}))

    def history(self, bbox: BBox, t0: float, t1: float, step: float = 10.0) -> pd.DataFrame:
        frames = [self.parse(self._get("/historic/flight-positions/full",
                                       {"bounds": self._bounds(bbox), "timestamp": int(t)}))
                  for t in np.arange(t0, t1 + 1e-9, step)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
