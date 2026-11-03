"""readsb/tar1090 "v2" style community APIs (adsb.lol, airplanes.live, ADS-B Exchange).

All expose `/v2/point/{lat}/{lon}/{radius_nm}` returning {"ac": [...], "now": ms}
with readsb field names (alt_baro/alt_geom in ft, gs in kt, baro_rate/geom_rate
in ft/min, seen_pos = seconds since the position was received).

  * adsb.lol      https://api.adsb.lol          free, no key (ODbL data); live only.
                  Daily full-history dumps: github.com/adsblol/globe_history_* (offline use).
  * airplanes.live https://api.airplanes.live   live only; currently returns 403 to
                  unregistered clients - email them for access; 1 req/s.
  * ADS-B Exchange via RapidAPI (paid key: ADSBX_RAPIDAPI_KEY). Historical "trace"
                  files require their commercial/enterprise products.
These are *live* snapshots: use `poll()` while you record to build tracks.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

from .base import BBox, DataSource, SourceError
from ..tracks import FT, KT, FPM

PRESETS = {
    "adsblol": dict(base="https://api.adsb.lol/v2/point/{lat}/{lon}/{r}", headers={}, interval=2.0),
    "airplaneslive": dict(base="https://api.airplanes.live/v2/point/{lat}/{lon}/{r}", headers={}, interval=1.0),
    "adsbx": dict(base="https://adsbexchange-com1.p.rapidapi.com/v2/lat/{lat}/lon/{lon}/dist/{r}/",
                  headers={"X-RapidAPI-Host": "adsbexchange-com1.p.rapidapi.com"},
                  key_env="ADSBX_RAPIDAPI_KEY", key_header="X-RapidAPI-Key", interval=1.0),
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def parse_v2(js: dict, source="readsb") -> pd.DataFrame:
    now = float(js.get("now", 0)) / (1000.0 if js.get("now", 0) > 1e11 else 1.0)
    recs = []
    for a in js.get("ac", []):
        if "lat" not in a or "lon" not in a:
            continue
        ground = a.get("alt_baro") == "ground"
        recs.append(dict(
            time=now - _num(a.get("seen_pos", 0) or 0), icao24=a.get("hex", "").lstrip("~"),
            callsign=(a.get("flight") or "").strip(), lat=_num(a["lat"]), lon=_num(a["lon"]),
            alt_baro=np.nan if ground else _num(a.get("alt_baro")) * FT,
            alt_geom=_num(a.get("alt_geom")) * FT, gs=_num(a.get("gs")) * KT, track=_num(a.get("track")),
            vrate=_num(a.get("geom_rate", a.get("baro_rate"))) * FPM, on_ground=ground, source=source))
    return pd.DataFrame(recs)


class ReadsbV2(DataSource):
    name = "readsb"

    def __init__(self, base: str, headers: dict | None = None, interval: float = 2.0, **kw):
        super().__init__(**kw)
        self.base, self.min_poll_interval = base, interval
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "dopplerloc/0.1 (research)", **(headers or {})})

    @classmethod
    def preset(cls, name, **kw):
        p = dict(PRESETS[name])
        headers = dict(p["headers"])
        if "key_env" in p:
            key = os.environ.get(p["key_env"])
            if not key:
                raise SourceError(f"{name}: set {p['key_env']}")
            headers[p["key_header"]] = key
        obj = cls(p["base"], headers, p["interval"], **kw)
        obj.name = name
        return obj

    def snapshot(self, bbox: BBox) -> pd.DataFrame:
        lat, lon = bbox.center
        url = self.base.format(lat=round(lat, 5), lon=round(lon, 5), r=min(250, int(np.ceil(bbox.radius_nm))))
        try:
            r = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as e:
            raise SourceError(f"{self.name}: {e}") from e
        if r.status_code != 200:
            raise SourceError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}")
        df = parse_v2(r.json(), self.name)
        return df[bbox.contains(df["lat"], df["lon"])] if len(df) else df
