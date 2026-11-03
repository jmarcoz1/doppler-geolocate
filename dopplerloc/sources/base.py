from __future__ import annotations

import hashlib
import json
import logging
import math
import time as _time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import tracks

log = logging.getLogger(__name__)


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    @classmethod
    def around(cls, lat, lon, radius_km):
        dlat = radius_km / 111.32
        dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
        return cls(lat - dlat, lon - dlon, lat + dlat, lon + dlon)

    @property
    def center(self):
        return (self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2

    @property
    def radius_nm(self):
        lat, _ = self.center
        dy = (self.lat_max - self.lat_min) / 2 * 111.32
        dx = (self.lon_max - self.lon_min) / 2 * 111.32 * math.cos(math.radians(lat))
        return math.hypot(dx, dy) / 1.852

    def contains(self, lat, lon):
        return (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)


class DataSource:
    """Base class. Subclasses implement `snapshot` (live) and/or `history`."""
    name = "base"
    supports_history = False
    min_poll_interval = 10.0

    def __init__(self, cache_dir: str | Path = "data/cache", timeout: float = 30.0):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    # -- to implement --------------------------------------------------------
    def snapshot(self, bbox: BBox) -> pd.DataFrame:  # pragma: no cover
        raise SourceError(f"{self.name}: live snapshots not supported")

    def history(self, bbox: BBox, t0: float, t1: float) -> pd.DataFrame:  # pragma: no cover
        raise SourceError(f"{self.name}: historical queries not supported")

    # -- generic -------------------------------------------------------------
    def poll(self, bbox: BBox, duration_s: float, interval_s: float | None = None,
             on_snapshot=None) -> pd.DataFrame:
        """Build tracks by repeatedly polling live snapshots (e.g. while you record)."""
        interval_s = max(interval_s or self.min_poll_interval, self.min_poll_interval)
        frames, t_end = [], _time.time() + duration_s
        while True:
            t_req = _time.time()
            try:
                df = self.snapshot(bbox)
                frames.append(df)
                if on_snapshot:
                    on_snapshot(df)
            except SourceError as e:
                log.warning("%s snapshot failed: %s", self.name, e)
            if _time.time() + interval_s > t_end:
                break
            _time.sleep(max(0.0, interval_s - (_time.time() - t_req)))
        return tracks.normalize(pd.concat(frames)) if frames else tracks.empty_frame()

    def fetch(self, bbox: BBox, t0: float | None = None, t1: float | None = None,
              use_cache: bool = True) -> pd.DataFrame:
        """Historical fetch with on-disk parquet cache keyed by (source,bbox,t0,t1)."""
        key = hashlib.sha1(json.dumps([self.name, bbox.__dict__, t0, t1]).encode()).hexdigest()[:16]
        path = self.cache_dir / f"{self.name}_{key}.parquet"
        for p in (path, path.with_suffix(".csv")):
            if use_cache and p.exists():
                log.info("cache hit %s", p)
                return tracks.load_tracks(p)
        if t0 is None:
            df = self.snapshot(bbox)
        else:
            df = self.history(bbox, t0, t1)
        df = tracks.normalize(df)
        tracks.save_tracks(df, path)
        return df
