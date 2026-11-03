"""Pluggable flight-data sources. All return the canonical long-format frame
(see dopplerloc.tracks.COLUMNS)."""
from .base import BBox, DataSource, SourceError
from .opensky import OpenSkyRest, OpenSkyTrino
from .readsb import ReadsbV2
from .fr24 import Flightradar24Api

SOURCES = {
    "opensky": OpenSkyRest,
    "opensky-trino": OpenSkyTrino,
    "adsblol": lambda **kw: ReadsbV2.preset("adsblol", **kw),
    "airplaneslive": lambda **kw: ReadsbV2.preset("airplaneslive", **kw),
    "adsbx": lambda **kw: ReadsbV2.preset("adsbx", **kw),
    "fr24": Flightradar24Api,
}


def get_source(name: str, **kw) -> DataSource:
    try:
        return SOURCES[name](**kw)
    except KeyError:
        raise SourceError(f"unknown source {name!r}; choose from {sorted(SOURCES)}") from None
