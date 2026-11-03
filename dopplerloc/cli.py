"""Command line: python -m dopplerloc {fetch,spectrogram,extract,locate,simulate}"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_time(s):
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return pd.Timestamp(s).tz_localize("UTC").timestamp() if pd.Timestamp(s).tzinfo is None \
            else pd.Timestamp(s).timestamp()


def cmd_fetch(a):
    from . import tracks
    from .sources import BBox, get_source
    src = get_source(a.source, cache_dir=a.cache_dir)
    bbox = BBox.around(a.lat, a.lon, a.radius_km)
    if a.poll:
        print(f"polling {a.source} every {a.interval or src.min_poll_interval}s for {a.poll}s ...")
        df = src.poll(bbox, a.poll, a.interval, on_snapshot=lambda d: print(f"  {time.strftime('%H:%M:%S')} {len(d)} aircraft"))
    else:
        df = src.fetch(bbox, parse_time(a.start), parse_time(a.end), use_cache=not a.no_cache)
    if a.resample:
        df = tracks.resample_uniform(df, a.resample)
    p = tracks.save_tracks(df, a.out)
    print(f"{len(df)} rows, {df['icao24'].nunique() if len(df) else 0} aircraft -> {p}")
    if len(df):
        print(df.groupby("icao24").agg(callsign=("callsign", "first"), n=("time", "size"),
                                       alt_m=("alt_baro", "median")).head(30).to_string())


def _load(a):
    from . import audio, pipeline
    x, sr = audio.load_audio(a.audio)
    cfg = pipeline.ExtractConfig(sr=a.sr, win_length=a.win, hop=a.hop, n_fft=a.nfft, fmin=a.fmin, fmax=a.fmax,
                                 mains=a.mains, n_ridges=a.n_ridges, snr_threshold_db=a.snr_db,
                                 reassigned=getattr(a, "reassigned", False))
    return x, sr, cfg


def cmd_spectrogram(a):
    from . import audio, plotting, pipeline
    x, sr, cfg = _load(a)
    if sr != cfg.sr:
        x, sr = audio.resample(x, sr, cfg.sr), cfg.sr
    S = pipeline.spectrogram(x, sr, cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plotting.plot_spectrogram(S, a.out, fmax=a.fmax, title=f"{Path(a.audio).name}"
                              + (" (reassigned)" if cfg.reassigned else ""),
                              vmin_pct=60 if cfg.reassigned else 20, vmax_pct=99.9)
    print(f"-> {a.out}")


def cmd_extract(a):
    from . import pipeline, plotting
    x, sr, cfg = _load(a)
    S_raw, S_line, ridges, S = pipeline.extract(x, sr, cfg)
    df = pd.concat([r.to_frame() for r in ridges]) if ridges else pd.DataFrame()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.plot:
        Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(ridges)} ridges -> {a.out}")
    if a.plot:
        plotting.plot_spectrogram_pair(S_raw, S_line, ridges, a.plot, fmax=a.fmax)
        print(f"-> {a.plot}")


def cmd_locate(a):
    from . import geodesy, locate, plotting, tracks
    from .physics import Propagation
    from .ridge import Ridge
    rdf = pd.read_csv(a.ridges)
    ridges = [Ridge(g["t"].to_numpy(), g["f"].to_numpy(), g["snr_db"].to_numpy(), g["conf"].to_numpy(),
                    g["sigma_f"].to_numpy(), str(k)) for k, g in rdf.groupby("ridge", sort=False)]
    models = tracks.build_models(tracks.load_tracks(a.tracks), geoid_n=a.geoid_n)
    t0 = parse_time(a.start)
    lat, lon = map(float, a.center.split(","))
    wind = tuple(map(float, a.wind.split(","))) if a.wind else None
    prop = Propagation(dT=a.isa_dt, wind_enu=wind)
    res = locate.locate(ridges, models, t0, (lat, lon), a.radius_km * 1000, a.grid_step, a.obs_h, a.start_sigma, prop)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plotting.plot_heatmap(res, res.models, out / "location_heatmap.png", center=(lat, lon), zoom_m=1000)
    plotting.plot_doppler_fit([ridges[i] for i in res.used_ridges], res, res.models, t0, prop,
                              out / "doppler_fit.png", all_ridges=ridges)
    infl = a.sigma_inflate
    summary = dict(lat=res.lat, lon=res.lon, h=res.h, clock_offset_s=res.dt, sigma_east_m=res.sigma_east_m * infl,
                   sigma_north_m=res.sigma_north_m * infl, rms_hz=res.rms_hz,
                   note="sigma = formal LSQ sigma x sigma_inflate (STFT frame correlation); excludes "
                        "systematic track/propagation errors, which usually dominate",
                   assignment={ridges[i].label: res.models[res.assignment[k]].callsign or res.models[res.assignment[k]].icao24
                               for k, i in enumerate(res.used_ridges)},
                   alternatives=[dict(lat=x.lat, lon=x.lon, score=x.score) for x in res.alternatives])
    (out / "locate_result.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


def cmd_simulate(a):
    from . import demo
    demo.run(a.out_dir, a.radius_km * 1000)


def build_parser():
    p = argparse.ArgumentParser(prog="dopplerloc", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="fetch aircraft tracks for a bbox / time window")
    f.add_argument("--source", default="opensky",
                   choices=["opensky", "opensky-trino", "adsblol", "airplaneslive", "adsbx", "fr24"])
    f.add_argument("--lat", type=float, default=39.4893)
    f.add_argument("--lon", type=float, default=-0.4816)
    f.add_argument("--radius-km", type=float, default=25.0)
    f.add_argument("--start", help="unix seconds or ISO time (UTC) - historical sources only")
    f.add_argument("--end")
    f.add_argument("--poll", type=float, help="poll live snapshots for N seconds (use while recording)")
    f.add_argument("--interval", type=float)
    f.add_argument("--resample", type=float, help="interpolate to a uniform grid with this step [s]")
    f.add_argument("--out", default="data/tracks.parquet")
    f.add_argument("--cache-dir", default="data/cache")
    f.add_argument("--no-cache", action="store_true")
    f.set_defaults(fn=cmd_fetch)

    for name, fn, hlp in (("spectrogram", cmd_spectrogram, "plot a spectrogram"),
                          ("extract", cmd_extract, "clean audio and extract Doppler ridges to CSV")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("audio")
        s.add_argument("--out", required=True)
        s.add_argument("--sr", type=int, default=8000)
        s.add_argument("--win", type=int, default=4096)
        s.add_argument("--hop", type=int, default=400)
        s.add_argument("--nfft", type=int, default=8192)
        s.add_argument("--fmin", type=float, default=150.0)
        s.add_argument("--fmax", type=float, default=2500.0)
        s.add_argument("--mains", type=float, default=50.0, help="mains frequency (0 = off; 60 in the Americas)")
        s.add_argument("--n-ridges", type=int, default=8)
        s.add_argument("--snr-db", type=float, default=8.0)
        s.add_argument("--reassigned", action="store_true")
        s.add_argument("--plot")
        s.set_defaults(fn=fn)

    l = sub.add_parser("locate", help="geolocate from ridges.csv + tracks")
    l.add_argument("--ridges", required=True)
    l.add_argument("--tracks", required=True)
    l.add_argument("--start", required=True, help="nominal recording start (unix or ISO UTC)")
    l.add_argument("--start-sigma", type=float, default=2.0, help="clock uncertainty [s]")
    l.add_argument("--center", required=True, help="search centre 'lat,lon'")
    l.add_argument("--radius-km", type=float, default=10.0)
    l.add_argument("--grid-step", type=float, default=600.0)
    l.add_argument("--obs-h", type=float, default=100.0, help="observer ellipsoidal height [m] (MSL + ~50 m in Spain)")
    l.add_argument("--geoid-n", type=float, default=0.0, help="add to track altitudes (if they are MSL)")
    l.add_argument("--isa-dt", type=float, default=0.0, help="ISA temperature offset [K]")
    l.add_argument("--wind", help="'we,wn' wind towards east,north [m/s]")
    l.add_argument("--sigma-inflate", type=float, default=3.2, help="sqrt(win/hop) for correlated STFT frames")
    l.add_argument("--out-dir", default="outputs")
    l.set_defaults(fn=cmd_locate)

    s = sub.add_parser("simulate", help="run the end-to-end synthetic LEVC demo")
    s.add_argument("--out-dir", default="outputs")
    s.add_argument("--radius-km", type=float, default=12.0)
    s.set_defaults(fn=cmd_simulate)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)
    from .sources import SourceError
    try:
        a.fn(a)
    except SourceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
