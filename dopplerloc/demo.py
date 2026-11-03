"""End-to-end synthetic demo: simulate -> extract -> locate -> report + PNGs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import audio, geodesy, lloyd, locate, pipeline, plotting, simulate, tracks
from .physics import Propagation, emission_time, lloyd_delay


def run(out_dir="outputs", half_width_m=12000.0, verbose=True, seed=7, adsb_seed=1):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = time.time()
    sc = simulate.scenario()
    obs = sc["observer"]
    x, sr, t_nom = simulate.synthesize(sc, seed=seed)
    audio.save_wav(out / "synthetic_recording.wav", x, sr)
    df = simulate.adsb_samples(sc, seed=adsb_seed)
    tracks.save_tracks(df, out / "synthetic_adsb_tracks.parquet")
    print(f"[demo] synthesized {len(x)/sr:.0f} s @ {sr} Hz; hidden observer {obs.lat:.5f},{obs.lon:.5f}; "
          f"recording clock error {t_nom - sc['t_rec0']:+.2f} s")

    cfg = pipeline.ExtractConfig()
    S_raw, S_line, ridges, S = pipeline.extract(x, sr, cfg)
    print(f"[demo] extracted {len(ridges)} ridges")
    models = tracks.build_models(df, geoid_n=0.0)
    prop = Propagation()
    fringe = lloyd.cepstral_delay(S, fmin=150.0, fmax=2500.0)
    center = simulate.LEVC[:2]
    res = locate.locate(ridges, models, t_nom, center=center, half_width_m=half_width_m, step_m=600.0,
                        h_obs=obs.h, dt_sigma=2.0, prop=prop, fringe=fringe, verbose=verbose)
    err = float(geodesy.haversine_m(res.lat, res.lon, obs.lat, obs.lon))
    true_dt = sc["t_rec0"] - t_nom
    used = [ridges[i] for i in res.used_ridges]
    # correlated STFT frames: inflate formal sigma by sqrt(win/hop)
    infl = float(np.sqrt(cfg.win_length / cfg.hop))
    summary = dict(
        truth=dict(lat=obs.lat, lon=obs.lon, h=obs.h, clock_offset_s=true_dt, h_receiver=sc["h_receiver"]),
        estimate=dict(lat=res.lat, lon=res.lon, h=res.h, clock_offset_s=res.dt, h_receiver=res.h_receiver),
        error_m=err, sigma_east_m=res.sigma_east_m * infl, sigma_north_m=res.sigma_north_m * infl,
        rms_hz=res.rms_hz, n_ridges=len(ridges), n_used=len(used),
        ridge_assignment={ridges[i].label: res.models[res.assignment[k]].callsign for k, i in enumerate(res.used_ridges)},
        f_src_hz={ridges[i].label: round(float(f), 2) for i, f in zip(res.used_ridges, res.f_src)},
        true_tones={a.callsign: [t[0] for t in a.tones] for a in sc["aircraft"] if a.audible},
        alternatives=[dict(lat=a.lat, lon=a.lon, score=a.score,
                           dist_from_truth_m=float(geodesy.haversine_m(a.lat, a.lon, obs.lat, obs.lon)))
                      for a in res.alternatives],
        best_score=res.score, runtime_s=time.time() - T)
    (out / "synthetic_result.json").write_text(json.dumps(summary, indent=2, default=float))

    plotting.plot_spectrogram_pair(S_raw, S_line, ridges, out / "spectrogram_ridges.png", fmax=2200)
    plotting.plot_doppler_fit(used, res, res.models, t_nom, prop, out / "doppler_fit.png", all_ridges=ridges)
    plotting.plot_heatmap(res, res.models, out / "location_heatmap.png", truth=obs,
                          t_window=(t_nom - 30, t_nom + 180), center=center, zoom_m=400)
    if res.h_receiver:
        from .physics import Observer
        ob = Observer(res.lat, res.lon, res.h)
        tt = np.linspace(0, S.times[-1], 600)
        Rs, taus = [], []
        for mi in sorted(set(res.assignment)):
            te, g = emission_time(res.models[mi], ob, t_nom + res.dt + tt, prop)
            Rs.append(g["R"]); taus.append(lloyd_delay(g["P"], ob, res.h_receiver, g["c"]))
        tp = np.stack(taus)[np.argmin(np.stack(Rs), 0), np.arange(len(tt))]
        plotting.plot_fringes(fringe, tt, tp, out / "lloyd_fringes.png", res.h_receiver)
    print(f"[demo] estimate {res.lat:.6f},{res.lon:.6f}  error = {err:.1f} m  "
          f"(1-sigma E/N ~ {summary['sigma_east_m']:.1f}/{summary['sigma_north_m']:.1f} m formal x{infl:.1f}), "
          f"clock offset {res.dt:+.2f} s (true {true_dt:+.2f}), h_r {res.h_receiver} (true {sc['h_receiver']})")
    print(f"[demo] outputs in {out.resolve()}  ({time.time()-T:.0f} s)")
    return summary
