"""Step-by-step version of locate.locate() that keeps every intermediate stage.

Used by the walkthrough notebook to show the location likelihood being refined:
coarse grid -> medium grid(s) around the best coarse minima -> fine grid -> exact least squares
(+ outlier-ridge rejection, clock/along-track polish and Lloyd's-mirror mic height).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import locate as L
from .physics import Propagation


@dataclass
class Stage:
    name: str
    step_m: float | None
    grid: L.GridResult | None
    lat: float
    lon: float
    dt: float
    seconds: float
    note: str = ""
    result: L.LocateResult | None = None
    extra: dict = field(default_factory=dict)


def _best(g: L.GridResult):
    iy, ix = np.unravel_index(np.argmin(g.chi2), g.chi2.shape)
    return iy, ix, float(g.lat[iy, ix]), float(g.lon[iy, ix]), float(g.best_dt[iy, ix])


def progressive_locate(ridges, models, t_start, center, half_width_m=12000.0, coarse_step_m=1200.0,
                       h_obs=0.0, dt_sigma=2.0, prop=None, fringe=None, n_starts=3, fine_step_m=100.0,
                       verbose=True):
    prop = prop or Propagation()
    t_end = t_start + max(r.t[-1] for r in ridges) + 60
    models = L.prefilter_models(models, center, half_width_m, t_start - 3 * dt_sigma - 60, t_end)
    sampled = [L.SampledTrack(m, 0.1, h_obs, prop) for m in models]
    stages = []

    def log(s):
        if verbose:
            print(f"[{s.name}] best {s.lat:.5f},{s.lon:.5f} dt={s.dt:+.2f}s  ({s.seconds:.1f} s) {s.note}")

    # 1. coarse grid over the whole search area
    T = time.time()
    g1 = L.grid_search(ridges, models, t_start, center, half_width_m, coarse_step_m, h_obs, dt_sigma,
                       dt_step=1.0, prop=prop, sampled=sampled, decimate=8)
    _, _, la, lo, dt = _best(g1)
    stages.append(Stage(f"1 coarse grid ({coarse_step_m:.0f} m)", coarse_step_m, g1, la, lo, dt, time.time() - T,
                        note=f"{g1.chi2.size} cells, {len(models)} aircraft"))
    log(stages[-1])

    # 2. medium grids around the n best separated coarse minima (mirror / along-track ambiguities)
    T = time.time()
    med_step = coarse_step_m / 4
    cands = []
    for (iy, ix) in L._local_minima(g1.chi2, n_starts, sep=max(1, int(2400 / coarse_step_m))):
        c = (float(g1.lat[iy, ix]), float(g1.lon[iy, ix]))
        dtc = float(g1.best_dt[iy, ix])
        g = L.grid_search(ridges, models, t_start, c, 1.5 * coarse_step_m, med_step, h_obs, dt_sigma,
                          dt_range=(dtc - 2, dtc + 2), dt_step=0.5, prop=prop, sampled=sampled, decimate=8)
        cands.append(g)
    # rank candidates by the exact global score at their best cell (common footing across grids)
    scored = []
    for g in cands:
        iy, ix, la, lo, dt = _best(g)
        r = L.refine(ridges, models, t_start, la, lo, h_obs, dt, g.assignment[iy, ix], prop, dt_sigma)
        scored.append((L.global_score(ridges, models, t_start, r, prop), g, r))
    scored.sort(key=lambda s: s[0])
    g2 = scored[0][1]
    _, _, la, lo, dt = _best(g2)
    stages.append(Stage(f"2 medium grid ({med_step:.0f} m)", med_step, g2, la, lo, dt, time.time() - T,
                        note=f"{len(cands)} candidate basins, scores {[round(s[0], 1) for s in scored]}",
                        extra=dict(candidates=[s[1] for s in scored])))
    log(stages[-1])

    # 3. fine grid around the winner
    T = time.time()
    g3 = L.grid_search(ridges, models, t_start, (la, lo), 4 * fine_step_m * 2, fine_step_m, h_obs, dt_sigma,
                       dt_range=(dt - 1.5, dt + 1.5), dt_step=0.25, prop=prop, sampled=sampled, decimate=4)
    iy, ix, la, lo, dt = _best(g3)
    stages.append(Stage(f"3 fine grid ({fine_step_m:.0f} m)", fine_step_m, g3, la, lo, dt, time.time() - T))
    log(stages[-1])

    # 4. exact least squares with outlier-ridge rejection + clock/along-track polish (+ Lloyd h_r)
    T = time.time()
    assign = g3.assignment[iy, ix]
    res = L.refine(ridges, models, t_start, la, lo, h_obs, dt, assign, prop, dt_sigma)
    used = list(range(len(ridges)))
    for _ in range(3):
        sub = [ridges[i] for i in used]
        mis = L._ridge_misfit(sub, models, t_start, res, prop)
        keep = [u for u, m in zip(used, mis) if m <= max(2.5 * np.median(mis), 2.0)]
        if len(keep) == len(used) or len(keep) < 2:
            break
        used = keep
        res = L.refine([ridges[i] for i in used], models, t_start, res.lat, res.lon, h_obs, res.dt,
                       [assign[i] for i in used], prop, dt_sigma)
    res.used_ridges = used
    res.score = L.global_score(ridges, models, t_start, res, prop)
    sub = [ridges[i] for i in used]
    for ddt in (-2.0, -1.0, 1.0, 2.0):
        r2 = L.refine(sub, models, t_start, res.lat, res.lon, h_obs, res.dt + ddt, res.assignment, prop, dt_sigma)
        r2.score = L.global_score(ridges, models, t_start, r2, prop)
        if r2.score < res.score - 1e-6:
            r2.used_ridges = used
            res = r2
    if fringe is not None:
        res.h_receiver = L.refine(sub, models, t_start, res.lat, res.lon, h_obs, res.dt, res.assignment, prop,
                                  dt_sigma, fringe=fringe).h_receiver
    res.grid, res.fine_grid, res.models = g1, g3, models
    stages.append(Stage("4 least squares", None, None, res.lat, res.lon, res.dt, time.time() - T,
                        note=f"{len(used)}/{len(ridges)} ridges used, rms {res.rms_hz:.2f} Hz"
                             + (f", h_r {res.h_receiver:.2f} m" if res.h_receiver else ""), result=res))
    log(stages[-1])
    return stages, models
