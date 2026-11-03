r"""Observer geolocation from Doppler ridges + aircraft tracks.

Model for ridge r assigned to aircraft m, observer x = (lat, lon, h), clock
offset dt (true recording start = nominal + dt):

    f_r(t_i) = f_src,r * D_m(x, t0 + dt + t_i),   D = c_w / (c_w + v_r(t_e))

For fixed (x, dt, m) the best f_src is linear least squares:
    f_src = sum w f D / sum w D^2
so the grid search only scans (x, dt) and picks, per ridge, the aircraft that
explains it best (data association). Misfit chi2(x) = min_dt [ sum_r min_m chi2_rm + (dt/sigma_t)^2 ].
A coarse grid gives a likelihood map L ~ exp(-chi2/2); the best few local minima
are refined with scipy.optimize.least_squares using the exact physics model
(spline trajectory, Newton retarded time, ISA c_eff, wind) and the covariance
(J^T J)^{-1} s^2 gives the uncertainty.

Key geometry: the time of the S-curve inflection = CPA time (+ R_cpa/c) fixes
*where along* the track you are; the steepness df/dt|_cpa ~= -f v^2/(c R_cpa)
fixes the slant distance R_cpa, hence (knowing aircraft altitude) the lateral
offset. One straight pass leaves a left/right mirror ambiguity; a second
non-parallel track (or a turn) resolves it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize

from . import geodesy
from .atmosphere import effective_sound_speed
from .physics import Observer, Propagation, emission_time, doppler_factor, lloyd_delay


# ------------------------------------------------------------- fast sampled track
class SampledTrack:
    """Track pre-sampled on a fine uniform grid for vectorised grid search."""

    def __init__(self, model, dt=0.1, h_obs=0.0, prop: Propagation | None = None):
        self.model = model
        self.t = np.arange(model.t0, model.t1 + 1e-9, dt)
        self.dt = dt
        self.P = model.position(self.t)
        self.V = model.velocity(self.t)
        _, _, hs = geodesy.ecef_to_geodetic(self.P)
        prop = prop or Propagation()
        if prop.fixed_c is not None:
            self.c = np.full_like(hs, prop.fixed_c)
        else:
            self.c = effective_sound_speed(np.full_like(hs, prop.h_ground if prop.h_ground is not None else h_obs),
                                           hs, prop.dT, prop.temp_profile)

    def _interp(self, te):
        x = np.clip((te - self.t[0]) / self.dt, 0, len(self.t) - 1.000001)
        i = x.astype(int)
        a = (x - i)[..., None]
        P = self.P[i] * (1 - a) + self.P[i + 1] * a
        V = self.V[i] * (1 - a) + self.V[i + 1] * a
        c = self.c[i] * (1 - a[..., 0]) + self.c[i + 1] * a[..., 0]
        return P, V, c

    def doppler(self, O, t_obs, w_ecef=None, iters=5):
        """O: (G,3) observers, t_obs: (T,) -> factor D (G,T), valid mask (G,T)."""
        O = O[:, None, :]
        te = np.broadcast_to(t_obs[None, :], (O.shape[0], len(t_obs))) - 0.0
        P, _, _ = self._interp(te)
        te = te - np.linalg.norm(P - O, axis=-1) / 340.0
        for _ in range(iters):
            P, V, c = self._interp(te)
            d = P - O
            R = np.linalg.norm(d, axis=-1)
            u = d / R[..., None]
            cw = c - (u @ w_ecef if w_ecef is not None else 0.0)
            vr = np.einsum("gti,gti->gt", V, u)
            te = te - (te + R / cw - t_obs[None, :]) / (1 + vr / cw)
        valid = (te >= self.t[0]) & (te <= self.t[-1])
        return cw / (cw + vr), valid


# ------------------------------------------------------------------ result types
@dataclass
class GridResult:
    lat: np.ndarray
    lon: np.ndarray
    chi2: np.ndarray          # (ny, nx)
    best_dt: np.ndarray       # (ny, nx)
    loglik: np.ndarray
    assignment: np.ndarray    # (ny, nx, n_ridges) model index


@dataclass
class LocateResult:
    lat: float
    lon: float
    h: float
    dt: float
    f_src: np.ndarray
    assignment: list
    sigma_east_m: float
    sigma_north_m: float
    cov_en: np.ndarray
    cost: float
    rms_hz: float
    grid: GridResult | None = None
    alternatives: list = field(default_factory=list)
    h_receiver: float | None = None
    used_ridges: list = field(default_factory=list)
    assignment_all: list = field(default_factory=list)
    fine_grid: GridResult | None = None
    models: list = field(default_factory=list)
    score: float = float("nan")


def _ridge_arrays(ridges, sigma_scale=1.0):
    t = np.concatenate([r.t for r in ridges])
    f = np.concatenate([r.f for r in ridges])
    s = np.concatenate([(r.sigma_f if r.sigma_f is not None else np.ones_like(r.f)) for r in ridges]) * sigma_scale
    w = np.concatenate([r.conf for r in ridges]) / s ** 2
    idx = np.concatenate([np.full(len(r), i) for i, r in enumerate(ridges)])
    return t, f, w, idx


def grid_search(ridges, models, t_start, center, half_width_m=10000.0, step_m=250.0, h_obs=0.0,
                dt_sigma=2.0, dt_range=None, dt_step=0.5, prop: Propagation | None = None,
                sigma_scale=1.0, chunk=512, sampled=None, decimate=4):
    prop = prop or Propagation()
    if decimate > 1:  # grid search on every k-th frame (refinement uses all)
        from .ridge import Ridge
        ridges = [Ridge(r.t[::decimate], r.f[::decimate], r.snr_db[::decimate], r.conf[::decimate],
                        None if r.sigma_f is None else r.sigma_f[::decimate], r.label) for r in ridges]
    lat0, lon0 = center
    xs = np.arange(-half_width_m, half_width_m + 1e-6, step_m)
    E, N = np.meshgrid(xs, xs)
    lat, lon = geodesy.offset_latlon(lat0, lon0, E, N)
    O = geodesy.geodetic_to_ecef(lat.ravel(), lon.ravel(), np.full(lat.size, h_obs))
    w_ecef = None
    if prop.wind_enu is not None:
        w_ecef = geodesy.enu_velocity_to_ecef(np.array([*prop.wind_enu[:2], 0.0]), lat0, lon0)
    sampled = sampled or [SampledTrack(m, 0.1, h_obs, prop) for m in models]
    t, f, w, idx = _ridge_arrays(ridges, sigma_scale)
    tu, inv = np.unique(t, return_inverse=True)
    nr = len(ridges)
    dts = np.arange(*(dt_range or (-3 * dt_sigma, 3 * dt_sigma)), dt_step) if dt_sigma > 0 else np.array([0.0])
    if dt_sigma > 0:
        dts = np.append(dts, dts[-1] + dt_step)
    G = O.shape[0]
    nd = len(dts)
    chi_all = np.full((G, nd), np.inf)              # data chi2 per (observer, clock offset)
    asg_all = np.zeros((G, nd, nr), dtype=np.int16)
    for g0 in range(0, G, chunk):
        Og = O[g0:g0 + chunk]
        for di, dt in enumerate(dts):
            chis = np.full((len(sampled), len(Og), nr), np.inf)
            for mi, st in enumerate(sampled):
                D, valid = st.doppler(Og, t_start + dt + tu, w_ecef)
                D, valid = D[:, inv], valid[:, inv]
                for r in range(nr):
                    sel = idx == r
                    Dr, fr, wr = D[:, sel], f[sel], w[sel]
                    if not valid[:, sel].all(axis=1).any():
                        continue
                    fs = (wr * fr * Dr).sum(1) / (wr * Dr * Dr).sum(1)
                    c2 = (wr * (fr - fs[:, None] * Dr) ** 2).sum(1)
                    chis[mi, :, r] = np.where(valid[:, sel].mean(axis=1) > 0.95, c2, np.inf)
            asg_all[g0:g0 + chunk, di] = np.argmin(chis, axis=0)
            tot = np.min(chis, axis=0)
            # a ridge no aircraft can explain -> large finite penalty
            chi_all[g0:g0 + chunk, di] = np.where(np.isfinite(tot), tot, 1e12).sum(1)
    # The ridge sigmas are optimistic (track/model errors dominate), so the data chi2
    # is rescaled by its reduced value at the optimum before the clock prior is added.
    n_dof = max(1, len(f) - nr - 3)
    s2 = max(np.min(chi_all) / n_dof, 1.0)
    prior = (dts / dt_sigma) ** 2 if dt_sigma > 0 else np.zeros(nd)
    total = chi_all / s2 + prior[None, :]
    bi = np.argmin(total, axis=1)
    shape = E.shape
    chi2 = total[np.arange(G), bi].reshape(shape)
    best_dt = dts[bi]
    best_assign = asg_all[np.arange(G), bi]
    loglik = -0.5 * (chi2 - chi2.min())
    return GridResult(lat, lon, chi2, best_dt.reshape(shape), loglik,
                      best_assign.reshape(shape + (nr,)))


def _local_minima(chi2, k=3, sep=5):
    mn = ndimage.minimum_filter(chi2, size=2 * sep + 1, mode="nearest")
    cand = np.argwhere((chi2 == mn) & np.isfinite(chi2))
    order = np.argsort(chi2[tuple(cand.T)])
    return [tuple(c) for c in cand[order][:k]]


def refine(ridges, models, t_start, lat0, lon0, h_obs, dt0, assignment, prop=None, dt_sigma=2.0,
           fit_height=False, sigma_scale=1.0, fringe=None, h_receiver0=1.5, fringe_sigma=1e-4,
           loss="soft_l1", f_scale=2.0, max_dt_sigmas=4.0, max_move_m=3000.0):
    """Exact-physics least squares on (east, north, [up], dt, [h_r]).

    The per-ridge source frequencies f_src enter linearly and are projected out
    (variable projection): for each trial geometry f_src,r = sum w f D / sum w D^2.
    """
    prop = prop or Propagation()
    use_fr = fringe is not None and len(fringe.t) > 5
    POS = 100.0   # position parameters in units of 100 m (keeps finite differences sane)

    def unpack(p):
        e, n = p[0] * POS, p[1] * POS
        j = 2
        u = 0.0
        if fit_height:
            u, j = p[2] * POS, 3
        dt = p[j]
        hr = p[j + 1] if use_fr else None
        return e, n, u, dt, hr

    def obs_of(e, n, u):
        la, lo, _ = geodesy.enu_to_geodetic(np.array([e, n, 0.0]), lat0, lon0, h_obs)
        return Observer(float(la), float(lo), float(h_obs + u))

    def model_eval(p):
        e, n, u, dt, hr = unpack(p)
        ob = obs_of(e, n, u)
        res, fsrc = [], []
        for r, ridge in enumerate(ridges):
            m = models[assignment[r]]
            te, g = emission_time(m, ob, t_start + dt + ridge.t, prop)
            D = doppler_factor(g["vr"], g["c"])
            s = (ridge.sigma_f if ridge.sigma_f is not None else np.ones_like(ridge.f)) * sigma_scale
            w = ridge.conf / s ** 2
            fs = float((w * ridge.f * D).sum() / (w * D * D).sum())
            fsrc.append(fs)
            z = (ridge.f - fs * D) * np.sqrt(ridge.conf) / s
            if robust:   # soft-L1 applied to the frequency residuals only (priors stay quadratic)
                z = np.sign(z) * f_scale * np.sqrt(2.0 * (np.sqrt(1.0 + (z / f_scale) ** 2) - 1.0))
            res.append(z)
        if dt_sigma > 0:
            res.append(np.array([dt / dt_sigma]))
        if use_fr:
            taus = []
            for mi in sorted(set(int(a) for a in assignment)):
                te, g = emission_time(models[mi], ob, t_start + dt + fringe.t, prop)
                taus.append((g["R"], lloyd_delay(g["P"], ob, hr, g["c"])))
            Rs = np.stack([a for a, _ in taus])
            tp = np.stack([b for _, b in taus])[np.argmin(Rs, 0), np.arange(len(fringe.t))]
            res.append((fringe.tau - tp) / fringe_sigma)
        return np.concatenate(res), np.array(fsrc), ob

    p0 = [0.0, 0.0] + ([0.0] if fit_height else []) + [dt0] + ([h_receiver0] if use_fr else [])
    lb = [-max_move_m / POS, -max_move_m / POS] + ([-np.inf] if fit_height else []) + [-max_dt_sigmas * dt_sigma if dt_sigma > 0 else -np.inf] \
        + ([0.2] if use_fr else [])
    ub = [max_move_m / POS, max_move_m / POS] + ([np.inf] if fit_height else []) + [max_dt_sigmas * dt_sigma if dt_sigma > 0 else np.inf] \
        + ([20.0] if use_fr else [])
    p0[len(p0) - 1 - int(use_fr)] = float(np.clip(dt0, lb[len(p0) - 1 - int(use_fr)] + 1e-6,
                                                   ub[len(p0) - 1 - int(use_fr)] - 1e-6))
    robust = True
    nfreq = sum(len(r) for r in ridges)
    for it in range(2):
        sol = optimize.least_squares(lambda p: model_eval(p)[0], np.array(p0), x_scale=1.0, method="trf",
                                     loss="linear", diff_step=1e-4, bounds=(lb, ub))
        # rescale optimistic ridge sigmas by the reduced chi2 of the frequency residuals so the
        # clock prior (and Lloyd term) keep their intended weight, then refit once
        robust = False
        zf = model_eval(sol.x)[0][:nfreq]
        robust = True
        red = float(np.sum(zf ** 2)) / max(1, nfreq - len(sol.x) - len(ridges))
        if it == 0 and red > 1.5:
            sigma_scale = sigma_scale * np.sqrt(red)
            p0 = list(sol.x)
        else:
            break
    robust = False   # covariance / rms from plain residuals
    e, n, u, dt, hr = unpack(sol.x)
    fun, fsrc, ob = model_eval(sol.x)
    J = sol.jac
    dof = max(1, len(fun) - len(sol.x) - len(ridges))
    s2 = float(np.sum(fun ** 2)) / dof   # reduced chi2 (un-robustified residuals)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((len(sol.x), len(sol.x)), np.nan)
    cov[:2, :2] *= POS ** 2
    resid_hz = []
    for r, ridge in enumerate(ridges):
        te, g = emission_time(models[assignment[r]], ob, t_start + dt + ridge.t, prop)
        resid_hz.append(ridge.f - fsrc[r] * doppler_factor(g["vr"], g["c"]))
    rms = float(np.sqrt(np.mean(np.concatenate(resid_hz) ** 2)))
    return LocateResult(ob.lat, ob.lon, ob.h, float(dt), fsrc, list(map(int, assignment)),
                        float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])), cov[:2, :2],
                        float(sol.cost), rms, h_receiver=None if hr is None else float(hr))


def prefilter_models(models, center, half_width_m, t0, t1, max_range_m=15000.0):
    """Keep aircraft that come within max_range_m of the search area during [t0, t1]."""
    lat0, lon0 = center
    keep = []
    for m in models:
        if m.t1 < t0 - 60 or m.t0 > t1:
            continue
        tt = np.linspace(max(m.t0, t0 - 60), min(m.t1, t1), 200)
        la, lo, h = m.geodetic(tt)
        enu = geodesy.geodetic_to_enu(la, lo, h, lat0, lon0, 0.0)
        dx = np.maximum(np.abs(enu[:, 0]) - half_width_m, 0)
        dy = np.maximum(np.abs(enu[:, 1]) - half_width_m, 0)
        if np.min(np.sqrt(dx ** 2 + dy ** 2 + enu[:, 2] ** 2)) < max_range_m:
            keep.append(m)
    return keep


def _ridge_misfit(ridges, models, t_start, res, prop):
    from .physics import Observer
    ob = Observer(res.lat, res.lon, res.h)
    out = []
    for r, ridge in enumerate(ridges):
        te, g = emission_time(models[res.assignment[r]], ob, t_start + res.dt + ridge.t, prop)
        pred = res.f_src[r] * doppler_factor(g["vr"], g["c"])
        s = ridge.sigma_f if ridge.sigma_f is not None else 1.0
        out.append(float(np.sqrt(np.mean(((ridge.f - pred) / s) ** 2))))
    return np.array(out)


def global_score(ridges, models, t_start, res, prop, cap=5.0):
    """Truncated chi2 of *all* ridges at a solution (each ridge -> best aircraft).

    Used to rank competing local solutions on the same data, so a solution can't
    win just by rejecting inconvenient ridges.
    """
    from .physics import Observer
    ob = Observer(res.lat, res.lon, res.h)
    total = 0.0
    for ridge in ridges:
        s = ridge.sigma_f if ridge.sigma_f is not None else np.ones_like(ridge.f)
        best = np.inf
        for m in models:
            te, g = emission_time(m, ob, t_start + res.dt + ridge.t, prop)
            D = doppler_factor(g["vr"], g["c"])
            fs = (ridge.f * D / s ** 2).sum() / (D * D / s ** 2).sum()
            z = (ridge.f - fs * D) / s
            best = min(best, float(np.minimum(z ** 2, cap ** 2).sum()))
        total += best
    return total


def locate(ridges, models, t_start, center, half_width_m=10000.0, step_m=600.0, h_obs=0.0,
           dt_sigma=2.0, prop=None, n_starts=6, fit_height=False, fringe=None, h_receiver0=1.5,
           sigma_scale=1.0, fine_half_width_m=None, fine_step_m=150.0, reject_outliers=True,
           max_range_m=15000.0, verbose=True):
    """Coarse grid -> fine grids around the best local minima -> exact least squares.

    Returns LocateResult with .grid (coarse map), .fine_grid and .alternatives.
    Ridges that no aircraft explains (e.g. a ridge that jumped between two
    crossing tones) are rejected if their normalised misfit is > 4x the median.
    """
    prop = prop or Propagation()
    t_end = t_start + max(r.t[-1] for r in ridges) + 60
    models = prefilter_models(models, center, half_width_m, t_start - 3 * dt_sigma - 60, t_end, max_range_m)
    if verbose:
        print(f"[locate] {len(models)} candidate aircraft, {len(ridges)} ridges")
    if not models:
        raise RuntimeError("no aircraft tracks near the search area in the recording window")
    sampled = [SampledTrack(m, 0.1, h_obs, prop) for m in models]
    grid = grid_search(ridges, models, t_start, center, half_width_m, step_m, h_obs, dt_sigma,
                       dt_step=1.0, prop=prop, sigma_scale=sigma_scale, sampled=sampled, decimate=8)
    fine_hw = fine_half_width_m or 2.5 * step_m
    results = []
    for (iy, ix) in _local_minima(grid.chi2, n_starts, sep=max(1, int(1200 / step_m))):
        c = (float(grid.lat[iy, ix]), float(grid.lon[iy, ix]))
        dtc = float(grid.best_dt[iy, ix])
        fg = grid_search(ridges, models, t_start, c, fine_hw, fine_step_m, h_obs, dt_sigma,
                         dt_range=(dtc - 1.5, dtc + 1.5), dt_step=0.5, prop=prop, sigma_scale=sigma_scale,
                         sampled=sampled, decimate=8)
        jy, jx = np.unravel_index(np.argmin(fg.chi2), fg.chi2.shape)
        try:
            args = (float(fg.lat[jy, jx]), float(fg.lon[jy, jx]), h_obs, float(fg.best_dt[jy, jx]))
            assign = fg.assignment[jy, jx]
            res = refine(ridges, models, t_start, *args, assign, prop, dt_sigma, fit_height, sigma_scale)
            used = list(range(len(ridges)))
            for _ in range(3 if reject_outliers else 0):
                if len(used) <= 2:
                    break
                sub = [ridges[i] for i in used]
                mis = _ridge_misfit(sub, models, t_start, res, prop)
                keep = [u for u, m in zip(used, mis) if m <= max(2.5 * np.median(mis), 2.0)]
                if len(keep) == len(used):
                    break
                if verbose:
                    print(f"[locate]   rejecting {[ridges[i].label for i in used if i not in keep]} "
                          f"(normalised rms misfit {dict((ridges[u].label, round(float(m), 1)) for u, m in zip(used, mis))})")
                used = keep
                res = refine([ridges[i] for i in used], models, t_start, res.lat, res.lon, h_obs, res.dt,
                             [assign[i] for i in used], prop, dt_sigma, fit_height, sigma_scale)
            if fringe is not None:
                res2 = refine([ridges[i] for i in used], models, t_start, res.lat, res.lon, h_obs, res.dt,
                              res.assignment, prop, dt_sigma, fit_height, sigma_scale, fringe, h_receiver0)
                res.h_receiver = res2.h_receiver
            res.used_ridges = used
            res.assignment_all = [int(a) for a in assign]
            res.fine_grid = fg
            results.append(res)
            if verbose:
                print(f"[locate] start {c[0]:.5f},{c[1]:.5f} -> {res.lat:.5f},{res.lon:.5f} cost={res.cost:.1f} "
                      f"rms={res.rms_hz:.3f} Hz dt={res.dt:+.2f}s")
        except Exception as e:  # pragma: no cover
            print("refine failed:", e)
    # compare on a common footing: cost per used point
    for r in results:
        r.score = global_score(ridges, models, t_start, r, prop)
    results.sort(key=lambda r: r.score)
    # polish: clock offset and along-track position trade off, so restart the winner from
    # a few shifted clock offsets and keep whatever explains all ridges best
    b0 = results[0]
    for ddt in (-2.0, -1.0, 1.0, 2.0):
        try:
            sub = [ridges[i] for i in b0.used_ridges]
            r2 = refine(sub, models, t_start, b0.lat, b0.lon, h_obs, b0.dt + ddt, b0.assignment, prop, dt_sigma,
                        fit_height, sigma_scale)
            r2.score = global_score(ridges, models, t_start, r2, prop)
            if r2.score < results[0].score - 1e-6:
                r2.used_ridges, r2.assignment_all, r2.fine_grid = b0.used_ridges, b0.assignment_all, b0.fine_grid
                r2.h_receiver = b0.h_receiver
                if fringe is not None:
                    r2.h_receiver = refine(sub, models, t_start, r2.lat, r2.lon, h_obs, r2.dt, r2.assignment, prop,
                                           dt_sigma, fit_height, sigma_scale, fringe, h_receiver0).h_receiver
                if verbose:
                    print(f"[locate] polish dt0{ddt:+.0f}s -> {r2.lat:.5f},{r2.lon:.5f} dt={r2.dt:+.2f}s score={r2.score:.1f}")
                results[0] = r2
        except Exception as e:  # pragma: no cover
            print("polish failed:", e)
    results.sort(key=lambda r: r.score)
    if verbose:
        print("[locate] solutions ranked by truncated chi2 over all ridges:",
              [round(r.score, 1) for r in results])
    best = results[0]
    best.grid = grid
    best.alternatives = results[1:]
    best.models = models
    return best
