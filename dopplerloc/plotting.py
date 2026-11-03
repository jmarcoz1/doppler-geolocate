"""Matplotlib helpers (Agg backend; saves PNGs)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import geodesy  # noqa: E402
from .physics import doppler_factor, emission_time  # noqa: E402


def _maxpool(db, freqs, times, max_f=700, max_t=1400):
    """Block-max downsampling for display so 1-bin-wide tonal lines survive rendering."""
    kf = max(1, int(np.ceil(db.shape[0] / max_f)))
    kt = max(1, int(np.ceil(db.shape[1] / max_t)))
    F, T = db.shape[0] // kf * kf, db.shape[1] // kt * kt
    d = db[:F, :T].reshape(F // kf, kf, T // kt, kt).max(axis=(1, 3))
    return d, freqs[:F].reshape(-1, kf).mean(1), times[:T].reshape(-1, kt).mean(1)


def plot_spectrogram(S, path=None, ridges=(), fmax=None, title="Spectrogram", vmin_pct=20, vmax_pct=99.7,
                     ax=None, notches=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(12, 5))
    m = S.freqs <= (fmax or S.freqs[-1])
    db, fr, tm = _maxpool(S.db[m], S.freqs[m], S.times)
    vmin, vmax = np.percentile(db, [vmin_pct, vmax_pct])
    ax.pcolormesh(tm, fr, db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    for r in ridges:
        ax.plot(r.t, r.f, lw=1.2, color="cyan", alpha=0.9)
        ax.text(r.t[0], r.f[0], r.label, color="cyan", fontsize=8)
    if notches is not None:
        t, fn = notches
        ax.plot(t, fn, "w:", lw=0.6, alpha=0.6)
    ax.set_xlabel("time since recording start [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(title)
    if own:
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)


def plot_spectrogram_pair(S_raw, S_clean, ridges, path, fmax=2000):
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    plot_spectrogram(S_raw, ax=axs[0], fmax=fmax, title="Raw spectrogram (dB): tones + Lloyd fringes + 50 Hz hum + noise")
    plot_spectrogram(S_clean, ax=axs[1], ridges=ridges, fmax=fmax, vmin_pct=50,
                     title="Cleaned (hum notched, noise floor & broadband removed) + extracted Doppler ridges")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_doppler_fit(ridges, res, models, t_start, prop, path, truth=None, all_ridges=None):
    from .physics import Observer
    fig, ax = plt.subplots(figsize=(12, 5))
    ob = Observer(res.lat, res.lon, res.h)
    for r, ridge in enumerate(ridges):
        m = models[res.assignment[r]]
        tt = np.linspace(ridge.t[0] - 10, ridge.t[-1] + 10, 400)
        te, g = emission_time(m, ob, t_start + res.dt + tt, prop)
        ax.plot(ridge.t, ridge.f, ".", ms=2, label=f"{ridge.label} measured")
        ax.plot(tt, res.f_src[r] * doppler_factor(g["vr"], g["c"]), "k-", lw=0.8)
        ax.text(ridge.t[-1], ridge.f[-1], f" {ridge.label}->{m.callsign or m.icao24} f_src={res.f_src[r]:.1f} Hz",
                fontsize=8)
    for r in (all_ridges or []):
        if not any(r is q for q in ridges):
            ax.plot(r.t, r.f, "x", ms=2, color="grey", alpha=0.5)
    ax.set_xlabel("time since recording start [s]   (grey x = rejected ridges)")
    ax.set_ylabel("f_obs [Hz]")
    ax.set_title(f"Fitted Doppler curves (black) vs extracted ridges   rms={res.rms_hz:.3f} Hz, clock offset dt={res.dt:+.2f} s")
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _misfit_panel(ax, g, lat0, lon0, label):
    E = geodesy.geodetic_to_enu(g.lat, g.lon, 0 * g.lat, lat0, lon0, 0)
    z = np.log10(np.maximum(g.chi2 / np.nanmin(g.chi2), 1.0))
    z = np.where(np.isfinite(z), z, np.nanmax(z[np.isfinite(z)]))
    im = ax.pcolormesh(E[..., 0] / 1e3, E[..., 1] / 1e3, -z, shading="auto", cmap="viridis")
    L = np.exp(g.loglik - g.loglik.max())
    ax.contour(E[..., 0] / 1e3, E[..., 1] / 1e3, L, levels=[0.05], colors="w", linewidths=0.8)
    plt.colorbar(im, ax=ax, label=label, fraction=0.046)
    return E


def plot_heatmap(res, models, path, truth=None, t_window=None, center=None, zoom_m=None):
    """Left: coarse grid misfit over the whole search area (+ tracks).
    Right: fine grid around the winning solution, 95 % ellipse, truth if known.
    Colour = -log10(chi2/chi2_min) (0 = best); white contour = 5 % likelihood.
    """
    g = res.grid
    lat0, lon0 = center or (float(g.lat.mean()), float(g.lon.mean()))
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    be = geodesy.geodetic_to_enu(res.lat, res.lon, 0, lat0, lon0, 0)
    panels = [(axs[0], g, "coarse grid: -log10(chi2/chi2_min)")]
    if res.fine_grid is not None:
        panels.append((axs[1], res.fine_grid, "fine grid: -log10(chi2/chi2_min)"))
    for k, (ax, gg, lab) in enumerate(panels):
        E = _misfit_panel(ax, gg, lat0, lon0, lab)
        for m in models:
            tt = np.linspace(m.t0, m.t1, 800) if t_window is None else np.linspace(*t_window, 800)
            la, lo, h = m.geodetic(tt)
            pe = geodesy.geodetic_to_enu(la, lo, 0 * la, lat0, lon0, 0)
            ax.plot(pe[:, 0] / 1e3, pe[:, 1] / 1e3, "-", color="orange", lw=1.2)
            ax.annotate(f"{m.callsign or m.icao24}", (pe[-1, 0] / 1e3, pe[-1, 1] / 1e3), color="orange", fontsize=8)
        ax.plot(be[0] / 1e3, be[1] / 1e3, "r+", ms=16, mew=2, label="estimate")
        if k == 0:
            for alt in res.alternatives[:3]:
                ae = geodesy.geodetic_to_enu(alt.lat, alt.lon, 0, lat0, lon0, 0)
                ax.plot(ae[0] / 1e3, ae[1] / 1e3, "mx", ms=9, mew=2, label="other local min (worse)")
        if np.all(np.isfinite(res.cov_en)):
            w, V = np.linalg.eigh(res.cov_en)
            a = np.linspace(0, 2 * np.pi, 100)
            ell = V @ (np.sqrt(np.maximum(w, 0) * 5.991)[:, None] * np.stack([np.cos(a), np.sin(a)]))
            ax.plot((be[0] + ell[0]) / 1e3, (be[1] + ell[1]) / 1e3, "r-", lw=1)
        if truth is not None:
            te = geodesy.geodetic_to_enu(truth.lat, truth.lon, 0, lat0, lon0, 0)
            err = float(geodesy.haversine_m(res.lat, res.lon, truth.lat, truth.lon))
            ax.plot(te[0] / 1e3, te[1] / 1e3, "o", color="white", mfc="none", ms=13, mew=1.5,
                    label=f"truth (hidden), error {err:.1f} m")
        ax.set_xlim(E[..., 0].min() / 1e3, E[..., 0].max() / 1e3)
        ax.set_ylim(E[..., 1].min() / 1e3, E[..., 1].max() / 1e3)
        ax.set_xlabel(f"east [km] from {lat0:.4f},{lon0:.4f}")
        ax.set_ylabel("north [km]")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title("observer-position misfit, whole search area" if k == 0 else
                     f"fine grid around best solution ({res.lat:.5f}, {res.lon:.5f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fringes(fr, pred_t, pred_tau, path, h_r=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(fr.t, fr.tau * 1e3, s=4, c=fr.strength, cmap="viridis", label="cepstral delay (measured)")
    ax.plot(pred_t, pred_tau * 1e3, "r-", lw=1, label="predicted (fitted observer" + (f", h_r={h_r:.2f} m)" if h_r else ")"))
    ax.set_xlabel("time since recording start [s]")
    ax.set_ylabel("ground-reflection delay tau [ms]")
    ax.set_title("Lloyd's-mirror fringe analysis: tau = Delta/c ~ 2 h_r sin(elevation)/c")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
