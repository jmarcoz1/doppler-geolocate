#!/usr/bin/env python
"""Real-world attempt: what can be learned about *where* a recording was made from its audio alone?

    python scripts/real_video_attempt.py URL [--out outputs/real_video] [--adsb-snapshots 12]

Rules followed (see README "Real-video attempt"):
  * only the audio stream is downloaded (yt-dlp, bestaudio); title, description, upload date, tags,
    comments and video frames are NOT read by this script; the video id is the only thing printed;
  * everything below is inferred from the waveform: flyover count/spacing, Doppler-curve fits
    (speed, slant closest-approach distance, CPA time), harmonic structure -> engine/propeller type,
    Lloyd's-mirror fringes -> mic-height x aircraft-altitude product, stereo time-difference sweep
    -> direction of travel, mains hum 50 vs 60 Hz -> grid region;
  * the method needs the recording time and contemporaneous ADS-B to produce a position. Without the
    time there is nothing to match against, so the last step quantifies the ambiguity: how many
    aircraft in global ADS-B snapshots (ADS-B Exchange free samples) fit the audio constraints.
Grading against the video's metadata is a separate, manual step done afterwards (see README).
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, optimize, signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dopplerloc import audio, cleaning, lloyd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_SOUND = 343.0   # m/s, ~20 C at ground; reported sensitivity: v and d scale ~linearly with c


# ----------------------------------------------------------------------------- download
def download_audio(url: str, dest: Path) -> Path:
    """bestaudio only -> 48 kHz stereo wav. No info json, no thumbnails, no metadata printed."""
    dest.mkdir(parents=True, exist_ok=True)
    vid = subprocess.run([sys.executable, "-m", "yt_dlp", "--print", "id", "--skip-download", url],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    wav = dest / f"{vid}_stereo.wav"
    if not wav.exists():
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "-f", "bestaudio", "--no-write-info-json",
                        "--no-write-thumbnail", "--no-embed-metadata", "-o", str(dest / f"{vid}.%(ext)s"), url],
                       check=True)
        src = next(p for p in dest.glob(f"{vid}.*") if p.suffix not in (".wav", ".part"))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ar", "48000", "-ac", "2",
                        "-map_metadata", "-1", str(wav)], check=True)
    print(f"[audio] video id {vid} -> {wav}")
    return wav


# ----------------------------------------------------------------------------- flyovers
def detect_flyovers(x, sr, min_prom_db=20.0, half=60.0):
    y = signal.sosfiltfilt(signal.butter(4, [100, 1500], "bandpass", fs=sr, output="sos"), x)
    e = 10 * np.log10(np.convolve(y ** 2, np.ones(2 * sr) / (2 * sr), "same")[::sr] + 1e-15)
    pk, pr = signal.find_peaks(e, prominence=min_prom_db, distance=90)
    ev = [dict(t_peak=float(p), prominence_db=float(q), t0=float(max(0, p - half)), t1=float(min(len(e) - 1, p + half)))
          for p, q in zip(pk, pr["prominences"])]
    weak, wq = signal.find_peaks(e, prominence=8, distance=60)
    return ev, e, [(float(p), float(q)) for p, q in zip(weak, wq["prominences"]) if q < min_prom_db]


# ----------------------------------------------------------------------------- flyby model
def flyby_doppler(t_obs, v, d, tc, c=C_SOUND):
    """Straight, level, constant-speed pass; slant CPA distance d, CPA (emission) time tc.

    Retarded time: t_obs = t_e + R(t_e)/c  (closed form for a straight line).  D = c / (c + v_r).
    """
    t_obs = np.asarray(t_obs, float)
    # solve (t_obs - t_e) c = sqrt(d^2 + v^2 (t_e - tc)^2) for t_e < t_obs
    a = c * c - v * v
    u = t_obs - tc                      # tau = t_e - tc ; (u - tau)^2 c^2 = d^2 + v^2 tau^2
    b = -2 * c * c * u
    cc = c * c * u * u - d * d
    tau = (-b - np.sqrt(np.maximum(b * b - 4 * a * cc, 0))) / (2 * a)
    R = np.sqrt(d * d + (v * tau) ** 2)
    vr = v * v * tau / R
    return c / (c + vr), R, tau


@dataclass
class FlybyFit:
    v: float
    d: float
    tc: float
    sig_v: float
    sig_d: float
    sig_tc: float
    rms_hz: float
    f_src: list
    ridges_used: list
    n_points: int


def fit_flyby(ridges, t_guess, c=C_SOUND, reject=True, p0=None):
    def resid(p, rs):
        v, d, tc = p
        out = []
        for r in rs:
            D, _, _ = flyby_doppler(r.t, v, d, tc, c)
            w = r.conf / r.sigma_f ** 2
            fs = (w * r.f * D).sum() / (w * D * D).sum()
            out.append((r.f - fs * D) * np.sqrt(r.conf) / r.sigma_f)
        return np.concatenate(out)

    rs = list(ridges)
    best = None
    starts = [(p0[0], p0[1])] if p0 is not None else [(v, d) for v in (35.0, 50.0, 70.0, 100.0) for d in (100.0, 250.0, 600.0)]
    if p0 is not None:
        t_guess = p0[2]
    for v0, d0 in starts:
        if True:
            try:
                s = optimize.least_squares(resid, [v0, d0, t_guess], args=(rs,), loss="soft_l1", f_scale=3.0,
                                           bounds=([10, 20, t_guess - 60], [250, 5000, t_guess + 60]))
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    for _ in range(3 if reject else 0):
        mis = []
        for r in rs:
            z = resid(best.x, [r])
            mis.append(np.sqrt(np.mean(z ** 2)))
        mis = np.array(mis)
        keep = mis <= max(2.5 * np.median(mis), 1.5)
        if keep.all() or keep.sum() < 2:
            break
        rs = [r for r, k in zip(rs, keep) if k]
        best = optimize.least_squares(resid, best.x, args=(rs,), loss="soft_l1", f_scale=3.0,
                                      bounds=([10, 20, t_guess - 60], [250, 5000, t_guess + 60]))
    z = resid(best.x, rs)
    J = best.jac
    dof = max(1, len(z) - 3 - len(rs))
    cov = np.linalg.pinv(J.T @ J) * float(np.sum(z ** 2)) / dof
    # STFT frames overlap (win/hop = 20): inflate formal sigmas by sqrt(20)
    sig = np.sqrt(np.diag(cov)) * np.sqrt(20.0)
    v, d, tc = best.x
    fsrc, res_hz = [], []
    for r in rs:
        D, _, _ = flyby_doppler(r.t, v, d, tc, c)
        w = r.conf / r.sigma_f ** 2
        fs = float((w * r.f * D).sum() / (w * D * D).sum())
        fsrc.append(fs)
        res_hz.append(r.f - fs * D)
    rms = float(np.sqrt(np.mean(np.concatenate(res_hz) ** 2)))
    return FlybyFit(float(v), float(d), float(tc), *map(float, sig), rms, fsrc, [r.label for r in rs],
                    int(sum(len(r) for r in rs))), rs


def _comb_count(fsrc, F, tol_hz=1.2, nmax=40):
    n = np.round(fsrc / F)
    return int(((n >= 1) & (n <= nmax) & (np.abs(fsrc - n * F) < tol_hz)).sum())


def harmonic_fundamental(fsrc, fmin=18.0, fmax=140.0):
    """Rest-frame tone analysis.

    Returns (F_main, harmonic numbers, n matched, F_sub, n matched by F_sub):
      F_main = largest comb spacing that explains >= half of the tones (the dominant harmonic family,
               e.g. propeller blade-pass / firing frequency);
      F_sub  = largest spacing that explains >= 90 % of the tones (e.g. engine half-order / shaft rate).
    """
    fsrc = np.asarray(sorted(fsrc))
    grid = np.arange(fmin, fmax, 0.02)
    cnt = np.array([_comb_count(fsrc, F) for F in grid])

    def pick(frac):
        ok = grid[cnt >= frac * len(fsrc)]
        if not len(ok):
            return None
        F0 = ok.max()
        n = np.round(fsrc / F0)
        m = np.abs(fsrc - n * F0) < 1.2
        return float((fsrc[m] * n[m]).sum() / (n[m] ** 2).sum())
    Fm = pick(0.5) or float(grid[np.argmax(cnt)])
    Fs = pick(0.9) or Fm
    harm = np.round(fsrc / Fm).astype(int)
    return Fm, harm.tolist(), _comb_count(fsrc, Fm), Fs, _comb_count(fsrc, Fs)


# ----------------------------------------------------------------------------- Doppler de-warping
def line_spectrogram(x4, a, b, sr4=4000):
    S = audio.stft(x4[int(a * sr4):int(b * sr4)], sr4, win_length=4096, hop=200, n_fft=8192)
    Lf = cleaning.enhance_tonal(cleaning.subtract_noise_floor(S), freq_width_hz=25)
    return S.with_power(Lf.power), S.times + a


def dewarp_objective(p, t, f, L, fr=np.arange(45, 900, 0.5)):
    """Undo the Doppler shift D(t) of a trial pass and integrate over time: when (v, d, t_CPA) are right,
    every tonal line (all harmonics, any engine/propeller order) collapses into a sharp stationary peak."""
    v, d, tc = p
    if not (15 < v < 150 and 20 < d < 5000):
        return 0.0, fr, np.zeros_like(fr)
    D, _, _ = flyby_doppler(t, v, d, tc)
    idx = (fr[:, None] * D[None, :] - f[0]) / (f[1] - f[0])
    i0 = np.clip(idx.astype(int), 0, len(f) - 2)
    w = idx - i0
    cols = np.arange(len(t))[None, :]
    V = L[i0, cols] * (1 - w) + L[i0 + 1, cols] * w
    Srest = np.nan_to_num(V.mean(1))
    pk = np.maximum(Srest - ndimage.median_filter(Srest, 31), 0)
    return -float(np.sum(pk ** 2)), fr, Srest


def dewarp_fit(Sl, t, t_guess, dec=8):
    m = (Sl.freqs >= 40) & (Sl.freqs <= 1000)
    f, L = Sl.freqs[m], np.minimum(Sl.db[m], 20.0)[:, ::dec]
    td = t[::dec]
    best = (0.0, None)
    for v in np.arange(25, 121, 5.0):
        for d in (60, 100, 150, 220, 330, 500, 750, 1100, 1600, 2400):
            for tc in np.arange(t_guess - 15, t_guess + 15.1, 1.0):
                o = dewarp_objective((v, d, tc), td, f, L)[0]
                if o < best[0]:
                    best = (o, (v, d, tc))
    s = optimize.minimize(lambda p: dewarp_objective(p, td, f, L)[0], best[1], method="Nelder-Mead",
                          options=dict(xatol=0.05, fatol=1e-3, maxiter=800))
    _, fr, Sr = dewarp_objective(s.x, td, f, L)
    return s.x, fr, Sr


def comb_ridges(Sl, t, p, fr, Sr, n_lines=16, snr_db=6.0, half_hz=3.0):
    """Ridges = local spectral peaks that follow f_k * D(t) for each rest-frame line f_k."""
    from dopplerloc.ridge import Ridge
    v, d, tc = p
    D, _, _ = flyby_doppler(t, v, d, tc)
    pk, _ = signal.find_peaks(Sr - ndimage.median_filter(Sr, 31), distance=8)
    pk = pk[np.argsort((Sr - ndimage.median_filter(Sr, 31))[pk])[::-1][:n_lines]]
    db = Sl.db
    df = Sl.df
    hb = max(1, int(round(half_hz / df)))
    ridges = []
    for fk in sorted(fr[pk]):
        k0 = np.round(fk * D / df).astype(int)
        cols = np.arange(len(t))
        win = np.stack([db[np.clip(k0 + j, 1, db.shape[0] - 2), cols] for j in range(-hb, hb + 1)])
        j = np.argmax(win, 0) - hb
        k = np.clip(k0 + j, 1, db.shape[0] - 2)
        y0, y1, y2 = db[k - 1, cols], db[k, cols], db[k + 1, cols]
        dk = np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-9), -0.5, 0.5)
        ok = (y1 >= snr_db) & (np.abs(j) < hb)
        ok = ndimage.binary_opening(ok, np.ones(5))
        if ok.sum() < 40:
            continue
        s = y1[ok]
        sig = df * np.maximum(0.25, 1.0 / np.sqrt(10 ** (s / 10.0)))
        conf = np.clip((s - snr_db) / 12.0 + 0.3, 0, 1)
        ridges.append(Ridge(t[ok], (k[ok] + dk[ok]) * df, s, conf, sig, label=f"{fk:.1f}Hz"))
    return ridges


# ----------------------------------------------------------------------------- Lloyd fringes
def fringe_track(x8, sr8, t0, t1):
    S = audio.stft(x8[int(t0 * sr8):int(t1 * sr8)], sr8, win_length=1024, hop=400, n_fft=2048)
    P = ndimage.median_filter(S.db, size=(9, 1))        # remove narrow tonal lines, keep broadband comb
    fr = lloyd.cepstral_delay(S.with_power(10 ** (P / 10)), tau_min=1.2e-3, tau_max=12e-3, fmin=300,
                              fmax=3800, smooth_frames=5, min_strength=0)
    return fr.t + t0, fr.tau, fr.strength


def fit_fringes(t, tau, strength, fit: FlybyFit, c=C_SOUND, min_strength=4.5, half_window_s=25.0):
    """tau(t) ~= 2 h_r H / (c R(t_e))  ->  only the product h_r*H is identifiable.

    Only confident cepstral peaks near the pass (|t - t_CPA,obs| < half_window_s, tau > 1.6 ms to stay
    clear of the tau_min edge) are used; robust (soft-L1) fit of the single parameter h_r*H.
    """
    t_cpa_obs = fit.tc + fit.d / c
    sel = (strength >= min_strength) & (np.abs(t - t_cpa_obs) < half_window_s) & (tau > 1.6e-3)
    if sel.sum() < 8:
        return None
    t, tau, st = t[sel], tau[sel], strength[sel]
    _, R, _ = flyby_doppler(t, fit.v, fit.d, fit.tc, c)
    w = np.sqrt(st / st.mean())

    def res(p):
        return w * (tau - 2 * p[0] / (c * R)) / 3e-4
    best = None
    for p0 in (100.0, 300.0, 800.0, 2000.0):
        s = optimize.least_squares(res, [p0], loss="cauchy", f_scale=1.0, bounds=([1.0], [2e4]))
        if best is None or s.cost < best.cost:
            best = s
    inl = np.abs(res(best.x)) < 3
    z = res(best.x)[inl]
    J = best.jac[inl]
    cov = np.linalg.pinv(J.T @ J) * max(1.0, float(np.mean(z ** 2)))
    P = float(best.x[0])
    return dict(hrH_m2=P, sig_hrH_m2=float(np.sqrt(cov[0, 0])), n=int(inl.sum()),
                tau_max_ms=float(2 * P / (c * fit.d) * 1e3), hr_min_m=float(P / fit.d),   # since H <= d
                t=t[inl].tolist(), tau=tau[inl].tolist())


# ----------------------------------------------------------------------------- stereo
def stereo_lag_track(L, R, sr, t0, t1, win=1.0, step=0.5, max_lag_ms=1.2):
    sos = signal.butter(4, [200, 3000], "bandpass", fs=sr, output="sos")
    a, b = int(t0 * sr), int(t1 * sr)
    Lb, Rb = signal.sosfiltfilt(sos, L[a:b]), signal.sosfiltfilt(sos, R[a:b])
    n, h, ml = int(win * sr), int(step * sr), int(max_lag_ms * 1e-3 * sr)
    out = []
    for i in range(0, len(Lb) - n, h):
        l, r = Lb[i:i + n], Rb[i:i + n]
        cc = signal.correlate(l, r, "full", method="fft")
        lags = np.arange(-n + 1, n)
        m = np.abs(lags) <= ml
        k = np.argmax(cc[m])
        y0, y1, y2 = cc[m][max(k - 1, 0)], cc[m][k], cc[m][min(k + 1, m.sum() - 1)]
        dk = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
        rho = y1 / np.sqrt((l ** 2).sum() * (r ** 2).sum() + 1e-30)
        out.append((t0 + (i + n / 2) / sr, (lags[m][k] + dk) / sr * 1e3, rho))
    return np.array(out)


# ----------------------------------------------------------------------------- mains hum
def mains_hum_test(x, sr, quiet_spans, nh=6):
    """Prominence of exactly-stationary lines at k*50 vs k*60 Hz in quiet stretches (0.02 Hz res)."""
    res = {}
    for f0 in (50.0, 60.0, 45.0, 55.0, 65.0):
        proms = []
        for (a, b) in quiet_spans:
            seg = x[int(a * sr):int(b * sr)]
            n = len(seg)
            X = np.abs(np.fft.rfft(seg * np.hanning(n), 4 * n))
            f = np.fft.rfftfreq(4 * n, 1 / sr)
            for k in range(1, nh + 1):
                fk = k * f0
                m = np.abs(f - fk) < 0.3
                bg = np.median(X[(np.abs(f - fk) > 1) & (np.abs(f - fk) < 6)])
                proms.append(20 * np.log10(X[m].max() / bg))
        res[f"{int(f0)}Hz"] = dict(median_prominence_db=float(np.median(proms)), max_prominence_db=float(np.max(proms)))
    ctrl = float(np.mean([res[k]["median_prominence_db"] for k in ("45Hz", "55Hz", "65Hz")]))
    e50, e60 = res["50Hz"]["median_prominence_db"] - ctrl, res["60Hz"]["median_prominence_db"] - ctrl
    res["excess_over_control_db"] = dict(f50=e50, f60=e60, control_db=ctrl)
    verdict = ("no mains hum above the control families (45/55/65 Hz) -> no grid-region (50 vs 60 Hz) hint; "
               "consistent with a battery recorder outdoors" if max(e50, e60) < 3
               else ("50 Hz grid (Europe/Africa/most of Asia/Oceania)" if e50 > e60 + 3 else
                     "60 Hz grid (Americas, parts of Asia)" if e60 > e50 + 3 else "inconclusive"))
    res["verdict"] = verdict
    return res


# ----------------------------------------------------------------------------- ADS-B ambiguity
def adsb_ambiguity(classes, out_dir: Path, n_snap=12, day="2025/10/01"):
    """Count aircraft in global ADS-B Exchange sample snapshots that match each audio-derived class.

    classes: {name: (v_min, v_max [m/s], alt_max_ft)}. Also counts 1x1 degree cells where every
    class is matched in the same snapshot (a crude 'scene' match: all aircraft types heard, same area).
    """
    import requests
    rows, snaps = [], []
    for k in range(n_snap):
        hh = int(24 * k / n_snap)
        name = f"{hh:02d}0000Z.json.gz"
        url = f"https://samples.adsbexchange.com/readsb-hist/{day}/{name}"
        cache = out_dir / "adsbx_cache" / day.replace("/", "-") / name
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not cache.exists():
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                cache.write_bytes(r.content)
            raw = cache.read_bytes()
            js = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        except Exception as e:
            print(f"[adsb] {url}: {e}")
            continue
        n_air = 0
        cells = {c: set() for c in classes}
        cnt = {c: 0 for c in classes}
        for ac in js.get("aircraft", []):
            alt, gs = ac.get("alt_baro"), ac.get("gs")
            if ac.get("lat") is None or gs is None or alt in (None, "ground"):
                continue
            n_air += 1
            v = gs * 0.514444
            for c, (v0, v1, amax) in classes.items():
                if v0 <= v <= v1 and alt <= amax:
                    cnt[c] += 1
                    cells[c].add((int(np.floor(ac["lat"])), int(np.floor(ac["lon"]))))
                    rows.append((hh, c, ac["lat"], ac["lon"]))
        joint = set.intersection(*cells.values()) if cells else set()
        snaps.append(dict(hour_utc=hh, airborne=n_air, matches=cnt, cells_with_all_classes=len(joint)))
        print(f"[adsb] {day} {hh:02d}Z: {n_air} airborne; matches {cnt}; 1-deg cells with every class: {len(joint)}")
    return snaps, rows


def plot_local_geometry(fl, hr_lo, hr_hi, path):
    """Plan view in a microphone-centred frame. Absolute azimuth is unknown from audio; the stereo
    delay sweep only fixes the direction of travel along the (unknown) mic axis, and the side of the
    track (front/back of the mic pair) stays ambiguous, so each pass is drawn on both sides."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    cols = ["tab:blue", "tab:orange", "tab:green"]
    for i, f in enumerate(fl):
        if not f.get("lateral_offset_m_range"):
            continue
        d = f["fit"]["d"]
        lo, hi = f["lateral_offset_m_range"]
        sgn = 1 if (f.get("stereo") or {}).get("direction", "L->R") == "L->R" else -1
        for side in (1, -1):
            for y, ls in ((lo, "-"), (hi, "--")):
                axs[0].annotate("", xy=(sgn * 2500, side * y), xytext=(-sgn * 2500, side * y),
                                arrowprops=dict(arrowstyle="->", color=cols[i % 3], ls=ls, lw=1.3, alpha=0.9 if side > 0 else 0.35))
        axs[0].text(-2400, max(hi, 60) + 40, f"pass {i+1}: {f['fit']['v']:.0f} m/s, d={d:.0f} m", color=cols[i % 3], fontsize=8)
        H = f["altitude_agl_m_range"]
        axs[1].plot([lo, hi], [H[1], H[0]], "-", color=cols[i % 3], lw=3, label=f"pass {i+1}")
        th = np.linspace(0, np.pi / 2, 100)
        axs[1].plot(d * np.cos(th), d * np.sin(th), ":", color=cols[i % 3], lw=0.8)
    axs[0].plot(0, 0, "k^", ms=10)
    axs[0].set_xlim(-2600, 2600)
    axs[0].set_ylim(-2300, 2300)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("along stereo-mic axis [m] (azimuth of axis unknown)")
    axs[0].set_ylabel("lateral offset [m] (side ambiguous: faint = mirror)")
    axs[0].set_title("Plan view: track offsets and travel direction (solid = h_r min, dashed = h_r max)")
    axs[1].plot(0, 0, "k^", ms=10)
    axs[1].set_xlabel("lateral offset of track [m]")
    axs[1].set_ylabel("altitude above mic [m]")
    axs[1].set_title(f"Side view: altitude/offset trade-off for mic height {hr_lo:.2f}-{hr_hi:.1f} m\n(dotted = slant CPA distance from Doppler)")
    axs[1].set_aspect("equal")
    axs[1].set_xlim(0, 2200)
    axs[1].set_ylim(0, 2200)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "real_video"))
    ap.add_argument("--data", default=str(ROOT / "data" / "real_video"))
    ap.add_argument("--adsb-snapshots", type=int, default=12)
    ap.add_argument("--hr-max", type=float, default=2.0, help="largest plausible mic height [m]")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wav = download_audio(args.url, Path(args.data))
    xs, sr = sf.read(wav, dtype="float64")
    L, R = xs[:, 0], xs[:, 1]
    x = xs.mean(1)
    x4 = signal.resample_poly(x, 1, 12)          # 4 kHz for tonal analysis
    x8 = signal.resample_poly(x, 1, 6)           # 8 kHz for fringes
    dur = len(x) / sr
    print(f"[audio] {dur:.1f} s, {sr} Hz, 2 ch")
    events, env, weak = detect_flyovers(x4, 4000)
    print(f"[flyovers] {len(events)} loud passes at t = {[round(e['t_peak']) for e in events]} s; "
          f"weaker events {[round(w[0]) for w in weak]}")

    summary = dict(duration_s=dur, n_loud_flyovers=len(events), flyovers=[], weak_events=weak)
    fig_s, axs_s = plt.subplots(len(events), 1, figsize=(13, 4.2 * len(events)), squeeze=False)
    fig_d, axs_d = plt.subplots(len(events), 2, figsize=(14, 4.0 * len(events)), squeeze=False)
    for i, ev in enumerate(events):
        a, b = ev["t0"], ev["t1"]
        a, b = ev["t_peak"] - 35, ev["t_peak"] + 35     # Doppler-dominated part of the pass
        S_line, tS = line_spectrogram(x4, a, b)
        p_dw, fr_rest, S_rest = dewarp_fit(S_line, tS, ev["t_peak"])
        ridges = comb_ridges(S_line, tS, p_dw, fr_rest, S_rest)
        fit, used = fit_flyby(ridges, ev["t_peak"], p0=p_dw)
        print(f"[flyover {i+1}] de-warp start v={p_dw[0]:.1f} d={p_dw[1]:.0f} tc={p_dw[2]:.1f}; {len(ridges)} comb ridges")
        F, harm, nmatch, F_sub, n_sub = harmonic_fundamental(fit.f_src)
        # c sensitivity (0..35 C)
        sens = {}
        for cc in (331.3 + 0.606 * 0, 331.3 + 0.606 * 35):
            f2, _ = fit_flyby(used, fit.tc, c=cc, reject=False, p0=(fit.v, fit.d, fit.tc))
            sens[f"c={cc:.0f}"] = dict(v=f2.v, d=f2.d)
        ft, ftau, fst = fringe_track(x8, 8000, ev['t0'], ev['t1'])
        fr = fit_fringes(ft, ftau, fst, fit)
        lag = stereo_lag_track(L, R, sr, a, b)
        strong = lag[:, 2] > 0.6
        lag_s = lag[strong]
        sweep = None
        if len(lag_s) > 5:
            pre = lag_s[lag_s[:, 0] < fit.tc - 5, 1]
            post = lag_s[lag_s[:, 0] > fit.tc + 5, 1]
            if len(pre) and len(post):
                sweep = dict(lag_before_ms=float(np.median(pre)), lag_after_ms=float(np.median(post)),
                             direction="L->R" if np.median(pre) > np.median(post) else "R->L")
        rec = dict(window_s=[a, b], dewarp_start=list(map(float, p_dw)), rest_frame_peaks_hz=[float(r.label[:-2]) for r in ridges], t_peak_s=ev["t_peak"], peak_prominence_db=ev["prominence_db"],
                   fit=asdict(fit), speed_kt=fit.v / 0.514444, c_sensitivity=sens,
                   fundamental_hz=F, harmonic_numbers=harm, n_harmonic_matches=nmatch, n_tones=len(fit.f_src),
                   sub_comb_hz=F_sub, n_sub_comb_matches=n_sub,
                   rpm_if_2blade=F * 60 / 2, rpm_if_3blade=F * 60 / 3,
                   fringes=None if fr is None else {k: v for k, v in fr.items() if k not in ("t", "tau")},
                   stereo=sweep)
        summary["flyovers"].append(rec)
        print(f"[flyover {i+1}] t_cpa={fit.tc:.1f}s v={fit.v:.1f}±{fit.sig_v:.1f} m/s ({fit.v/0.514444:.0f} kt) "
              f"slant CPA d={fit.d:.0f}±{fit.sig_d:.0f} m rms={fit.rms_hz:.2f} Hz, {len(used)}/{len(ridges)} ridges; "
              f"main comb F={F:.2f} Hz ({nmatch}/{len(fit.f_src)} tones), sub-comb {F_sub:.2f} Hz ({n_sub}); fringes {None if fr is None else round(fr['hrH_m2'],1)} m^2; stereo {sweep}")
        # ---- figures
        ax = axs_s[i, 0]
        m = S_line.freqs <= 1000
        ax.pcolormesh(tS, S_line.freqs[m], S_line.db[m], shading="auto", cmap="magma", vmin=0, vmax=20)
        for r in ridges:
            ax.plot(r.t, r.f, lw=0.8, color="cyan" if any(r is u for u in used) else "0.6", alpha=0.9)
        tt = np.linspace(a, b, 600)
        D, _, _ = flyby_doppler(tt, fit.v, fit.d, fit.tc)
        for fs in fit.f_src:
            ax.plot(tt, fs * D, "w--", lw=0.6, alpha=0.7)
        ax.set_title(f"Flyover {i+1}: line-enhanced spectrogram, comb ridges (cyan = used, grey = rejected), "
                     f"straight-pass fit (dashed): v={fit.v:.0f} m/s, d_CPA={fit.d:.0f} m")
        ax.set_ylabel("Hz")
        ax.set_xlabel("time in recording [s]")
        ax = axs_d[i, 0]
        for r, fs in zip(used, fit.f_src):
            ax.plot(r.t - fit.tc, r.f / fs, ".", ms=1.5)
        ax.plot(tt - fit.tc, D, "k-", lw=1.2, label=f"fit v={fit.v:.1f} m/s, d={fit.d:.0f} m")
        ax.set_xlabel("t - t_CPA [s]")
        ax.set_ylabel("f_obs / f_src")
        ax.legend(fontsize=8)
        ax.set_title(f"Flyover {i+1}: all harmonics collapse onto one Doppler factor")
        ax = axs_d[i, 1]
        ax.scatter(ft, ftau * 1e3, c=fst, s=4, cmap="viridis", vmin=2, vmax=8)
        if fr is not None:
            ax.plot(fr["t"], np.array(fr["tau"]) * 1e3, "o", mfc="none", mec="r", ms=4, lw=0.5)
            _, Rr, _ = flyby_doppler(tt, fit.v, fit.d, fit.tc)
            ax.plot(tt, 2 * fr["hrH_m2"] / (C_SOUND * Rr) * 1e3, "r-",
                    label=f"h_r·H = {fr['hrH_m2']:.0f} m² (h_r ≥ {fr['hr_min_m']:.2f} m)")
            ax.legend(fontsize=8)
        ax.set_ylim(0, 10)
        ax.set_xlabel("time in recording [s]")
        ax.set_ylabel("ground-reflection delay τ [ms]")
        ax.set_title("Lloyd's-mirror fringe delay (cepstrum)")
    fig_s.tight_layout()
    fig_s.savefig(out / "spectrogram_ridges.png", dpi=110)
    fig_d.tight_layout()
    fig_d.savefig(out / "doppler_and_fringe_fits.png", dpi=110)
    plt.close("all")

    # ---- joint mic height / altitude solution
    fl = summary["flyovers"]
    # mic-height lower bound from passes with a well-sampled fringe track only (>= 20 cepstral inliers)
    good = [f for f in fl if f["fringes"] and f["fringes"]["n"] >= 20]
    for f in fl:
        if f["fringes"]:
            f["fringes"]["reliable"] = f in good
    hr_min = max([f["fringes"]["hr_min_m"] for f in (good or [f for f in fl if f["fringes"]])] or [np.nan])
    for f in fl:
        if f["fringes"]:
            P, d = f["fringes"]["hrH_m2"], f["fit"]["d"]
            H_hi, H_lo = P / hr_min, P / args.hr_max
            f["altitude_agl_m_range"] = [min(H_lo, d), min(H_hi, d)]
            f["lateral_offset_m_range"] = [float(np.sqrt(max(d * d - min(H_hi, d) ** 2, 0))),
                                           float(np.sqrt(max(d * d - H_lo ** 2, 0)))]
            f["elevation_at_cpa_deg_range"] = [float(np.degrees(np.arcsin(min(1, H_lo / d)))),
                                               float(np.degrees(np.arcsin(min(1, H_hi / d))))]
    summary["mic_height_m_min"] = hr_min
    summary["mic_height_assumed_max_m"] = args.hr_max
    tcs = [f["fit"]["tc"] for f in fl]
    summary["flyover_spacing_s"] = np.diff(tcs).tolist()
    Fs = [f["fundamental_hz"] for f in fl]
    summary["fundamentals_hz"] = Fs

    # ---- mains hum (quiet spans = lowest-envelope 20 s windows)
    k = 20
    lev = np.array([env[i:i + k].mean() for i in range(0, len(env) - k, k)])
    quiet = [(i * k, i * k + k) for i in np.argsort(lev)[:4]]
    summary["mains_hum"] = mains_hum_test(x4, 4000, quiet)
    print("[hum]", summary["mains_hum"])

    # ---- stereo figure
    fig, ax = plt.subplots(figsize=(12, 3.5))
    lag = stereo_lag_track(L, R, sr, 0, dur - 1, win=1.0, step=1.0)
    ax.scatter(lag[:, 0], lag[:, 1], c=lag[:, 2], s=6, cmap="viridis", vmin=0.2, vmax=1)
    for f in fl:
        ax.axvline(f["fit"]["tc"], color="r", lw=0.8)
    ax.set_xlabel("time in recording [s]")
    ax.set_ylabel("L-R delay [ms]")
    ax.set_title("Stereo inter-channel delay (colour = coherence); red = Doppler CPA times")
    fig.tight_layout()
    fig.savefig(out / "stereo_tdoa.png", dpi=110)
    plt.close(fig)

    plot_local_geometry(fl, hr_min, args.hr_max, out / "local_geometry.png")

    # ---- ADS-B ambiguity: global snapshots (passes grouped by engine signature)
    classes = {}
    for i, f in enumerate(fl):
        key = f"pass{i+1}"
        H = (f.get("altitude_agl_m_range") or [0, 900])[1]
        classes[key] = (0.85 * f["fit"]["v"], 1.15 * f["fit"]["v"], H / 0.3048 + 2000.0)  # + terrain/QNH slack
    snaps, rows = adsb_ambiguity(classes, out, n_snap=args.adsb_snapshots) if args.adsb_snapshots else ([], [])
    summary["adsb_ambiguity"] = dict(source="ADS-B Exchange free samples (readsb-hist, global, 2025-10-01)",
                                     classes=classes, snapshots=snaps)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["hour", "cls", "lat", "lon"])
        fig, ax = plt.subplots(figsize=(13, 6.5))
        _coastlines(ax)
        for c, col in zip(classes, ["tab:blue", "tab:orange", "tab:green"]):
            q = df[df.cls == c]
            ax.scatter(q.lon, q.lat, s=1, c=col, alpha=0.4, label=f"{c}-like: {len(q)}")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 75)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(alpha=0.3)
        ax.legend(markerscale=8, loc="lower left")
        nj = int(np.mean([s["cells_with_all_classes"] for s in snaps]))
        ax.set_title(f"Aircraft in {len(snaps)} global ADS-B snapshots (2025-10-01) matching the audio-only speed/altitude "
                     f"constraints of each pass\n~{nj} 1°×1° cells per snapshot contain all three at once -> "
                     f"without the recording time the location is not identifiable")
        fig.tight_layout()
        fig.savefig(out / "candidate_map_global.png", dpi=110)
        plt.close(fig)
    (out / "audio_only_result.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {out}")
    return summary


def _coastlines(ax):
    """Natural Earth 110m coastline (public domain) if reachable; silently skipped otherwise."""
    try:
        import requests
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"
        cache = ROOT / "data" / "cache" / "ne_110m_coastline.geojson"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            cache.write_bytes(requests.get(url, timeout=30).content)
        gj = json.loads(cache.read_text())
        for ftr in gj["features"]:
            g = ftr["geometry"]
            lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
            for ln in lines:
                a = np.array(ln)
                ax.plot(a[:, 0], a[:, 1], color="0.4", lw=0.5)
    except Exception as e:
        print("[map] coastline skipped:", e)


if __name__ == "__main__":
    main()
