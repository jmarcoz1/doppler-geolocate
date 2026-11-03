"""Doppler ridge (tonal line) extraction with Viterbi dynamic programming.

score[f, t] = dB above local background (from cleaning.enhance_tonal).
For one ridge we maximise   sum_t score[k_t, t] - lam * (k_t - k_{t-1})^2
subject to |k_t - k_{t-1}| <= max_jump bins. After each ridge the cells around
it are masked and the next one is extracted. Frames where the ridge is weak
are flagged by a confidence in [0, 1]; ridges are split into confident
segments, sub-bin refined by parabolic interpolation, and stationary
(constant-frequency) lines are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import Spectrogram


@dataclass
class Ridge:
    t: np.ndarray          # frame times relative to recording start [s]
    f: np.ndarray          # observed frequency [Hz]
    snr_db: np.ndarray     # line strength above background
    conf: np.ndarray       # 0..1
    sigma_f: np.ndarray = field(default=None)  # 1-sigma frequency uncertainty [Hz]
    label: str = ""

    def __len__(self):
        return len(self.t)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(dict(t=self.t, f=self.f, snr_db=self.snr_db, conf=self.conf,
                                 sigma_f=self.sigma_f, ridge=self.label))


def viterbi_ridge(score, max_jump=4, lam=0.5):
    """Best path k_t through score (F, T). Returns int array (T,)."""
    F, T = score.shape
    jumps = np.arange(-max_jump, max_jump + 1)
    pen = lam * jumps.astype(float) ** 2
    acc = score[:, 0].copy()
    back = np.zeros((F, T), dtype=np.int16)
    for t in range(1, T):
        cand = np.full((len(jumps), F), -np.inf)
        for j, d in enumerate(jumps):  # predecessor k - d
            if d >= 0:
                cand[j, d:] = acc[:F - d] if d else acc
            else:
                cand[j, :d] = acc[-d:]
            cand[j] -= pen[j]
        best = np.argmax(cand, axis=0)
        acc = cand[best, np.arange(F)] + score[:, t]
        back[:, t] = jumps[best]
    k = np.empty(T, dtype=int)
    k[-1] = int(np.argmax(acc))
    for t in range(T - 1, 0, -1):
        k[t - 1] = k[t] - back[k[t], t]
    return k


def _parabolic(db, k):
    F = db.shape[0]
    t = np.arange(db.shape[1])
    km, kp = np.clip(k - 1, 0, F - 1), np.clip(k + 1, 0, F - 1)
    a, b, c = db[km, t], db[k, t], db[kp, t]
    den = a - 2 * b + c
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    return np.clip(delta, -0.5, 0.5)


def _snap(ref, k, w):
    F, T = ref.shape
    t = np.arange(T)
    offs = np.arange(-w, w + 1)
    idx = np.clip(k[None, :] + offs[:, None], 0, F - 1)
    return idx[np.argmax(ref[idx, t[None, :]], axis=0), t]


def extract_ridges(score: Spectrogram, n_ridges=6, fmin=100.0, fmax=None, max_jump_hz=None,
                   max_slope_hz_s=40.0, lam=0.3, snr_threshold_db=6.0, mask_hz=12.0,
                   min_duration_s=8.0, min_span_hz=3.0, refine_db: np.ndarray | None = None,
                   gap_fill_s=1.0, max_rough_bins=0.35) -> list[Ridge]:
    """Extract up to n_ridges tonal Doppler curves from a line-enhanced spectrogram."""
    fmax = fmax or score.freqs[-1]
    band = (score.freqs >= fmin) & (score.freqs <= fmax)
    freqs = score.freqs[band]
    sc = score.db[band].copy()           # dB above background, >= 0
    ref = (refine_db[band] if refine_db is not None else sc)
    df, dt = score.df, score.dt
    if max_jump_hz is None:
        max_jump_hz = max_slope_hz_s * dt
    max_jump = max(1, int(np.ceil(max_jump_hz / df)))
    mask_bins = max(1, int(round(mask_hz / df)))
    work = np.minimum(sc, 30.0)          # cap so one loud frame can't drag the path
    ridges = []
    for r in range(n_ridges):
        k = viterbi_ridge(work, max_jump=max_jump, lam=lam)
        tt = np.arange(len(k))
        snr = sc[k, tt]
        # smooth confidence; fill short dropouts
        strong = snr >= snr_threshold_db
        g = max(1, int(round(gap_fill_s / dt)))
        from scipy import ndimage
        strong = ndimage.binary_closing(strong, structure=np.ones(g)) if g > 1 else strong
        strong = ndimage.binary_opening(strong, structure=np.ones(3))
        lab, n = ndimage.label(strong)
        for i in range(1, n + 1):
            sel = lab == i
            if sel.sum() * dt < min_duration_s:
                continue
            # snap to the true local peak of the (uncapped) reference spectrum, then
            # sub-bin parabolic interpolation
            kr = _snap(ref, k, 3)
            kk = kr[sel]
            f = freqs[kk] + _parabolic(ref, kr)[sel] * df
            # noise-following paths are jagged; real Doppler lines are smooth
            if len(f) > 10 and np.median(np.abs(np.diff(f, 2))) > max_rough_bins * df:
                continue
            if np.ptp(f) < min_span_hz:     # stationary line: hum, whine, not an aircraft
                continue
            s = snr[sel]
            conf = np.clip((s - snr_threshold_db) / 12.0 + 0.3, 0, 1)
            # crude Cramer-Rao-like sigma: bin width / sqrt(linear SNR), floored at 0.25 bin
            # (parabolic-interpolation bias, chirp within the window, Lloyd-mirror beating)
            sigma = df * np.maximum(0.25, 1.0 / np.sqrt(10 ** (s / 10.0)))
            ridges.append(Ridge(score.times[sel], f, s, conf, sigma, label=f"R{len(ridges)}"))
        # mask the whole path (confident or not) so next pass finds a new line
        for d in range(-mask_bins, mask_bins + 1):
            kk = np.clip(k + d, 0, len(freqs) - 1)
            work[kk, tt] = np.where(strong, 0.0, work[kk, tt] * 0.5)
    return ridges
