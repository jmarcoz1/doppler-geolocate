r"""Lloyd's-mirror (ground reflection) fringe analysis.

Microphone at height h_r over reflecting ground receives the direct ray plus
a reflected ray delayed by tau = Delta/c with Delta ~= 2 h_r sin(elevation).
The received broadband spectrum is multiplied by |1 + Q e^{-i 2 pi f tau}|^2:
a comb of notches at f_n = (n + 1/2)/tau (hard ground). As the aircraft
passes, elevation changes -> fringes sweep (classic 'interference fringes' in
flyover spectrograms). tau(t) is measured from the power cepstrum
(log-spectrum periodicity) and encodes the elevation angle history, i.e. an
independent constraint on observer-track geometry and on h_r.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .audio import Spectrogram


@dataclass
class FringeTrack:
    t: np.ndarray
    tau: np.ndarray       # [s]
    strength: np.ndarray  # cepstral peak prominence


def cepstral_delay(S: Spectrogram, tau_min=0.4e-3, tau_max=12e-3, fmin=100.0, fmax=None,
                   smooth_frames=5, min_strength=3.0) -> FringeTrack:
    fmax = fmax or S.freqs[-1] * 0.95
    m = (S.freqs >= fmin) & (S.freqs <= fmax)
    logp = np.log(S.power[m] + 1e-20)
    logp = ndimage.uniform_filter1d(logp, smooth_frames, axis=1)
    logp = logp - logp.mean(axis=0, keepdims=True)
    # remove slow spectral tilt (quefrencies << tau_min) with a detrend along f
    n = logp.shape[0]
    nfft = 8 * n
    ceps = np.abs(np.fft.rfft(logp * np.hanning(n)[:, None], nfft, axis=0))
    df = S.df
    q = np.fft.rfftfreq(nfft, d=df)  # quefrency axis [s]
    sel = (q >= tau_min) & (q <= tau_max)
    c = ceps[sel]
    k = np.argmax(c, axis=0)
    strength = c[k, np.arange(c.shape[1])] / (np.median(c, axis=0) + 1e-12)
    tau = q[sel][k]
    ok = strength >= min_strength
    return FringeTrack(S.times[ok], tau[ok], strength[ok])
