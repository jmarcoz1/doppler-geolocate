"""Pre-processing: filters, mains-hum removal, noise-floor subtraction, line enhancement."""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .audio import Spectrogram


def bandpass(x, sr, fmin=None, fmax=None, order=4):
    """Zero-phase Butterworth high/low/band-pass."""
    if fmin and fmax:
        sos = signal.butter(order, [fmin, fmax], "bandpass", fs=sr, output="sos")
    elif fmin:
        sos = signal.butter(order, fmin, "highpass", fs=sr, output="sos")
    elif fmax:
        sos = signal.butter(order, fmax, "lowpass", fs=sr, output="sos")
    else:
        return x
    return signal.sosfiltfilt(sos, x)


def remove_mains_hum(x, sr, f0=50.0, n_harmonics=20, bw_hz=1.5, track_f0=True):
    """Notch out mains hum (50 Hz in Spain/EU) and harmonics.

    If track_f0, the actual grid frequency (49.9-50.1 Hz) is estimated from the
    strongest low harmonic first so the narrow notches stay centred.
    Caveat: an aircraft tone that sits on k*50 Hz for a long plateau is notched
    too - keep bw_hz small (default 1.5 Hz).
    """
    if track_f0:
        f0 = estimate_mains_f0(x, sr, f0)
    y = x
    for k in range(1, n_harmonics + 1):
        fk = k * f0
        if fk >= sr / 2 * 0.95:
            break
        b, a = signal.iirnotch(fk, fk / bw_hz, fs=sr)  # constant ~1.5 Hz -3 dB bandwidth
        y = signal.filtfilt(b, a, y)
    return y


def estimate_mains_f0(x, sr, f0=50.0, search=0.3):
    n = min(len(x), sr * 30)
    X = np.abs(np.fft.rfft(x[:n] * np.hanning(n), 8 * n))
    f = np.fft.rfftfreq(8 * n, 1 / sr)
    best, best_val = f0, -1
    for k in (1, 2, 3):
        m = (f > k * (f0 - search)) & (f < k * (f0 + search))
        if m.any():
            i = np.argmax(X[m])
            if X[m][i] > best_val:
                best_val, best = X[m][i], f[m][i] / k
    return best


def subtract_noise_floor(S: Spectrogram, percentile=20.0) -> Spectrogram:
    """Per-frequency noise floor (low percentile over time) removal in the dB domain.

    A low percentile (20 %) rather than the median is used so that a Doppler
    plateau occupying the same bins for a large part of the recording is not
    mistaken for background.

    Kills stationary lines (hum remnants, fridge whine, insects at fixed f) and
    whitens the background. Result: dB above floor, clipped at 0.
    """
    db = S.db
    floor = np.percentile(db, percentile, axis=1, keepdims=True)
    return S.with_power(10 ** (np.maximum(db - floor, 0.0) / 10.0))


def enhance_tonal(S: Spectrogram, freq_width_hz=40.0, time_smooth_frames=3) -> Spectrogram:
    """Narrow-band line enhancement.

    Subtract a running median *across frequency* (width >> line width) so that
    broadband energy (jet noise, Lloyd's-mirror fringes, wind) is removed and
    only narrow tonal lines remain; then a short median along time keeps
    continuous lines and suppresses clicks.
    """
    db = S.db
    k = max(3, int(round(freq_width_hz / S.df)) | 1)
    bg = ndimage.median_filter(db, size=(k, 1), mode="nearest")
    line = np.maximum(db - bg, 0.0)
    if time_smooth_frames > 1:
        line = ndimage.median_filter(line, size=(1, time_smooth_frames), mode="nearest")
    return S.with_power(10 ** (line / 10.0))


def suppress_stationary_lines(S: Spectrogram, f0=50.0, width_hz=3.0) -> Spectrogram:
    """Zero out narrow bands around k*f0 in the spectrogram (belt and braces)."""
    p = S.power.copy()
    for k in range(1, int(S.freqs[-1] // f0) + 1):
        p[np.abs(S.freqs - k * f0) < width_hz] = p.min()
    return S.with_power(p)


def clean_pipeline(x, sr, fmin=80.0, fmax=None, mains=50.0):
    """Waveform-domain cleaning: band-pass + mains notches."""
    y = bandpass(x, sr, fmin, fmax)
    if mains:
        y = remove_mains_hum(y, sr, mains)
    return y
