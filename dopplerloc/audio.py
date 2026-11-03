"""Audio I/O, resampling, STFT (dB) and reassigned spectrogram."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal


def load_audio(path, sr: int | None = None, mono: bool = True):
    """Load wav/flac/ogg via soundfile; mp3/m4a/anything else via ffmpeg. Returns (x, sr)."""
    path = Path(path)
    x = None
    try:
        import soundfile as sf
        x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"cannot decode {path}: install soundfile or ffmpeg")
        fs = sr or 16000
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                              "-ar", str(fs), "-"], check=True, capture_output=True).stdout
        x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)[:, None]
    if mono:
        x = x.mean(axis=1)
    if sr and sr != fs:
        x, fs = resample(x, fs, sr), sr
    return x, int(fs)


def save_wav(path, x, sr):
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype="PCM_16")


def resample(x, sr_in, sr_out):
    fr = Fraction(int(sr_out), int(sr_in)).limit_denominator(1000)
    return signal.resample_poly(x, fr.numerator, fr.denominator, axis=0)


@dataclass
class Spectrogram:
    times: np.ndarray      # frame centre times [s] relative to recording start
    freqs: np.ndarray      # [Hz]
    power: np.ndarray      # (F, T) linear power
    sr: int
    n_fft: int
    hop: int
    win_length: int
    t_start: float = 0.0   # absolute unix time of sample 0 (if known)

    @property
    def db(self):
        return 10.0 * np.log10(self.power + 1e-20)

    @property
    def df(self):
        return self.freqs[1] - self.freqs[0]

    @property
    def dt(self):
        return self.times[1] - self.times[0]

    def band(self, fmin=0.0, fmax=np.inf):
        m = (self.freqs >= fmin) & (self.freqs <= fmax)
        return Spectrogram(self.times, self.freqs[m], self.power[m], self.sr, self.n_fft, self.hop,
                           self.win_length, self.t_start)

    def with_power(self, p):
        return Spectrogram(self.times, self.freqs, p, self.sr, self.n_fft, self.hop, self.win_length, self.t_start)


def _frames(x, n, hop):
    pad = n // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(n_frames)[:, None]
    return xp[idx]


def stft(x, sr, win_length=2048, hop=None, n_fft=None, window="hann", t_start=0.0) -> Spectrogram:
    """Magnitude-squared STFT. Defaults: hop = win/8, n_fft = 2*win (zero-padded)."""
    hop = hop or win_length // 8
    n_fft = n_fft or 2 * win_length
    w = signal.get_window(window, win_length)
    fr = _frames(x, win_length, hop) * w
    X = np.fft.rfft(fr, n=n_fft, axis=1).T
    P = (np.abs(X) ** 2) / (w ** 2).sum()
    times = np.arange(P.shape[1]) * hop / sr
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    return Spectrogram(times, freqs, P, sr, n_fft, hop, win_length, t_start)


def reassigned_spectrogram(x, sr, win_length=2048, hop=None, n_fft=None, t_start=0.0):
    """Time-frequency reassigned spectrogram (Auger & Flandrin 1995).

    Uses three STFTs with h (Hann), dh/dt and t*h. Each cell's energy is moved
    to its instantaneous frequency / group delay:
        f_hat = f - Im(X_dh conj(X_h)) / |X_h|^2 * sr / (2 pi)
        t_hat = t + Re(X_th conj(X_h)) / |X_h|^2 / sr
    and re-binned onto the ordinary grid -> much sharper tonal lines.
    """
    hop = hop or win_length // 8
    n_fft = n_fft or 2 * win_length
    n = np.arange(win_length) - (win_length - 1) / 2.0
    h = 0.5 + 0.5 * np.cos(2 * np.pi * n / win_length)
    dh = -np.pi / win_length * np.sin(2 * np.pi * n / win_length)  # dh/dn
    th = n * h
    fr = _frames(x, win_length, hop)
    Xh = np.fft.rfft(fr * h, n_fft, axis=1).T
    Xd = np.fft.rfft(fr * dh, n_fft, axis=1).T
    Xt = np.fft.rfft(fr * th, n_fft, axis=1).T
    P = np.abs(Xh) ** 2
    eps = P.max() * 1e-12
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    times = np.arange(P.shape[1]) * hop / sr
    fhat = freqs[:, None] - np.imag(Xd * np.conj(Xh)) / (P + eps) * sr / (2 * np.pi)
    that = times[None, :] + np.real(Xt * np.conj(Xh)) / (P + eps) / sr
    fi = np.clip(np.rint(fhat / (freqs[1] - freqs[0])).astype(int), 0, len(freqs) - 1)
    ti = np.clip(np.rint(that / (hop / sr)).astype(int), 0, len(times) - 1)
    out = np.zeros_like(P)
    np.add.at(out, (fi.ravel(), ti.ravel()), P.ravel())
    # empty cells after re-binning would be -inf dB; add a small floor (1 % of median power)
    out += 0.01 * np.median(P)
    return Spectrogram(times, freqs, out / (h ** 2).sum(), sr, n_fft, hop, win_length, t_start)
