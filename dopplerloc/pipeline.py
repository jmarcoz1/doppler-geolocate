"""High-level glue used by the CLI and the demo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import audio, cleaning, ridge as ridge_mod


@dataclass
class ExtractConfig:
    sr: int = 8000
    win_length: int = 4096      # 0.51 s @ 8 kHz -> ~2 Hz resolution (zero-padded to 8192)
    hop: int = 400              # 50 ms
    n_fft: int = 8192
    fmin: float = 150.0
    fmax: float = 2500.0
    mains: float = 50.0
    n_ridges: int = 8
    snr_threshold_db: float = 8.0
    min_duration_s: float = 10.0
    max_slope_hz_s: float = 40.0
    reassigned: bool = False


def spectrogram(x, sr, cfg: ExtractConfig, t_start=0.0):
    f = audio.reassigned_spectrogram if cfg.reassigned else audio.stft
    return f(x, sr, win_length=cfg.win_length, hop=cfg.hop, n_fft=cfg.n_fft, t_start=t_start)


def extract(x, sr, cfg: ExtractConfig | None = None, t_start=0.0):
    """waveform -> (raw spectrogram, cleaned line spectrogram, ridges)."""
    cfg = cfg or ExtractConfig()
    if sr != cfg.sr:
        x, sr = audio.resample(x, sr, cfg.sr), cfg.sr
    S_raw = spectrogram(x, sr, cfg, t_start)
    y = cleaning.clean_pipeline(x, sr, fmin=max(20.0, cfg.fmin * 0.5), mains=cfg.mains)
    S = spectrogram(y, sr, cfg, t_start)
    S_floor = cleaning.subtract_noise_floor(S)
    S_line = cleaning.enhance_tonal(S_floor)
    if cfg.mains:
        S_line = cleaning.suppress_stationary_lines(S_line, cfg.mains, 2.5)
    ridges = ridge_mod.extract_ridges(S_line, n_ridges=cfg.n_ridges, fmin=cfg.fmin, fmax=cfg.fmax,
                                      max_slope_hz_s=cfg.max_slope_hz_s, snr_threshold_db=cfg.snr_threshold_db,
                                      min_duration_s=cfg.min_duration_s, refine_db=S.db)
    return S_raw, S_line, ridges, S
