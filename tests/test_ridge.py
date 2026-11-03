import numpy as np

from dopplerloc import audio, cleaning, pipeline, ridge


def _chirp(sr=8000, dur=40.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    f_inst = 720 - 150 * np.tanh((t - dur / 2) / 4.0)          # Doppler-like S-curve (870 -> 570 Hz)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    x = 0.3 * np.sin(phase) + 0.05 * rng.normal(size=len(t))
    x += 0.2 * np.sin(2 * np.pi * 50.0 * t) + 0.1 * np.sin(2 * np.pi * 150.0 * t)  # mains
    x += 0.15 * np.sin(2 * np.pi * 1000.0 * t)                                     # stationary whine
    return x, sr, t, f_inst


def test_mains_removal():
    x, sr, t, _ = _chirp()
    y = cleaning.remove_mains_hum(x, sr)
    X = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / sr)
    X0 = np.abs(np.fft.rfft(x))
    k = np.argmin(np.abs(f - 50.0))
    assert X[k] < 0.05 * X0[k]


def test_ridge_tracks_s_curve_and_rejects_stationary():
    x, sr, t, f_inst = _chirp()
    cfg = pipeline.ExtractConfig(n_ridges=3, min_duration_s=8.0, fmin=150, fmax=1500)
    S_raw, S_line, ridges, S = pipeline.extract(x, sr, cfg)
    assert len(ridges) >= 1
    r = max(ridges, key=len)
    truth = np.interp(r.t, t, f_inst)
    assert np.median(np.abs(r.f - truth)) < 0.5
    assert len(r) * S.dt > 30
    # the 1000 Hz constant line must not be reported as a Doppler ridge
    assert not any(np.ptp(q.f) < 3 and abs(np.median(q.f) - 1000) < 5 for q in ridges)


def test_viterbi_simple():
    score = np.zeros((50, 30))
    path = (20 + np.round(5 * np.sin(np.arange(30) / 5))).astype(int)
    score[path, np.arange(30)] = 10
    score[40, ::2] = 12   # intermittent distractor
    k = ridge.viterbi_ridge(score, max_jump=2, lam=0.1)
    assert (k == path).mean() > 0.9


def test_reassigned_sharpens_tone():
    sr = 8000
    t = np.arange(sr * 4) / sr
    f0 = 1234.56
    x = np.sin(2 * np.pi * f0 * t)
    S = audio.stft(x, sr, win_length=1024, n_fft=1024)
    R = audio.reassigned_spectrogram(x, sr, win_length=1024, n_fft=1024)
    mid = S.power.shape[1] // 2

    def width(P):
        col = P[:, mid] / P[:, mid].max()
        return (col > 0.1).sum()
    assert width(R.power) < width(S.power)
    assert abs(R.freqs[np.argmax(R.power[:, mid])] - f0) <= R.df
