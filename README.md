# dopplerloc: geolocate an audio recording from aircraft Doppler curves

When a jet flies past, the tones it radiates (fan blade-pass frequency, compressor and turbine
tones, propeller harmonics) reach you Doppler-shifted. The shape of that frequency-vs-time
"S-curve" depends on where you stood relative to the flight path. ADS-B flight-tracking data says
where every aircraft was, second by second. If you match the curves in a recording against the
recorded tracks, you can solve for the microphone position, and often the true recording time as well.

This repository is a working, tested implementation of the full chain:

```
ADS-B tracks ──► continuous ECEF trajectories ─┐
                                               ├─► forward model f_obs(t | observer) ─► grid search ─► least squares ─► position ± σ
audio ──► STFT ──► cleaning ──► ridge tracking ┘        (WGS84, retarded time, ISA c, wind)        (+ Lloyd's-mirror h_r)
```

> **Ethics.** This can reveal where a recording was made, and so where a person was. Use it only on
> your own recordings, on recordings whose owners consent, or for legitimate research and
> verification work such as journalism or OSINT that follows the usual harm-minimisation rules. Do
> not use it to locate, track, or expose people without their consent. Also respect each data
> provider's terms. In particular, this project does **not** scrape Flightradar24, because their
> ToS forbids it. Use their official paid API instead.

---

## 1. Setup

```bash
cd doppler-geolocate
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e .[dev]
python -m pytest -q                    # 22 fast tests
RUN_SLOW=1 python -m pytest -q tests/test_end_to_end.py   # +1 full synthetic run (~2 min)
python -m dopplerloc simulate          # end-to-end demo -> outputs/*.png, synthetic_result.json
```

To decode mp3, m4a, and similar formats, `ffmpeg` must be on PATH. wav, flac, and ogg go through soundfile.

## 2. CLI

```bash
# 1) Flight data: live polling while you record (anonymous OpenSky: >=10 s; adsb.lol: ~2 s)
python -m dopplerloc fetch --source opensky --lat 39.4893 --lon -0.4816 --radius-km 30 \
       --poll 600 --out data/tracks.parquet
python -m dopplerloc fetch --source adsblol --poll 600 --interval 2 --out data/tracks.parquet
#    ...or historical (needs credentials, see §6)
python -m dopplerloc fetch --source opensky-trino --start 2026-11-03T17:00:00 --end 2026-11-03T17:10:00 \
       --out data/tracks.parquet
#    optional uniform resampling of the cached tracks
python -m dopplerloc fetch ... --resample 1.0

# 2) Spectrogram plot (optionally reassigned)
python -m dopplerloc spectrogram rec.wav --out outputs/spec.png --fmax 3000 [--reassigned]

# 3) Clean + extract Doppler ridges -> CSV (t, f, snr_db, conf, sigma_f, ridge)
python -m dopplerloc extract rec.wav --out outputs/ridges.csv --plot outputs/ridges.png --mains 50

# 4) Locate
python -m dopplerloc locate --ridges outputs/ridges.csv --tracks data/tracks.parquet \
       --start 2026-11-03T17:02:13 --start-sigma 2 --center 39.49,-0.48 --radius-km 10 \
       --obs-h 120 --out-dir outputs
#   --obs-h is the *ellipsoidal* microphone height (MSL + ~50 m geoid in Valencia)
#   --wind we,wn  --isa-dt K   adjust the propagation model

# 5) Full synthetic demo
python -m dopplerloc simulate --out-dir outputs
```

## 3. Package layout

| module | what it does |
|---|---|
| `sources/` | pluggable flight-data layer: `opensky` (REST, OAuth2), `opensky-trino` (historical DB via `pyopensky`), `adsblol` / `airplaneslive` / `adsbx` (readsb v2 `/v2/point` APIs), `fr24` (stub for the official paid FR24 API). Disk cache (parquet, CSV fallback). Live `poll()` builds tracks from snapshots. |
| `tracks.py` | canonical SI frame (`time, icao24, callsign, lat, lon, alt_geom, alt_baro, gs, track, vrate, ...`), GCV smoothing-spline or Hermite (reported-velocity) trajectory models in ECEF, uniform-grid resampling. |
| `geodesy.py` | WGS84 geodetic↔ECEF↔ENU (checked against pyproj to 1e-6 m). |
| `atmosphere.py` | ISA T(h), c(h), harmonic-mean effective c along a slant path, ISO 9613-1 absorption, spreading. |
| `physics.py` | retarded-time solver, radial velocity, moving-source Doppler with wind, Lloyd's-mirror delay and notch predictor. |
| `audio.py` | load and resample, STFT (dB), reassigned spectrogram. |
| `cleaning.py` | Butterworth HP/BP, adaptive 50 Hz mains-notch comb, per-frequency noise-floor subtraction, narrow-band line enhancement, stationary-line suppression. |
| `ridge.py` | multi-ridge Viterbi tracking with continuity, sub-bin refinement, confidence and σ_f, rejection of stationary and jagged paths. |
| `lloyd.py` | cepstral estimate of the ground-reflection delay τ(t). |
| `locate.py` | vectorised grid search with data association → fine grids → variable-projection least squares, covariance, multi-start ranking, outlier-ridge rejection. |
| `simulate.py`, `demo.py` | physically synthesised LEVC scenario and the end-to-end run. |
| `plotting.py` | spectrogram + ridges, fitted curves, likelihood map, fringe plot. |

## 4. The physics

**Geometry.** Everything is in WGS84 ECEF. For geodetic $(\varphi,\lambda,h)$, with $N=a/\sqrt{1-e^2\sin^2\varphi}$:
$x=(N+h)\cos\varphi\cos\lambda,\; y=(N+h)\cos\varphi\sin\lambda,\; z=(N(1-e^2)+h)\sin\varphi$.
Slant range is $R=\lVert\mathbf P_{src}-\mathbf P_{obs}\rVert$. At 50 km the Earth's curvature
drops the target ~196 m below the local horizontal, so flat-earth shortcuts aren't used.
Heights are ellipsoidal. ADS-B `alt_geom` is (nominally) height above the ellipsoid. `alt_baro` is
pressure altitude, which can be off by tens to hundreds of metres from true height depending on QNH
and temperature. Map elevations are orthometric (geoid N ≈ +50 m around Valencia).

**Trajectories.** Each aircraft's reports are fit with GCV smoothing splines on ECEF x, y, z, which
smooths out the 25 ft altitude quantisation and position noise. The velocity $\mathbf V(t)$ is the
spline derivative. A Hermite option uses the reported ADS-B velocities instead.

**Retarded time.** Sound heard at $t_{obs}$ left the aircraft at $t_e$:
$$t_{obs}=t_e+\frac{R(t_e)}{c_w(t_e)}$$
This is solved by Newton iteration with $g'(t_e)=1+v_r/c_w>0$, which holds for subsonic sources.
At 3 km the correction is ~9 s, during which an airliner moves ~700 m, so ignoring it shifts the
curve badly.

**Sound speed.** ISA gives $T(h)=288.15-0.0065h$ (+ΔT), $c=\sqrt{\gamma R_{air}T}$. A straight ray's
travel time is $\int ds/c$, so the correct average is the harmonic mean over height,
$c_{eff}=\Delta z/\int dz/c(z)$ (Gauss-Legendre). You can pass a measured temperature profile.

**Wind.** With wind $\mathbf w$ and propagation direction $\hat n=-\hat u$ (source→observer), the
ray speed becomes $c_w=c+\mathbf w\cdot\hat n$.

**Doppler (sign convention).** $\hat u=(\mathbf P_{src}-\mathbf P_{obs})/R$ and
$v_r=\mathbf V\cdot\hat u$, **positive = receding**. For a moving source and a static observer:
$$f_{obs}=f_{src}\,\frac{c_w}{c_w+v_r(t_e)}=f_{src}\,\frac{c_w}{c_w-v_{approach}}$$
Before CPA it reads above $f_{src}$, after CPA below.

**What constrains the position.** For a straight, level pass with speed $v$ and slant CPA distance
$R_0$:
* The inflection of the S-curve occurs at the heard CPA time, $t_{cpa}+R_0/c$. That fixes *where along* the track you are.
* The slope there is $\dot f\approx -f_{src}\,v^2/(c\,R_0)$. That fixes $R_0$. With the
  aircraft altitude known, $R_0$ gives the lateral offset.
* The asymptotes $f_{src}c/(c\mp v)$ fix $f_{src}$, which is fitted as a nuisance parameter.
* A single straight pass is **left/right mirror ambiguous**. A second, non-parallel track, a turn,
  or the Lloyd's-mirror cue resolves it. Clock error trades off against along-track position, so the
  start time and its uncertainty enter as a prior.

**Lloyd's mirror.** The reflected ray from hard ground arrives
$\tau=\Delta/c$ later, where $\Delta=\sqrt{d^2+(z_s+h_r)^2}-\sqrt{d^2+(z_s-h_r)^2}\approx 2h_r\sin\epsilon$.
This produces spectral notches at $f_n=(n+\tfrac12)/\tau$ (hard ground) or $n/\tau$ (soft ground). These
are the sweeping fringes visible in flyover spectrograms. `lloyd.cepstral_delay` measures $\tau(t)$
from the power cepstrum. That gives the elevation-angle history, an independent geometric
constraint, and the microphone height $h_r$.

**Amplitude.** Received amplitude is $p\propto R^{-1}\,10^{-\alpha(f)R/20}$, with $\alpha$ from ISO 9613-1
(`atmosphere.received_amplitude`). It's used in the simulator and available as a weighting.

## 5. Signal processing and inversion

1. **Load, resample** (8 kHz is plenty for fan tones), then **band-pass** and apply a **50 Hz mains
   comb** of 1.5 Hz-wide notches at k·f0. The actual grid frequency is estimated first.
2. **STFT** with a 4096-sample window (0.51 s), 50 ms hop, and zero-padding to 8192. The optional
   reassigned spectrogram (Auger-Flandrin) gives sharper lines.
3. **Noise floor:** subtract the 20th percentile over time per frequency, in dB. This whitens the
   spectrum and kills stationary lines. A low percentile is used so that a long Doppler plateau
   isn't mistaken for background.
4. **Line enhancement:** subtract a 40 Hz running median across frequency. This removes jet
   broadband noise and Lloyd fringes and keeps narrow tones.
5. **Ridges:** run Viterbi over the score with a quadratic jump penalty and |Δk| ≤ max slope. Then
   mask the path and repeat. The path is snapped to the local peak and refined to sub-bin
   precision. Each frame gets a confidence and a σ_f. Paths that are constant (hum, whine), short,
   or jagged (noise) are dropped.
6. **Grid search:** each ridge $r$ and aircraft $m$ gives $f_{src}$ in closed form, and each ridge is
   assigned to the aircraft that explains it best (data association). The score is
   $\chi^2(\mathbf x)=\min_{dt}\big[\sum_r\min_m\chi^2_{rm}+(dt/\sigma_t)^2\big]$, evaluated
   vectorised over thousands of observers. A coarse 600 m grid is followed by fine 150 m grids
   around the best local minima.
7. **Refinement:** `scipy.optimize.least_squares` (TRF, soft-L1) on (east, north, [up], dt, [h_r])
   with the exact physics. $f_{src}$ is projected out (variable projection). Ridges whose misfit
   is more than 2.5× the median are rejected iteratively; these are typically ridges that jumped
   between crossing tones. Competing solutions are ranked by truncated χ² over *all* ridges.
   The covariance is $(J^TJ)^{-1}s^2$. The demo inflates it by $\sqrt{win/hop}$ because STFT frames are correlated.

## 6. Data sources and their limits

| source | what you get | limits |
|---|---|---|
| **OpenSky REST** `/states/all?lamin..` | live state vectors | **Anonymous:** current data only (the `time` parameter is ignored), 10 s resolution, 400 credits/day (a bbox ≤25 sq° costs 1 credit). **OAuth2 client credentials** (mandatory since 2026-03-18, basic auth is gone): up to 1 h back, 5 s resolution, 4000 credits/day (8000 for active feeders). Set `OPENSKY_CLIENT_ID` and `OPENSKY_CLIENT_SECRET`. Some cloud and datacenter networks are blocked at the TLS level. |
| **OpenSky Trino** (`pyopensky`) | full history, ~1 s state vectors (`state_vectors_data4`) | needs an OpenSky account **approved for historical access** (research or non-commercial). `OpenSkyTrino.sql()` prints the query if you run it elsewhere. |
| **adsb.lol** `/v2/point/lat/lon/nm` | live, no key, ODbL | live only. Historical data comes as daily `globe_history` dumps on GitHub (offline use, not wired up yet). |
| **airplanes.live** | same API | currently answers **403** to unregistered clients (email them). 1 req/s. |
| **ADS-B Exchange** via RapidAPI | same schema | paid key `ADSBX_RAPIDAPI_KEY`. Historical traces are commercial. |
| **Flightradar24 official API** | live and historic positions | paid token `FR24_API_TOKEN`. The stub follows their documented endpoints but is **untested** (no key). |

Live check on 2026-11-03, ~19:47 CET:
* `adsblol` worked from this machine: a 40 s poll returned 5 aircraft and 25 rows. The cached
  snapshot is in `tests/fixtures/adsblol_point_valencia.json`.
* `opensky` (anonymous) failed from this machine with a TLS EOF (the host blocks this network).
  The same URL fetched from another network returned 6 aircraft over Valencia, stored in
  `tests/fixtures/opensky_states_valencia.json`, and the parser test passes on it.
* `airplanes.live` returned 403.

Practical recipe: start `fetch --poll` (adsb.lol at 2 s or OpenSky at 10 s) **while you record**.
Otherwise you'll need historical access.

## 7. Synthetic validation (LEVC)

`python -m dopplerloc simulate` builds a 150 s recording heard by a hidden observer ~6 km NW of
Valencia airport, about 900 m off the RWY 12 approach centreline, with a 1.5 m microphone. The
scenario contains:
* An A320-ish arrival on a 3° glide path (72 m/s, decelerating) with tones at 612, 1224, and 1538 Hz.
* A departure climbing through 1500 m at 150 m/s on heading 035, followed by a right turn, with tones at 455 and 910 Hz.
* An inaudible distractor aircraft cruising at 11.5 km, 25 km away.
* For every audio sample: retarded time on the direct path **and** the ground-reflected path, so
  Doppler and Lloyd's fringes emerge physically. Also 1/R spreading, ISO absorption, broadband
  jet noise, 50.03 Hz mains with 10 harmonics, pink noise, wind rumble, and a +1.3 s clock error.
* The ADS-B input is degraded to 5 s sampling, 4 m position noise, and 25 ft altitude quantisation.

The run writes to `outputs/`, with results in `synthetic_result.json`. Results from the run on 2026-11-03:

| run | position error | clock offset (true −1.30 s) | notes |
|---|---|---|---|
| default (`simulate`, audio seed 7, ADS-B seed 1) | **7.3 m** | −1.26 s | 9 of 13 ridges used, rms 1.9 Hz, h_r 1.27 m (true 1.5) |
| audio seeds 11 / 42 | 8.7 m / 8.1 m | −1.25 s | earlier code revision |
| ADS-B noise seeds 2 / 3 | 5.7 m / 10.8 m | −1.32 / −1.23 s | earlier code revision |
| audio seed 23 | **147 m** | −3.34 s | stuck in a clock/along-track local minimum. The true position scores better, but the search didn't converge to it. |

The formal 1σ (~1 m) is **too optimistic**. The real error, ~5-10 m, comes from reconstructing the
sparse and quantised ADS-B track.

Output files:
* `spectrogram_ridges.png` (raw and cleaned spectrogram with the extracted ridges)
* `doppler_fit.png`
* `location_heatmap.png`
* `lloyd_fringes.png`
* `spectrogram_reassigned.png`
* `synthetic_recording.wav`
* `synthetic_adsb_tracks.parquet`

## 8. Limitations and next steps

* **Real recordings** need: a known start time (±a few s; phone metadata is often enough); aircraft
  tones that are audible for tens of seconds, which is easiest near approach and departure paths;
  and track data for that time. Ground reflections, buildings, and refraction from wind and
  temperature gradients distort the curves. Ray bending isn't modelled (straight rays only).
* Engine power changes (N1 spool-up and spool-down) make $f_{src}$ time-varying. Fitting a slow
  polynomial $f_{src}(t)$ is the natural extension.
* The error is dominated by **track reconstruction**: 5-10 s sparse samples and quantised
  altitude. The formal σ (~1 m) underestimates the real error (~10 m in the demo). Propagating track
  uncertainty, or using 1 s Trino data, would make the error bars honest.
* The Lloyd's-mirror cue currently estimates $h_r$ at the solution and isn't used in the global
  search. Adding it to the grid would help break the mirror ambiguity for single passes.
* Ridge association is greedy per ridge. Linking harmonics of the same engine (common $f_{src}$
  ratio) would add constraints.
* **Local minima.** Clock offset and along-track position trade off. The solver uses a coarse grid,
  up to 6 fine grids, least squares, and a clock-offset polish, but one of 6 synthetic runs still
  ended 147 m off. Better options: a global optimiser over (x, dt), or fixing the ridge subset
  *after* polishing.
* Ridges can splice across two crossing tones. These are rejected as outliers, which loses data. A
  multi-hypothesis tracker would keep it.
* The geoid is approximated as a constant 50 m. Use EGM2008 (via pyproj grids) and a DEM for the observer height.
