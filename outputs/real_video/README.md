# Real-video attempt: locating a recording from its audio alone

Video: https://www.youtube.com/watch?v=G0fjcT-SP1o (picked from a `yt-dlp ytsearch` that printed
only ids and durations; chosen because its spectrogram showed three clean, unedited flyovers).
Reproduce with `python scripts/real_video_attempt.py https://www.youtube.com/watch?v=G0fjcT-SP1o`.
Only the audio stream is downloaded. The audio itself is not in git (copyright).

## Part 1: audio-only analysis (written before any metadata was read; see `audio_only_result.json`)

Recording: 874 s, 48 kHz, true stereo (the channels are only partly correlated, with an
inter-channel delay that sweeps).

| | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| closest approach (CPA) time in recording | 40.4 s | 432.4 s | 732.6 s |
| speed, straight-pass Doppler fit (c = 343 m/s) | **47.4 m/s (92 kt)** | **96.5 m/s (188 kt)** | **61.9 m/s (120 kt)** |
| slant CPA distance | **596 m** | **2031 m** | **678 m** |
| fit rms (16 tonal lines collapse onto one Doppler factor) | 1.27 Hz | 1.01 Hz | 1.17 Hz |
| sensitivity to c over 0–35 °C | 45.8–48.7 m/s, 576–613 m | 93–99 m/s, 1962–2088 m | 59.8–63.6 m/s, 655–697 m |
| rest-frame tone comb | 96.4 Hz (8 harmonics) + other orders | 61.9 Hz (12 harmonics); every line is a multiple of 20.65 Hz | 94.4 Hz (8 harmonics) + other orders |
| engine reading | piston propeller, no jet/fan tones | different piston engine: a 20.65 Hz cycle rate means ~2480 rpm 4-stroke; strong 1.5 and 3rd orders suggest a 6-cylinder | same comb as pass 1 within 2 %, so the same type and probably the same aircraft |
| Lloyd fringe, h_r·H | 511 m² | 2288 m² (only 7 points, weak) | 590 m² |
| altitude above mic (h_r between 0.87 and 2 m) | 256–588 m | 1140–2030 m (weak) | 295–678 m |
| stereo delay sweep | −0.32 → +0.41 ms (R→L) | +0.24 → −0.50 ms (L→R) | +0.67 → −0.54 ms (L→R) |

Other findings:
* Mic height is at least 0.87 m, because the fringe peak delay requires H ≤ d. So the mic is not on the ground.
* Passes are 392 s and 300 s apart. Passes 1 and 3 are the same type, in opposite directions, at
  low altitude and light-aircraft speed. That pattern looks like one aircraft circling or flying circuits nearby.
* The stereo pair's largest delay is about ±0.7 ms, so it is a spaced or near-coincident pair (≳0.2 m) rather than a single mic.
  A steady background source sits at a fixed delay of about +0.45 ms (a road or similar).
* There is no mains hum above the control families (45/55/65 Hz). So there is no 50 vs 60 Hz region
  hint, which fits a battery recorder outdoors.
* **A location estimate is not possible.** The method needs the recording time and the ADS-B tracks
  from that moment. Audio gives only relative geometry (speed, range, altitude, direction relative to
  the mic axis) and the aircraft class. In 12 global ADS-B Exchange snapshots from one sample day
  (2025-10-01), 128–1316 aircraft per snapshot match each pass's speed and altitude window. On average
  ~76 one-degree cells per snapshot contain all three classes at once, and that is one day out of years.
  See `candidate_map_global.png`. The historical sources themselves were reachable: ADS-B Exchange
  samples (only the 1st of each month), and adsb.lol globe_history releases on GitHub (~3.2 GB per day).

## Part 2: grading against the metadata (read only after Part 1 was saved)

* Title: "Planes Flying Overhead While the Birds Sing in the Distance". Location field: "PINK CLIFFS
  RESERVE", which is Heathcote, Victoria, Australia (~36.930 S, 144.705 E). Uploaded 2021-05-06.
* Description: close to highway noise, *"a plane flying loops over the general area"*. Gear: Zoom F6,
  **Rode NT5 stereo pair in ORTF** on a Manfrotto Nanopole stand.
* How the audio-only inferences held up:
  * Correct: there was a repeating light aircraft (passes 1 and 3 share a comb; "flying loops").
  * Correct: a steady road source (the highway) at a fixed stereo bearing.
  * Correct: a real stereo pair. ORTF capsules are 17 cm apart, about 0.5 ms of pure delay, so my ≳0.2 m
    spacing estimate is roughly right (the measured ±0.7 ms includes the effect of the capsule angles).
  * Consistent: mic height ≥ 0.87 m (it was on a stand).
  * Consistent: no hum. Australia is 50 Hz, but a battery recorder in the bush picks none up.
  * Not achievable: the location. Heathcote is not near a busy airport, the recording date is unknown
    (only the upload date), and there is no ADS-B time to match. The method returns no position, which
    is the correct, honest outcome.
