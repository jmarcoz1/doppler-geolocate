"""dopplerloc - geolocate an audio recording from aircraft Doppler curves.

Pipeline: flight tracks (ADS-B) + audio -> spectrogram -> cleaned tonal ridges
f_obs(t) -> physics forward model (WGS84 geometry, retarded time, ISA sound
speed, wind, moving-source Doppler) -> grid search + least-squares observer fit.
"""
__version__ = "0.1.0"
