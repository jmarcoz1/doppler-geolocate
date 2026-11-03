"""Interactive web maps (folium / Leaflet + OpenStreetMap tiles) for the location likelihood.

Optional dependency: `pip install -e .[notebook]`.
"""
from __future__ import annotations

import base64
import io

import numpy as np

from . import geodesy


def misfit_rgba(grid, cmap="magma", floor=-1.5, alpha=0.65):
    """RGBA image of -log10(chi2/chi2_min) (0 = best); cells worse than `floor` are transparent."""
    import matplotlib
    s = -np.log10(grid.chi2 / np.nanmin(grid.chi2))
    s = np.clip(s, floor, 0)
    norm = (s - floor) / -floor
    rgba = matplotlib.colormaps[cmap](norm)
    rgba[..., 3] = np.where(s <= floor + 1e-9, 0.0, alpha * (0.25 + 0.75 * norm))
    return rgba[::-1]   # image rows go north -> south


def _png_data_url(rgba):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def grid_bounds(grid):
    dlat = np.abs(np.diff(grid.lat[:, 0])).mean() / 2 if grid.lat.shape[0] > 1 else 0
    dlon = np.abs(np.diff(grid.lon[0])).mean() / 2 if grid.lon.shape[1] > 1 else 0
    return [[float(grid.lat.min() - dlat), float(grid.lon.min() - dlon)],
            [float(grid.lat.max() + dlat), float(grid.lon.max() + dlon)]]


def add_grid(fmap_or_group, grid, name="likelihood", **kw):
    import folium
    folium.raster_layers.ImageOverlay(_png_data_url(misfit_rgba(grid, **kw)), bounds=grid_bounds(grid), name=name,
                                      interactive=False, cross_origin=False, zindex=1).add_to(fmap_or_group)


def add_tracks(fmap_or_group, models, t_window=None, color="#ff8c00"):
    import folium
    for m in models:
        t0, t1 = (m.t0, m.t1) if t_window is None else (max(m.t0, t_window[0]), min(m.t1, t_window[1]))
        if t1 <= t0:
            continue
        tt = np.linspace(t0, t1, 300)
        la, lo, h = m.geodetic(tt)
        folium.PolyLine(np.c_[la, lo].tolist(), color=color, weight=3, opacity=0.9,
                        tooltip=f"{m.callsign or m.icao24}: {h.min():.0f}-{h.max():.0f} m").add_to(fmap_or_group)
        folium.Marker([la[-1], lo[-1]], icon=folium.DivIcon(
            html=f'<div style="color:{color};font-weight:bold;font-size:11px">&#9992; {m.callsign or m.icao24}</div>')
        ).add_to(fmap_or_group)


def refinement_map(stages, models, truth=None, upto=None, t_window=None, zoom_start=12, height=520):
    """Leaflet map with one toggleable layer per refinement stage (only stages[:upto] shown)."""
    import folium
    upto = len(stages) if upto is None else upto
    last = stages[upto - 1]
    fmap = folium.Map(location=[last.lat, last.lon], zoom_start=zoom_start, tiles="OpenStreetMap",
                      height=height, control_scale=True)
    add_tracks(folium.FeatureGroup(name="ADS-B tracks").add_to(fmap), models, t_window)
    colours = ["#7f7fff", "#00bfff", "#00e5a0", "#ff2a2a"]
    for k, s in enumerate(stages[:upto]):
        grp = folium.FeatureGroup(name=s.name, show=(k == upto - 1) or s.grid is None)
        if s.grid is not None:
            add_grid(grp, s.grid, name=s.name)
        folium.CircleMarker([s.lat, s.lon], radius=6 if s.grid is not None else 8, color=colours[k % 4], fill=True,
                            fill_opacity=0.9, tooltip=f"{s.name}: {s.lat:.5f}, {s.lon:.5f}").add_to(grp)
        grp.add_to(fmap)
    if upto > 1:
        folium.PolyLine([[s.lat, s.lon] for s in stages[:upto]], color="white", weight=2, dash_array="4",
                        tooltip="estimate path through the stages").add_to(fmap)
    res = last.result
    if res is not None and np.all(np.isfinite(res.cov_en)):
        w, V = np.linalg.eigh(res.cov_en)
        a = np.linspace(0, 2 * np.pi, 60)
        ell = V @ (np.sqrt(np.maximum(w, 0) * 5.991)[:, None] * np.stack([np.cos(a), np.sin(a)]))
        la, lo = geodesy.offset_latlon(res.lat, res.lon, ell[0], ell[1])
        folium.Polygon(np.c_[la, lo].tolist(), color="red", weight=1, fill=False, tooltip="95 % ellipse (formal)").add_to(fmap)
    if truth is not None:
        err = float(geodesy.haversine_m(last.lat, last.lon, truth.lat, truth.lon))
        folium.Marker([truth.lat, truth.lon], tooltip=f"truth (hidden from the solver); error {err:.1f} m",
                      icon=folium.Icon(color="green", icon="star")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap
