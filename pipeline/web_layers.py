"""
web_layers.py - turn rasters into transparent PNG overlays + bounds, so the frontend can drop
them on a Leaflet/folium map and build the swipe slider without touching GeoTIFFs.

Why we re-project: our rasters are on a UTM grid (EPSG:32642); a web map is lon/lat. If we
just stretched a UTM image between lon/lat corners it would be rotated about half a degree
(around 1 km at the edges of a 150 km scene). warp_to_wgs84() fixes that.

Everything except warp_to_wgs84() / read_tif() is plain numpy and unit-tested; those two need
rasterio. `warp` can be injected, which is how the tests run without rasterio.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from preprocess import export_web_png, nisar_rgb

Grid = Tuple[float, float, float, str]          # (origin_x, origin_y, pixel_m, crs)
Bounds = Tuple[float, float, float, float]      # (west, south, east, north)
WarpFn = Callable[[np.ndarray, Grid, str], Tuple[np.ndarray, Bounds]]


# --------------------------------------------------------------------------- needs rasterio
def warp_to_wgs84(arr: np.ndarray, grid: Grid, resampling: str = "nearest") -> Tuple[np.ndarray, Bounds]:
    """Re-project a north-up array on a metric grid to EPSG:4326. Returns (array, bounds)."""
    from rasterio.transform import array_bounds, from_origin
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    ox, oy, px, crs = grid
    h, w = arr.shape
    src_t = from_origin(ox, oy, px, px)
    dst_t, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h,
                                                left=ox, bottom=oy - h * px, right=ox + w * px, top=oy)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(np.asarray(arr, dtype="float32"), dst, src_transform=src_t, src_crs=crs,
              src_nodata=np.nan, dst_transform=dst_t, dst_crs="EPSG:4326", dst_nodata=np.nan,
              resampling=getattr(Resampling, resampling))
    west, south, east, north = array_bounds(dh, dw, dst_t)
    return dst, (west, south, east, north)


def read_tif(path: str | Path) -> Tuple[np.ndarray, Grid]:
    import rasterio
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
        return a, (s.transform.c, s.transform.f, abs(s.transform.a), str(s.crs))


# --------------------------------------------------------------------------- pure numpy
def colorize_mask(mask_f: np.ndarray, rgb=(31, 111, 139), alpha: int = 175) -> np.ndarray:
    """Float 0/1 (NaN = none) -> RGBA with only flooded pixels visible."""
    on = np.nan_to_num(mask_f, nan=0.0) >= 0.5
    out = np.zeros(mask_f.shape + (4,), dtype="uint8")
    out[on] = (*rgb, alpha)
    return out


def colorize_diverging(chg: np.ndarray, vmin: float = -8, vmax: float = 8) -> np.ndarray:
    """dB change -> RGBA with the same RdBu scale as the NISAR change figure. NaN transparent."""
    import matplotlib
    cmap = matplotlib.colormaps["RdBu"]
    norm = np.clip((np.nan_to_num(chg, nan=0.0) - vmin) / (vmax - vmin), 0, 1)
    rgba = (cmap(norm) * 255).astype("uint8")
    rgba[~np.isfinite(chg), 3] = 0
    return rgba


def rgb_to_rgba(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    a = np.where(valid, 255, 0).astype("uint8")
    return np.dstack([(np.clip(rgb, 0, 1) * 255).astype("uint8"), a])


def save_rgba(rgba: np.ndarray, path: Path) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def _leaflet(b: Bounds):
    west, south, east, north = b
    return [[south, west], [north, east]]       # Leaflet order: [[south, west], [north, east]]


def build_event_layers(out_dir: Path, grid: Grid, bands: Dict[str, np.ndarray],
                       masks: Dict[str, np.ndarray], warp: WarpFn = warp_to_wgs84) -> Dict:
    """Sentinel-1 overlays: VH pre / post (grey), and each flood mask (blue)."""
    lay = out_dir / "layers"
    layers, bounds = {}, None
    for name, key in (("s1_vh_pre", "vh_pre"), ("s1_vh_post", "vh_post")):
        w, bounds = warp(bands[key], grid, "bilinear")
        layers[name] = {"file": save_rgba(_png_gray(w), lay / f"{name}.png").name,
                        "label": f"Sentinel-1 C-band VH ({'before' if 'pre' in name else 'during flood'})"}
    for name, m in masks.items():
        if m is None:
            continue
        w, bounds = warp(m.astype("float32"), grid, "nearest")
        layers[name] = {"file": save_rgba(colorize_mask(w), lay / f"{name}.png").name,
                        "label": name.replace("_", " ")}
    info = {"bounds_leaflet": _leaflet(bounds), "bounds_wgs84": list(bounds), "layers": layers,
            "folder": "layers"}
    (lay / "layers.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _png_gray(db: np.ndarray) -> np.ndarray:
    g = np.clip((np.nan_to_num(db, nan=-25.0) + 25.0) / 25.0, 0, 1)
    a = np.where(np.isfinite(db), 255, 0).astype("uint8")
    return np.dstack([(g * 255).astype("uint8")] * 3 + [a])


def build_nisar_layers(nisar_dir: Path, tag: str, out_dir: Path,
                       warp: WarpFn = warp_to_wgs84, reader=read_tif) -> Dict:
    """NISAR overlays: HH/HV before+after, change maps, and the NASA-style RGB composite."""
    nd = Path(nisar_dir)
    arrs, grid = {}, None
    for pol in ("HHHH", "HVHV"):
        for lab in ("before", "after", "change"):
            arrs[(pol, lab)], grid = reader(nd / f"{tag}_{pol}_{lab}_db.tif")
    low, _ = reader(nd / f"{tag}_low_signal_mask.tif")
    low = low == 1

    lay = out_dir / "layers"
    layers, bounds = {}, None

    def put(name, rgba_fn, arr_list, label, resampling="bilinear"):
        nonlocal bounds
        warped = []
        for a in arr_list:
            w, bounds = warp(a, grid, resampling)
            warped.append(w)
        layers[name] = {"file": save_rgba(rgba_fn(*warped), lay / f"{name}.png").name, "label": label}

    for pol, short in (("HHHH", "hh"), ("HVHV", "hv")):
        for lab in ("before", "after"):
            put(f"nisar_{short}_{lab}", lambda a: _png_gray(a), [np.where(low, np.nan, arrs[(pol, lab)])],
                f"NISAR L-band {short.upper()} ({lab})")
        put(f"nisar_{short}_change", lambda a: colorize_diverging(a), [arrs[(pol, "change")]],
            f"NISAR {short.upper()} change (dB): blue brighter, red darker")
    for lab in ("before", "after"):
        put(f"nisar_rgb_{lab}",
            lambda hh, hv: rgb_to_rgba(nisar_rgb(hh, hv), np.isfinite(hh) & np.isfinite(hv)),
            [np.where(low, np.nan, arrs[("HHHH", lab)]), np.where(low, np.nan, arrs[("HVHV", lab)])],
            f"NISAR RGB ({lab}): R=HH, G=HV, B=HH/HV")
    info = {"bounds_leaflet": _leaflet(bounds), "bounds_wgs84": list(bounds), "layers": layers,
            "folder": "layers"}
    (lay / "nisar_layers.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
