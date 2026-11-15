"""
preprocess.py - radar clean-up helpers shared by the whole pipeline (numpy only at import time).

Where each Sentinel-1 input comes from, and what it needs:
    GEE  COPERNICUS/S1_GRD   already calibrated + terrain-corrected + in dB  -> use as is
    ASF HyP3 RTC GeoTIFFs     calibrated + terrain-corrected, POWER (linear)   -> to_db()
    Raw GRD .zip (SAFE)       NOT calibrated or terrain-corrected. Do not feed it to this
                              pipeline: proper processing needs SNAP or HyP3, not plain Python.
    NISAR L2 GCOV             power (linear), handled in nisar_extract.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter

DB_FLOOR = 1e-6  # linear power floor (-60 dB) so log10 never sees 0


def to_db(power: np.ndarray) -> np.ndarray:
    """Linear power -> dB. Non-positive / non-finite pixels become NaN (no data), not -60 dB."""
    p = np.asarray(power, dtype="float32")
    out = np.full(p.shape, np.nan, dtype="float32")
    ok = np.isfinite(p) & (p > 0)
    out[ok] = 10.0 * np.log10(np.maximum(p[ok], DB_FLOOR))
    return out


def looks_like_db(a: np.ndarray) -> bool:
    """Heuristic: backscatter in dB is mostly negative; linear power is positive and small."""
    v = a[np.isfinite(a)]
    return bool(v.size and np.median(v) < 0 and v.min() < -5)


def ensure_db(a: np.ndarray) -> np.ndarray:
    """Return dB whether the input was dB already or linear power."""
    return np.asarray(a, "float32") if looks_like_db(a) else to_db(a)


def speckle_filter_db(a: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter that leaves NaN (no-data) pixels as NaN."""
    nan = ~np.isfinite(a)
    filled = np.where(nan, np.nanmedian(a) if (~nan).any() else 0.0, a)
    out = median_filter(filled, size=size, mode="nearest")
    out[nan] = np.nan
    return out


def change_db(pre_db: np.ndarray, post_db: np.ndarray) -> np.ndarray:
    """post - pre in dB. Negative = got darker (water-like), positive = got brighter."""
    return (post_db - pre_db).astype("float32")


def slope_from_dem(dem_m: np.ndarray, pixel_m: float) -> np.ndarray:
    """Terrain slope in degrees. pixel_m MUST be the real DEM pixel size (30 for GLO-30)."""
    dy, dx = np.gradient(np.asarray(dem_m, "float32"), pixel_m, pixel_m)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype("float32")


def nisar_rgb(hh_db: np.ndarray, hv_db: np.ndarray, low_signal: Optional[np.ndarray] = None,
              pct=(2, 98)) -> np.ndarray:
    """
    False-colour composite used in NASA's NISAR flood demos:
        red = HH,  green = HV,  blue = HH/HV ratio (HH_dB - HV_dB).
    Each channel is stretched between its own 2nd and 98th percentile. Returns (H, W, 3) in 0..1.
    Interpretation guide: open water dark; flooded vegetation (double bounce) bright in HH.
    """
    chans = [hh_db, hv_db, hh_db - hv_db]
    rgb = np.zeros(hh_db.shape + (3,), dtype="float32")
    for i, c in enumerate(chans):
        v = c[np.isfinite(c)]
        lo, hi = (np.percentile(v, pct[0]), np.percentile(v, pct[1])) if v.size else (0, 1)
        rgb[..., i] = np.clip((np.nan_to_num(c, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    if low_signal is not None:
        rgb[low_signal.astype(bool)] = 0
    return rgb


def export_web_png(db: np.ndarray, out_png: str | Path, vmin: float = -25.0, vmax: float = 0.0) -> Path:
    """Grayscale PNG for web maps. No-data pixels are written fully transparent."""
    from PIL import Image
    g = np.clip((np.nan_to_num(db, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    a = np.where(np.isfinite(db), 255, 0).astype("uint8")
    rgba = np.dstack([(g * 255).astype("uint8")] * 3 + [a])
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out)
    return out


def align_raster_to_reference(src_path: str | Path, ref_path: str | Path,
                              method: str = "bilinear") -> np.ndarray:
    """Resample any GeoTIFF onto the grid of `ref_path` (needs rasterio). Use 'nearest' for classes."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        dst = np.full((ref.height, ref.width), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref.transform, dst_crs=ref.crs, dst_nodata=np.nan,
                  resampling=getattr(Resampling, method))
    return dst
