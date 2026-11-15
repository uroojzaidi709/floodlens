"""
baseline_otsu.py - threshold / Otsu / change-detection flood mapping (the transparent baseline).

Input arrays are Sentinel-1 backscatter in dB (what GEE's COPERNICUS/S1_GRD gives).
Every step records how many pixels it removed, so the app can show a "funnel":
    dark in post image -> darker than before -> not permanent water -> not steep -> not speckle
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import label
from skimage.filters import threshold_otsu

from forensics import quarantine_steep_pixels
from preprocess import speckle_filter_db


@dataclass
class MaskParams:
    vh_water_db: float = -18.0      # post-flood VH darker than this = water-like
    change_db: float = -3.0         # post minus pre must be below this
    jrc_occurrence_max: float = 50  # JRC occurrence above this % = permanent water
    max_slope_deg: float = 5.0
    min_pixels: int = 5             # drop speckle blobs smaller than this
    speckle_window: int = 3         # median filter size (pixels)
    use_otsu: bool = False          # derive the water threshold from the data instead


def otsu_threshold_db(a: np.ndarray, lo: float = -25.0, hi: float = -12.0) -> Dict:
    """
    Otsu threshold on a dB image. Otsu assumes two populations; in a scene with little
    water it lands in the land mode. So we accept it only inside [lo, hi] dB, otherwise
    fall back to -18 dB and say so.
    """
    v = a[np.isfinite(a)]
    if v.size < 100:
        return {"threshold_db": -18.0, "method": "fallback_fixed", "reason": "too few pixels"}
    t = float(threshold_otsu(v))
    if lo <= t <= hi:
        return {"threshold_db": round(t, 3), "method": "otsu"}
    return {"threshold_db": -18.0, "method": "fallback_fixed",
            "reason": f"Otsu gave {t:.1f} dB, outside [{lo}, {hi}]"}


def drop_small_blobs(m: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove connected components (8-neighbour) smaller than min_pixels."""
    lab, n = label(m, structure=np.ones((3, 3)))
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[lab]


def _count(m: np.ndarray) -> int:
    return int(np.count_nonzero(m))


def map_flood(vh_pre: np.ndarray, vh_post: np.ndarray,
              slope_deg: Optional[np.ndarray] = None,
              jrc_occurrence: Optional[np.ndarray] = None,
              params: MaskParams = MaskParams()) -> Dict:
    """Returns {'mask': bool array, 'funnel': {...}, 'params': {...}, 'threshold': {...}}."""
    if vh_pre.shape != vh_post.shape:
        raise ValueError("pre/post shape mismatch")
    pre = speckle_filter_db(vh_pre, params.speckle_window)
    post = speckle_filter_db(vh_post, params.speckle_window)
    valid = np.isfinite(pre) & np.isfinite(post)

    thr = ({"threshold_db": params.vh_water_db, "method": "fixed"} if not params.use_otsu
           else otsu_threshold_db(post))
    funnel = {"valid_pixels": _count(valid)}

    m = valid & (post < thr["threshold_db"])
    funnel["dark_in_post_image"] = _count(m)

    m &= (post - pre) < params.change_db
    funnel["and_darker_than_before"] = _count(m)

    if jrc_occurrence is not None:
        perm = np.nan_to_num(jrc_occurrence, nan=0.0) > params.jrc_occurrence_max
        funnel["removed_permanent_water"] = _count(m & perm)
        m &= ~perm
    funnel["after_permanent_water_removed"] = _count(m)

    if slope_deg is not None:
        m, n_steep = quarantine_steep_pixels(m, slope_deg, params.max_slope_deg)
        funnel["removed_steep_terrain"] = n_steep
    funnel["after_terrain_qc"] = _count(m)

    if params.min_pixels > 1 and m.any():
        m = drop_small_blobs(m, params.min_pixels)
    funnel["final_flood_pixels"] = _count(m)

    return {"mask": m, "funnel": funnel, "params": asdict(params), "threshold": thr}


def agreement(a: np.ndarray, b: np.ndarray, valid: Optional[np.ndarray] = None) -> Dict:
    """How much two masks agree (NOT accuracy - neither is ground truth)."""
    a, b = a.astype(bool), b.astype(bool)
    if valid is not None:
        a, b = a & valid, b & valid
    inter, union = _count(a & b), _count(a | b)
    return {"iou": round(inter / union, 4) if union else None,
            "only_a_pixels": _count(a & ~b), "only_b_pixels": _count(b & ~a),
            "both_pixels": inter}
