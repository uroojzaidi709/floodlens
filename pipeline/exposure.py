"""
exposure.py - turn a flood mask into numbers people understand: km2, people, farmland, roads.

All rasters must be on ONE common grid with square metre pixels (we export from GEE in
EPSG:32642, UTM 42N, which covers both Dadu 67.8E and Larkana 68.2E).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

# ESA WorldCover v200 class codes
WORLDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built-up",
              60: "bare", 70: "snow/ice", 80: "water", 90: "wetland", 95: "mangrove",
              100: "moss"}


def area_km2(mask: np.ndarray, pixel_m: float) -> float:
    return float(np.count_nonzero(mask)) * pixel_m * pixel_m / 1e6


def people_exposed(mask: np.ndarray, pop_count: np.ndarray, pop_cell_m2: np.ndarray,
                   pixel_m: float) -> Dict:
    """
    pop_count   : WorldPop people per native cell, resampled (nearest) onto our grid.
    pop_cell_m2 : area of that native cell in m2 (same grid).
    Each of our pixels then holds pop_count * pixel_area / cell_area people, so summing over
    pixels does not count one native cell several times.
    """
    if not (mask.shape == pop_count.shape == pop_cell_m2.shape):
        raise ValueError("mask, pop_count and pop_cell_m2 must share a grid")
    cell = np.where(pop_cell_m2 > 0, pop_cell_m2, np.nan)
    per_px = np.nan_to_num(pop_count, nan=0.0) * (pixel_m * pixel_m) / cell
    per_px = np.nan_to_num(per_px, nan=0.0)
    return {"people_in_flood": float(per_px[mask].sum()),
            "people_in_aoi": float(per_px.sum())}


def landcover_exposed(mask: np.ndarray, worldcover: np.ndarray, pixel_m: float) -> Dict[str, float]:
    """km2 of flood per WorldCover class (cropland and built-up are the headline numbers)."""
    out: Dict[str, float] = {}
    codes = np.unique(worldcover[mask & np.isfinite(worldcover)]).astype(int)
    for c in codes:
        out[WORLDCOVER.get(int(c), f"class_{c}")] = round(area_km2(mask & (worldcover == c), pixel_m), 4)
    return out


def roads_in_flood(roads_xy: Iterable[np.ndarray], mask: np.ndarray,
                   origin_x: float, origin_y: float, pixel_m: float,
                   step_m: float = 10.0) -> Dict:
    """
    Length of road lying on flooded pixels.
    roads_xy : iterable of (N,2) arrays of x,y in the GRID's CRS (metres), one per road line.
    origin_x/y : coordinates of the top-left corner of the grid (north-up).
    Each line is sampled every step_m metres; length = samples_in_flood * step_m.
    """
    h, w = mask.shape
    total = flooded = 0.0
    for line in roads_xy:
        line = np.asarray(line, dtype=float)
        if len(line) < 2:
            continue
        seg = np.hypot(*np.diff(line, axis=0).T)
        dist = np.concatenate([[0], np.cumsum(seg)])
        L = dist[-1]
        if L == 0:
            continue
        s = np.arange(0, L, step_m)
        x = np.interp(s, dist, line[:, 0])
        y = np.interp(s, dist, line[:, 1])
        col = np.floor((x - origin_x) / pixel_m).astype(int)
        row = np.floor((origin_y - y) / pixel_m).astype(int)
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        total += inside.sum() * step_m
        flooded += mask[row[inside], col[inside]].sum() * step_m
    return {"road_km_in_aoi": round(total / 1000, 3), "road_km_flooded": round(flooded / 1000, 3)}


def facilities_in_flood(points_xy: dict, mask: np.ndarray, origin_x: float, origin_y: float,
                        pixel_m: float) -> dict:
    """
    Count health / education facilities standing on flooded pixels.
    points_xy: {"hospital": (N,2) array, "school": (M,2) array, ...} in the grid CRS (metres).
    Points outside the grid are ignored (and reported as 'outside_aoi').
    """
    h, w = mask.shape
    out = {}
    for name, xy in points_xy.items():
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        col = np.floor((xy[:, 0] - origin_x) / pixel_m).astype(int)
        row = np.floor((origin_y - xy[:, 1]) / pixel_m).astype(int)
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        out[name] = {"total_in_aoi": int(inside.sum()),
                     "in_flood": int(mask[row[inside], col[inside]].sum()),
                     "outside_aoi": int((~inside).sum())}
    return out
