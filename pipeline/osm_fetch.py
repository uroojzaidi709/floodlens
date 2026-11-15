"""
osm_fetch.py - OpenStreetMap roads and health/education facilities for the flood AOI.

Needs: osmnx >= 2, geopandas (imported lazily). Needs internet, and for a 1.4 x 1.7 degree
box the Overpass query can take several minutes. The result is cached as .npz so you run
it once.

Output arrays are in the stack's projected CRS (metres), which is what exposure.py expects:
    roads_<name>.npz       line_0, line_1, ...   each (N, 2) x,y
    facilities_<name>.npz  health, education     each (M, 2) x,y
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential"]
HEALTH = ["hospital", "clinic", "doctors"]
EDUCATION = ["school", "college", "university"]


def fetch_osm(bounds_lonlat: Tuple[float, float, float, float], epsg: int
              ) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    """bounds_lonlat = (west, south, east, north). Returns (road_lines, facilities)."""
    import osmnx as ox

    west, south, east, north = bounds_lonlat
    bbox = (west, south, east, north)             # osmnx >= 2 order: left, bottom, right, top

    r = ox.features_from_bbox(bbox=bbox, tags={"highway": ROAD_CLASSES})
    r = r[r.geom_type.isin(["LineString", "MultiLineString"])].to_crs(epsg=epsg)
    lines: List[np.ndarray] = []
    for g in r.geometry:
        parts = [g] if g.geom_type == "LineString" else list(g.geoms)
        lines.extend(np.asarray(p.coords, dtype=float)[:, :2] for p in parts)

    f = ox.features_from_bbox(bbox=bbox, tags={"amenity": HEALTH + EDUCATION})
    f = f.to_crs(epsg=epsg)
    pts = f.geometry.centroid                      # polygons (building footprints) -> a point
    amenity = f["amenity"].astype(str)
    fac = {
        "health": np.column_stack([pts[amenity.isin(HEALTH)].x, pts[amenity.isin(HEALTH)].y]),
        "education": np.column_stack([pts[amenity.isin(EDUCATION)].x, pts[amenity.isin(EDUCATION)].y]),
    }
    return lines, fac


def save_cache(out_dir: str | Path, name: str, lines, fac) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"roads_{name}.npz", **{f"line_{i}": l for i, l in enumerate(lines)})
    np.savez_compressed(out / f"facilities_{name}.npz", **fac)


def load_cache(out_dir: str | Path, name: str):
    out = Path(out_dir)
    rz, fz = np.load(out / f"roads_{name}.npz"), np.load(out / f"facilities_{name}.npz")
    return [rz[k] for k in rz.files], {k: fz[k] for k in fz.files}


def get_osm(bounds_lonlat, epsg: int, cache_dir: str | Path, name: str):
    """Load from cache if present, else download and cache."""
    cdir = Path(cache_dir)
    if (cdir / f"roads_{name}.npz").exists() and (cdir / f"facilities_{name}.npz").exists():
        return load_cache(cdir, name)
    lines, fac = fetch_osm(bounds_lonlat, epsg)
    save_cache(cdir, name, lines, fac)
    return lines, fac
