"""
forensics.py - file-integrity and acquisition-geometry audit for FloodLens.

WHAT THIS MODULE PROVES
    * Integrity: SHA-256 of every input/output file. If one byte changes, the hash changes.
    * Geometry parity: before/after scenes come from the same orbit track and the same
      flight direction (otherwise a before/after comparison is not like-for-like).
    * Terrain QC: water pixels on steep slopes are quarantined (slope is only a PROXY for
      radar shadow, see quarantine_steep_pixels).

WHAT THIS MODULE DOES NOT PROVE
    It does not prove the flood map is scientifically correct. A perfectly intact file can
    still contain a wrong map. Say this on the slide.

Inputs
    Sentinel-1 : a CSV exported from Google Earth Engine (see gee/export_stack.js) with one
                 row per source scene. NISAR : the scene file names (they encode track,
                 direction, frame and cycle).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

NISAR_REPEAT_DAYS = 12  # NISAR nominal repeat cycle

# What each data source is and where it came from. Written into the manifest so the Forensic page
# can show provenance for EVERY dataset. These strings are what we DECLARE; the code cannot
# check them against the files (only the hashes are verified).
DATASETS: Dict[str, Dict[str, str]] = {
    "sentinel1": {"name": "Sentinel-1 IW GRD, VV+VH (ESA Copernicus)", "access": "GEE COPERNICUS/S1_GRD",
                  "role": "main flood evidence (C-band, 2022)"},
    "nisar": {"name": "NISAR L2 GCOV, L-band HH/HV (NASA-ISRO)", "access": "ASF DAAC, NISAR_L2_GCOV_PROVISIONAL_V1",
              "role": "L-band comparison (2026)"},
    "jrc_surface_water": {"name": "JRC Global Surface Water v1.4", "access": "GEE JRC/GSW1_4/GlobalSurfaceWater",
                          "role": "remove permanent water", "cite": "Pekel et al. 2016, Nature"},
    "copernicus_dem": {"name": "Copernicus GLO-30 DEM", "access": "GEE COPERNICUS/DEM/GLO30",
                       "role": "slope for terrain QC"},
    "worldpop": {"name": "WorldPop 100 m population, 2020", "access": "GEE WorldPop/GP/100m/pop",
                 "role": "people exposed"},
    "worldcover": {"name": "ESA WorldCover v200", "access": "GEE ESA/WorldCover/v200",
                   "role": "cropland / built-up exposed"},
    "osm": {"name": "OpenStreetMap (ODbL)", "access": "Overpass API via osmnx",
            "role": "roads, hospitals, schools exposed"},
    "sen1floods11": {"name": "Sen1Floods11 hand-labelled chips", "access": "gs://sen1floods11",
                     "role": "train/evaluate ML", "cite": "Bonafilia et al. 2020, CVPRW"},
}


# --------------------------------------------------------------------------- hashing
def sha256_file(path: str | Path, chunk_mb: int = 8) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_mb << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- scenes
@dataclass
class Scene:
    """One SAR acquisition, reduced to what the geometry check needs."""
    sensor: str            # "Sentinel-1" or "NISAR"
    scene_id: str
    start_utc: str         # ISO 8601
    track: str             # relative orbit (S1) / track (NISAR)
    direction: str         # ASCENDING / DESCENDING
    frame: Optional[str] = None
    cycle: Optional[int] = None
    polarisation: Optional[str] = None


def parse_nisar_name(name: str) -> Scene:
    """
    NISAR_L2_PR_GCOV_023_092_D_075_2005_QPDH_A_20260620T141344_20260620T141418_P05023_N_F_J_001
    tokens:  4=cycle 5=track 6=direction(A/D) 7=frame 9=pol mode 11=start 12=stop
    """
    stem = Path(name).name
    stem = re.sub(r"\.h5$", "", stem)
    t = stem.split("_")
    if len(t) < 13 or t[0] != "NISAR":
        raise ValueError(f"Not a NISAR product name: {name}")
    try:
        start = datetime.strptime(t[11], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        cycle = int(t[4])
    except ValueError as e:
        raise ValueError(f"Cannot parse NISAR name {name}: {e}") from e
    direction = {"A": "ASCENDING", "D": "DESCENDING"}.get(t[6], t[6])
    return Scene("NISAR", stem, start.isoformat(), t[5], direction, frame=t[7],
                 cycle=cycle, polarisation=t[9])


def read_gee_scene_table(csv_path: str | Path) -> List[Scene]:
    """
    Read the scene table exported by gee/export_stack.js.
    Expected columns: scene_id, relativeOrbitNumber_start, orbitProperties_pass,
                      time_start_ms (ms since epoch), role (pre|post)
    Returns scenes; the 'role' is stored in Scene.frame to keep the dataclass small.
    """
    scenes: List[Scene] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ms = float(row.get("time_start_ms") or row["system:time_start"])
            start = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
            scenes.append(Scene("Sentinel-1", row.get("scene_id") or row["system:index"], start,
                                str(int(float(row["relativeOrbitNumber_start"]))),
                                row["orbitProperties_pass"].strip().upper(),
                                frame=row.get("role", "").strip() or None,
                                polarisation=row.get("transmitterReceiverPolarisation") or None))
    if not scenes:
        raise ValueError(f"No scenes in {csv_path}")
    return scenes


# --------------------------------------------------------------------------- parity
def check_geometry_parity(pre: List[Scene], post: List[Scene]) -> Dict:
    """All scenes (pre and post) must share one track and one flight direction."""
    if not pre or not post:
        raise ValueError("Need at least one pre and one post scene")
    tracks = {s.track for s in pre + post}
    dirs = {s.direction for s in pre + post}
    frames = {s.frame for s in pre + post if s.sensor == "NISAR"}
    ok = len(tracks) == 1 and len(dirs) == 1 and len(frames) <= 1

    def d(s: Scene) -> datetime:
        return datetime.fromisoformat(s.start_utc)

    gap_days = (min(d(s) for s in post) - max(d(s) for s in pre)).days

    report = {
        "passed": ok,
        "tracks_found": sorted(tracks),
        "directions_found": sorted(dirs),
        "frames_found": sorted(f for f in frames if f),
        "n_pre": len(pre), "n_post": len(post),
        "days_between_pre_and_post": gap_days,
        "note": ("PASS: all scenes share one track and one flight direction."
                 if ok else
                 "FLAG: mixed track/direction. Before/after change may reflect viewing "
                 "geometry, not the ground."),
    }
    # NISAR only: cycle numbers should be consistent with the dates (12-day repeat).
    nisar = [s for s in pre + post if s.cycle is not None]
    if len(nisar) >= 2:
        first = min(nisar, key=d)
        last = max(nisar, key=d)
        expected = (last.cycle - first.cycle) * NISAR_REPEAT_DAYS
        actual = (d(last) - d(first)).days
        report["nisar_cycle_check"] = {
            "cycles": [first.cycle, last.cycle],
            "expected_days": expected, "actual_days": actual,
            "consistent": abs(expected - actual) <= 2,
        }
    return report


# --------------------------------------------------------------------------- terrain QC
def quarantine_steep_pixels(water: np.ndarray, slope_deg: np.ndarray,
                            max_slope_deg: float = 5.0) -> Tuple[np.ndarray, int]:
    """
    Remove 'water' pixels that sit on steep terrain. Standing floodwater cannot rest on a
    slope, so such dark pixels are more likely radar shadow or smooth rock/pavement.
    NOTE: this is a slope-only proxy. True shadow depends on slope relative to the radar
    look direction, so some real shadow survives and some valid pixels are removed.
    Default 5 deg suits the flat Indus plain; keep ONE value across code and slides.
    """
    if water.shape != slope_deg.shape:
        raise ValueError(f"shape mismatch: water {water.shape} vs slope {slope_deg.shape}")
    water = water.astype(bool)
    steep = water & (np.nan_to_num(slope_deg, nan=0.0) > max_slope_deg)
    return water & ~steep, int(steep.sum())


# --------------------------------------------------------------------------- manifest
def build_manifest(event_id: str, pre: List[Scene], post: List[Scene],
                   artifacts: Dict[str, str | Path],
                   qc: Optional[Dict] = None, parameters: Optional[Dict] = None,
                   nisar: Optional[Dict] = None) -> Dict:
    """
    Assemble provenance_manifest.json content. `artifacts` = {label: file path}; pass INPUT files
    (stack, scene table, NISAR products, model) as well as outputs, labels starting 'input_'.
    """
    parity = check_geometry_parity(pre, post)
    hashes = {label: sha256_file(p) for label, p in artifacts.items()}
    m = {
        "schema_version": "2.0",
        "project": "FloodLens - NASA Space Apps 2026",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_id": event_id,
        "status": "CHECKS_PASSED" if parity["passed"] else "FLAGGED",
        "sources": {"pre": [asdict(s) for s in pre], "post": [asdict(s) for s in post]},
        "geometry_audit": parity,
        "terrain_qc": qc or {},
        "parameters": parameters or {},
        "datasets": DATASETS,
        "artifact_sha256": hashes,
        "artifact_paths": {k: str(v) for k, v in artifacts.items()},
        "disclaimer": ("SHA-256 shows files were not altered after this manifest was made. "
                       "It does not show that the flood map is correct. Flood extent is the "
                       "water seen at satellite overpass time only."),
    }
    if nisar:
        m["nisar"] = nisar
    return m


def write_manifest(manifest: Dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def verify_manifest(manifest: Dict | str | Path, base_dir: str | Path = ".") -> Dict:
    """
    Re-hash every artifact and compare. This is the 'tamper test' for the app:
    change one byte of a file, run this, and the file shows up under 'modified'.
    """
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    base = Path(base_dir)
    ok, modified, missing = [], [], []
    for label, expected in manifest["artifact_sha256"].items():
        p = Path(manifest["artifact_paths"][label])
        p = p if p.is_absolute() else base / p
        if not p.is_file():
            missing.append(label)
        elif sha256_file(p) == expected:
            ok.append(label)
        else:
            modified.append(label)
    return {"all_intact": not modified and not missing,
            "intact": ok, "modified": modified, "missing": missing}


def verify_hash_table(summary_json: str | Path, folder: str | Path) -> Dict:
    """Check a {'files': {name: sha256}} table such as nisar_dadu_summary.json."""
    files = json.loads(Path(summary_json).read_text(encoding="utf-8"))["files"]
    ok, modified, missing = [], [], []
    for name, expected in files.items():
        p = Path(folder) / name
        if not p.is_file():
            missing.append(name)
        elif sha256_file(p) == expected:
            ok.append(name)
        else:
            modified.append(name)
    return {"all_intact": not modified and not missing,
            "intact": ok, "modified": modified, "missing": missing}


def nisar_block(summary_json: str | Path) -> Dict:
    """Parse the scene names out of nisar_<tag>_summary.json and run the parity check."""
    s = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    pre, post = parse_nisar_name(s["scenes"]["before"]), parse_nisar_name(s["scenes"]["after"])
    return {"scenes": {"before": asdict(pre), "after": asdict(post)},
            "geometry_audit": check_geometry_parity([pre], [post]),
            "low_signal_mask_pct_of_aoi": s.get("low_signal_mask", {}).get("pct_of_aoi"),
            "stats": s.get("polarizations", {})}


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("usage: python forensics.py <nisar_summary.json> <folder_with_tifs>")
    print(json.dumps(nisar_block(sys.argv[1]), indent=2))
    print(json.dumps(verify_hash_table(sys.argv[1], sys.argv[2]), indent=2))
