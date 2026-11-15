"""
services.py - the ONLY module the Streamlit frontend needs to import.

The frontend never touches rasters, models or hashes directly:

    from services import list_events, load_summary, load_layers, funnel_rows,
                         tamper_check, check_uploaded_file, explain_what_if

Everything reads files that run_pipeline.py wrote to outputs/<event>/. No server, no database.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import forensics as fx

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def list_events(outputs_dir: str | Path = OUTPUTS) -> List[str]:
    d = Path(outputs_dir)
    return sorted(p.name for p in d.iterdir() if (p / "summary.json").exists()) if d.exists() else []


def load_summary(event: str, outputs_dir: str | Path = OUTPUTS) -> Dict:
    return json.loads((Path(outputs_dir) / event / "summary.json").read_text(encoding="utf-8"))


def load_manifest(event: str, outputs_dir: str | Path = OUTPUTS) -> Dict:
    return json.loads((Path(outputs_dir) / event / "provenance_manifest.json").read_text(encoding="utf-8"))


def load_layers(event: str, kind: str = "event", outputs_dir: str | Path = OUTPUTS) -> Optional[Dict]:
    """
    kind = "event" (Sentinel-1 + flood masks) or "nisar".
    Returns {"bounds_leaflet": [[s, w], [n, e]], "layers": {name: {"path": abs path, "label": str}}}
    or None if layers were not generated (rasterio/pyproj missing when the pipeline ran).
    """
    base = Path(outputs_dir) / event / "layers"
    f = base / ("layers.json" if kind == "event" else "nisar_layers.json")
    if not f.exists():
        return None
    info = json.loads(f.read_text(encoding="utf-8"))
    for v in info["layers"].values():
        v["path"] = str(base / v["file"])
    return info


def funnel_rows(summary: Dict) -> List[Dict]:
    """Ordered rows for the 'how we cleaned the map' bar chart."""
    f = summary["threshold_map"]["funnel"]
    steps = [("Dark in flood image", "dark_in_post_image"),
             ("...and darker than before", "and_darker_than_before"),
             ("...not permanent water (JRC)", "after_permanent_water_removed"),
             ("...not steep terrain (DEM)", "after_terrain_qc"),
             ("...speckle removed (final)", "final_flood_pixels")]
    px = summary["pixel_m"]
    return [{"step": label, "pixels": f[k], "km2": round(f[k] * px * px / 1e6, 2)}
            for label, k in steps if k in f]


def tamper_check(event: str, base_dir: str | Path = ".", outputs_dir: str | Path = OUTPUTS) -> Dict:
    """Re-hash every file in the manifest. 'modified' lists files that changed since the run."""
    return fx.verify_manifest(load_manifest(event, outputs_dir), base_dir)


def check_uploaded_file(event: str, label: str, data: bytes, outputs_dir: str | Path = OUTPUTS) -> Dict:
    """
    Tamper demo for the app: the visitor uploads (or edits) a file and we compare its SHA-256
    with the one in the manifest. Returns {'label', 'expected', 'actual', 'match'}.
    """
    import hashlib
    m = load_manifest(event, outputs_dir)
    if label not in m["artifact_sha256"]:
        raise KeyError(f"'{label}' not in manifest; choose one of {sorted(m['artifact_sha256'])}")
    actual = hashlib.sha256(data).hexdigest()
    exp = m["artifact_sha256"][label]
    return {"label": label, "expected": exp, "actual": actual, "match": actual == exp}


def explain_what_if(model_path: str | Path, vv_db: float, vh_db: float,
                    vv_texture: float = 1.0, vh_texture: float = 1.0) -> Dict:
    """
    'What if a pixel had these radar values?' -> flood probability + SHAP contribution per feature.
    Powers the sliders on the Explainability page. Needs `shap` or `lightgbm` installed.
    """
    import numpy as np
    import explain
    import ml
    model = ml.load_model(model_path)
    row = np.array([vv_db, vh_db, vv_db - vh_db, vh_texture, vv_texture], dtype="float32")
    return explain.explain_one(model, row)
