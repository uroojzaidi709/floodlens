"""
run_pipeline.py - one command from GEE export to the files the Streamlit app reads.

    python pipeline/run_pipeline.py \
        --stack data/gee/sindh2022_stack.tif --scenes data/gee/sindh2022_scenes.csv \
        --event-id sindh2022 --nisar-summary data/nisar_aoi/nisar_dadu_summary.json \
        --nisar-dir data/nisar_aoi --model models/flood_lgbm.joblib --osm-fetch

Data sources used (see README): Sentinel-1 + JRC + Copernicus DEM + WorldPop + WorldCover come
inside the GEE stack; NISAR from nisar_extract.py; OSM via osm_fetch.py (--osm-fetch);
Sen1Floods11 via train_model.py (the --model file).

Outputs (outputs/<event-id>/):
    flood_threshold.tif / flood_ml.tif / ml_probability.tif   rasters on the stack's grid
    summary.json            every number the app shows  (the contract with the frontend)
    provenance_manifest.json  hashes + geometry audit
    shap_importance.png     (only if --model is given)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import exposure
import explain
import baseline_otsu as fm
import forensics as fx
import ml
import web_layers as wl

BANDS = ["vv_pre", "vh_pre", "vv_post", "vh_post", "slope_deg", "jrc_occurrence",
         "pop_count", "pop_cell_m2", "worldcover"]


@dataclass
class Stack:
    """Bands on one square-pixel grid. origin = top-left corner in the grid CRS (metres)."""
    bands: Dict[str, np.ndarray]
    pixel_m: float
    origin_x: float
    origin_y: float
    crs: str


def load_stack(path: str | Path) -> Stack:
    import rasterio
    with rasterio.open(path) as src:
        if src.count != len(BANDS):
            raise ValueError(f"{path} has {src.count} bands, expected {len(BANDS)}: {BANDS}")
        desc = [d for d in src.descriptions]
        if any(desc) and [d for d in desc] != BANDS:
            print(f"WARNING: band names {desc} differ from expected {BANDS}; using file order")
        t = src.transform
        if abs(abs(t.a) - abs(t.e)) > 1e-6 or src.crs is None or not src.crs.is_projected:
            raise ValueError("Stack must be in a projected CRS with square pixels (export EPSG:32642)")
        data = src.read().astype("float32")
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        return Stack({b: data[i] for i, b in enumerate(BANDS)}, abs(t.a), t.c, t.f, str(src.crs))


def write_raster(path: Path, arr: np.ndarray, stack: Stack) -> Path:
    """GeoTIFF via rasterio; if rasterio is missing, a .npy so the pipeline still completes."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        path = path.with_suffix(".npy")
        np.save(path, arr)
        return path
    a = arr.astype("float32") if arr.dtype != bool else arr.astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                       dtype=a.dtype, crs=stack.crs, compress="deflate",
                       transform=from_origin(stack.origin_x, stack.origin_y, stack.pixel_m, stack.pixel_m)) as d:
        d.write(a, 1)
    return path


def run(stack: Stack, scenes_csv: str | Path, out_dir: str | Path, event_id: str,
        model_path: Optional[str | Path] = None, nisar_summary: Optional[str | Path] = None,
        nisar_dir: Optional[str | Path] = None, roads_xy: Optional[List[np.ndarray]] = None,
        facilities_xy: Optional[Dict[str, np.ndarray]] = None,
        params: fm.MaskParams = fm.MaskParams(), expected_pop_total: Optional[float] = None,
        stack_path: Optional[str | Path] = None, warp=None) -> Dict:
    out = Path(out_dir) / event_id
    out.mkdir(parents=True, exist_ok=True)
    b, px = stack.bands, stack.pixel_m

    # 1. geometry parity (fail loudly in the manifest, never silently)
    scenes = fx.read_gee_scene_table(scenes_csv)
    pre = [s for s in scenes if s.frame == "pre"]
    post = [s for s in scenes if s.frame == "post"]

    # 2. threshold flood map (the transparent baseline)
    base = fm.map_flood(b["vh_pre"], b["vh_post"], b["slope_deg"], b["jrc_occurrence"], params)
    mask = base["mask"]
    valid = np.isfinite(b["vh_pre"]) & np.isfinite(b["vh_post"])
    files = {"flood_threshold": write_raster(out / "flood_threshold.tif", mask, stack)}

    summary: Dict = {
        "event_id": event_id, "pixel_m": px, "crs": stack.crs, "shape": list(mask.shape),
        "origin_xy": [stack.origin_x, stack.origin_y],
        "aoi_km2": round(float(valid.sum()) * px * px / 1e6, 2),
        "threshold_map": {"flood_km2": round(exposure.area_km2(mask, px), 2),
                          "funnel": base["funnel"], "threshold": base["threshold"],
                          "params": base["params"]},
        "scope": ("Maps water seen at satellite overpass time. It does not forecast floods and "
                  "has no ground-truth accuracy for Sindh 2022."),
    }

    # 3. optional ML map + SHAP
    final_mask, ml_mask = mask, None
    if model_path:
        model = ml.load_model(model_path)
        prob = ml.predict_probability(model, b["vv_post"], b["vh_post"])
        ml_raw = np.nan_to_num(prob, nan=0.0) >= 0.5
        ml_raw &= valid & (np.nan_to_num(b["jrc_occurrence"]) <= params.jrc_occurrence_max)
        ml_mask, n_steep = fx.quarantine_steep_pixels(ml_raw, b["slope_deg"], params.max_slope_deg)
        files["ml_probability"] = write_raster(out / "ml_probability.tif", prob, stack)
        files["flood_ml"] = write_raster(out / "flood_ml.tif", ml_mask, stack)
        rep_path = Path(model_path).with_suffix(".report.json")
        summary["ml_map"] = {
            "flood_km2": round(exposure.area_km2(ml_mask, px), 2),
            "steep_pixels_removed": n_steep,
            "agreement_with_threshold": fm.agreement(ml_mask, mask, valid),
            "heldout_validation": json.loads(rep_path.read_text()) if rep_path.exists() else None,
        }
        cube = ml.build_features(b["vv_post"], b["vh_post"]).reshape(-1, len(ml.FEATURES))
        pick = np.flatnonzero(np.isfinite(cube).all(axis=1) & (ml_raw.ravel() | mask.ravel()))
        pick = pick if pick.size else np.flatnonzero(np.isfinite(cube).all(axis=1))
        rng = np.random.default_rng(0)
        pick = rng.choice(pick, min(5000, pick.size), replace=False)
        res = explain.explain_pixels(model, cube[pick], mask.ravel()[pick])
        png = explain.save_importance_plot(res, out / "shap_importance.png")
        files["shap_importance"] = png
        summary["explainability"] = {"method": res["method"], "importance": res["importance"],
                                     "per_pixel": res["per_pixel"]}

    # 4. exposure (computed on the threshold map; swap `final_mask` if you prefer ML)
    people = exposure.people_exposed(final_mask, np.nan_to_num(b["pop_count"]),
                                     np.nan_to_num(b["pop_cell_m2"]), px)
    summary["exposure"] = {
        "map_used": "threshold",
        "people_in_flood": round(people["people_in_flood"]),
        "people_in_aoi": round(people["people_in_aoi"]),
        "landcover_km2": exposure.landcover_exposed(final_mask, b["worldcover"], px),
    }
    if expected_pop_total:
        diff = abs(people["people_in_aoi"] - expected_pop_total) / expected_pop_total
        summary["exposure"]["population_check"] = {
            "gee_total": expected_pop_total, "relative_difference": round(diff, 4),
            "ok": diff < 0.05}
    if roads_xy is not None:
        summary["exposure"]["roads"] = exposure.roads_in_flood(
            roads_xy, final_mask, stack.origin_x, stack.origin_y, px)
    if facilities_xy is not None:
        summary["exposure"]["facilities"] = exposure.facilities_in_flood(
            facilities_xy, final_mask, stack.origin_x, stack.origin_y, px)

    # 5. NISAR block
    nisar = None
    if nisar_summary:
        nisar = fx.nisar_block(nisar_summary)
        if nisar_dir:
            nisar["file_integrity"] = fx.verify_hash_table(nisar_summary, nisar_dir)
        summary["nisar"] = nisar

    # 5b. web overlays for the frontend (need rasterio; skipped with a note if unavailable)
    try:
        kw = {"warp": warp} if warp else {}
        grid = (stack.origin_x, stack.origin_y, px, stack.crs)
        summary["layers"] = wl.build_event_layers(
            out, grid, b, {"flood_threshold": mask, "flood_ml": ml_mask}, **kw)["folder"]
        if nisar_summary and nisar_dir:
            tag = json.loads(Path(nisar_summary).read_text())["tag"]
            wl.build_nisar_layers(nisar_dir, tag, out, **kw)
            summary["nisar_layers"] = "layers"
    except ImportError as e:
        summary["layers"] = None
        print(f"WARNING: web layers skipped ({e}). Install rasterio and rerun.")

    # 6. provenance manifest: hashes INPUTS (stack, scene table, NISAR products, model) and outputs
    artifacts = {f"output_{k}": v for k, v in files.items()}
    artifacts["input_scene_table"] = Path(scenes_csv)
    if stack_path:
        artifacts["input_gee_stack"] = Path(stack_path)
    if model_path:
        artifacts["input_ml_model"] = Path(model_path)
    if nisar_summary and nisar_dir:
        for name in json.loads(Path(nisar_summary).read_text()).get("files", {}):
            if name.endswith(".tif") and (Path(nisar_dir) / name).exists():
                artifacts[f"input_nisar_{name}"] = Path(nisar_dir) / name
    manifest = fx.build_manifest(
        event_id, pre, post, artifacts,
        qc={"max_slope_deg": params.max_slope_deg,
            "steep_pixels_quarantined": base["funnel"].get("removed_steep_terrain", 0)},
        parameters=base["params"], nisar=nisar)
    mp = fx.write_manifest(manifest, out / "provenance_manifest.json")
    summary["forensics"] = {"status": manifest["status"],
                            "geometry_audit": manifest["geometry_audit"],
                            "manifest": mp.name, "disclaimer": manifest["disclaimer"]}

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def stack_bounds_lonlat(stack: Stack):
    """(west, south, east, north) of the stack in degrees, for the OSM query."""
    from pyproj import Transformer
    h, w = next(iter(stack.bands.values())).shape
    xs = [stack.origin_x, stack.origin_x + w * stack.pixel_m]
    ys = [stack.origin_y, stack.origin_y - h * stack.pixel_m]
    t = Transformer.from_crs(stack.crs, 4326, always_xy=True)
    lon, lat = t.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    return min(lon), min(lat), max(lon), max(lat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--event-id", default="sindh2022")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model")
    ap.add_argument("--nisar-summary")
    ap.add_argument("--nisar-dir")
    ap.add_argument("--roads", help=".npz with arrays line_0, line_1, ... (x,y in the stack CRS)")
    ap.add_argument("--facilities", help=".npz with arrays health, education (x,y in the stack CRS)")
    ap.add_argument("--osm-fetch", action="store_true",
                    help="download roads + facilities from OpenStreetMap (cached in data/osm)")
    ap.add_argument("--expected-pop-total", type=float)
    ap.add_argument("--otsu", action="store_true", help="derive the water threshold with Otsu")
    a = ap.parse_args()

    stack = load_stack(a.stack)
    roads = fac = None
    if a.osm_fetch:
        import osm_fetch
        roads, fac = osm_fetch.get_osm(stack_bounds_lonlat(stack), int(stack.crs.split(":")[-1]),
                                       "data/osm", a.event_id)
    if a.roads:
        z = np.load(a.roads)
        roads = [z[k] for k in z.files]
    if a.facilities:
        z = np.load(a.facilities)
        fac = {k: z[k] for k in z.files}
    s = run(stack, a.scenes, a.out, a.event_id, a.model, a.nisar_summary, a.nisar_dir,
            roads, fac, fm.MaskParams(use_otsu=a.otsu), a.expected_pop_total, stack_path=a.stack)
    print(json.dumps({k: s[k] for k in ("threshold_map", "exposure", "forensics") if k in s}, indent=2)[:3000])


if __name__ == "__main__":
    main()
