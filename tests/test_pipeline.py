import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import exposure
import explain
import baseline_otsu as fm
import forensics as fx
import ml

RNG = np.random.default_rng(1)
N = 200


def synthetic_scene():
    """Known truth: flood block, permanent river, steep 'shadow' block, plus speckle."""
    pre = RNG.normal(-15, 1.5, (N, N)).astype("float32")
    post = pre + RNG.normal(0, 0.8, (N, N)).astype("float32")
    slope = np.zeros((N, N), "float32")
    jrc = np.zeros((N, N), "float32")

    flood = np.zeros((N, N), bool); flood[20:80, 20:90] = True
    river = np.zeros((N, N), bool); river[100:110, :] = True
    shadow = np.zeros((N, N), bool); shadow[150:180, 150:190] = True

    post[flood] = RNG.normal(-24, 1.0, flood.sum())
    post[river] = RNG.normal(-25, 1.0, river.sum()); jrc[river] = 100
    pre[100:110, 100:] = RNG.normal(-25, 1.0, (10, N - 100))   # right half always wet; left half was dry
    post[shadow] = RNG.normal(-24, 1.0, shadow.sum()); slope[shadow] = 25   # dark only in post
    post[5, 150] = -30                                                         # lone speckle pixel
    return pre, post, slope, jrc, flood, river, shadow


def test_map_flood_recovers_truth_and_funnel_is_monotonic():
    pre, post, slope, jrc, flood, river, shadow = synthetic_scene()
    r = fm.map_flood(pre, post, slope, jrc)
    m = r["mask"]
    assert fm.agreement(m, flood)["iou"] > 0.9
    assert not (m & river).any(), "permanent water must be removed"
    assert not (m & shadow).any(), "steep dark pixels must be quarantined"
    f = r["funnel"]
    assert f["removed_steep_terrain"] > 0 and f["removed_permanent_water"] > 0
    assert f["dark_in_post_image"] >= f["and_darker_than_before"] >= f["after_permanent_water_removed"] \
        >= f["after_terrain_qc"] >= f["final_flood_pixels"]


def test_nan_pixels_are_not_flood():
    pre, post, *_ = synthetic_scene()
    post[:, :10] = np.nan
    m = fm.map_flood(pre, post)["mask"]
    assert not m[:, :10].any()


def test_otsu_falls_back_when_outside_plausible_range():
    a = np.random.default_rng(0).normal(-8, 1, 10000)    # no water mode at all
    out = fm.otsu_threshold_db(a)
    assert out["method"] == "fallback_fixed" and out["threshold_db"] == -18.0


def test_quarantine_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fx.quarantine_steep_pixels(np.zeros((3, 3), bool), np.zeros((4, 4)))


# ------------------------------------------------------------------ forensics
NISAR_A = "NISAR_L2_PR_GCOV_023_092_D_075_2005_QPDH_A_20260620T141344_20260620T141418_P05023_N_F_J_001"
NISAR_B = "NISAR_L2_PR_GCOV_029_092_D_075_2005_QPDH_A_20260831T141340_20260831T141415_P05023_N_F_J_001"


def test_nisar_name_parsing_and_parity_pass():
    a, b = fx.parse_nisar_name(NISAR_A), fx.parse_nisar_name(NISAR_B + ".h5")
    assert (a.track, a.direction, a.frame, a.cycle) == ("092", "DESCENDING", "075", 23)
    rep = fx.check_geometry_parity([a], [b])
    assert rep["passed"] and rep["nisar_cycle_check"]["consistent"]


def test_parity_flags_mixed_direction_and_track():
    a = fx.parse_nisar_name(NISAR_A)
    b = fx.parse_nisar_name(NISAR_B.replace("_092_D_", "_093_A_"))
    rep = fx.check_geometry_parity([a], [b])
    assert not rep["passed"] and rep["note"].startswith("FLAG")


def test_bad_nisar_name_raises():
    with pytest.raises(ValueError):
        fx.parse_nisar_name("S1A_IW_GRDH_something")


def write_gee_csv(path, rows):
    path.write_text("system:index,relativeOrbitNumber_start,orbitProperties_pass,"
                    "system:time_start,role,transmitterReceiverPolarisation\n" +
                    "\n".join(",".join(map(str, r)) for r in rows))


def test_gee_table_parity_and_manifest_tamper_test(tmp_path):
    csv = tmp_path / "scenes.csv"
    write_gee_csv(csv, [
        ("S1A_a", 78.0, "ASCENDING", 1652400000000, "pre", "[VV, VH]"),
        ("S1A_b", 78.0, "ASCENDING", 1661750000000, "post", "[VV, VH]")])
    sc = fx.read_gee_scene_table(csv)
    pre = [s for s in sc if s.frame == "pre"]; post = [s for s in sc if s.frame == "post"]

    art = tmp_path / "mask.bin"; art.write_bytes(b"flood-mask-v1")
    man = fx.build_manifest("t", pre, post, {"mask": art})
    assert man["status"] == "CHECKS_PASSED"
    mp = fx.write_manifest(man, tmp_path / "provenance_manifest.json")

    assert fx.verify_manifest(mp)["all_intact"]
    art.write_bytes(b"flood-mask-v2")                 # tamper
    rep = fx.verify_manifest(mp)
    assert not rep["all_intact"] and rep["modified"] == ["mask"]
    art.unlink()
    assert fx.verify_manifest(mp)["missing"] == ["mask"]


def test_manifest_flagged_when_orbits_differ(tmp_path):
    csv = tmp_path / "s.csv"
    write_gee_csv(csv, [("a", 78, "ASCENDING", 1652400000000, "pre", "x"),
                        ("b", 7, "ASCENDING", 1661750000000, "post", "x")])
    sc = fx.read_gee_scene_table(csv)
    f = tmp_path / "f"; f.write_bytes(b"x")
    man = fx.build_manifest("t", sc[:1], sc[1:], {"f": f})
    assert man["status"] == "FLAGGED"


# ------------------------------------------------------------------ exposure
def test_area_and_people_conserve_counts():
    big = np.zeros((90, 90), bool); big[:30, :30] = True       # 900 px of 30 m = 0.81 km2
    assert exposure.area_km2(big, 30) == pytest.approx(0.81)
    # ONE native cell = 90 m x 90 m = 8100 m2 holding 1000 people = a 3x3 block of our pixels
    m = np.zeros((3, 3), bool); m[0, 0] = True
    pop = np.full((3, 3), 1000.0); cell = np.full((3, 3), 8100.0)
    r = exposure.people_exposed(m, pop, cell, 30)
    assert r["people_in_aoi"] == pytest.approx(1000.0)        # one cell, not 9 x 1000
    assert r["people_in_flood"] == pytest.approx(1000.0 / 9)


def test_roads_length_in_flood():
    m = np.zeros((100, 100), bool); m[:, 50:] = True          # east half flooded, 30 m pixels
    road = np.array([[0.0, -1500.0], [3000.0, -1500.0]])      # 3 km east-west line
    r = exposure.roads_in_flood([road], m, origin_x=0, origin_y=0, pixel_m=30)
    assert r["road_km_in_aoi"] == pytest.approx(3.0, abs=0.02)
    assert r["road_km_flooded"] == pytest.approx(1.5, abs=0.02)


def test_landcover_breakdown():
    m = np.zeros((10, 10), bool); m[:5] = True
    wc = np.full((10, 10), 40.0); wc[:, :5] = 50
    out = exposure.landcover_exposed(m, wc, 30)
    assert out["cropland"] == pytest.approx(25 * 900 / 1e6, rel=1e-6)
    assert out["built-up"] == pytest.approx(25 * 900 / 1e6, rel=1e-6)


# ------------------------------------------------------------------ ML + SHAP
def synthetic_chips():
    chips = []
    for ev in ["A", "B", "C", "Pakistan"]:
        for _ in range(3):
            lab = np.zeros((64, 64), "int16"); lab[10:40, 10:50] = 1; lab[0, :] = -1
            vh = RNG.normal(-15, 2.5, (64, 64)).astype("float32")
            vv = (vh + 7 + RNG.normal(0, 1.5, (64, 64))).astype("float32")
            vh[lab == 1] = RNG.normal(-23, 2, (lab == 1).sum())
            vv[lab == 1] = RNG.normal(-17, 2, (lab == 1).sum())
            chips.append({"event": ev, "vv": vv, "vh": vh, "label": lab})
    return chips


def test_leave_one_event_out_beats_chance_and_never_sees_holdout():
    res = ml.train_leave_one_event_out(synthetic_chips(), "Pakistan", max_per_chip=1500)
    rep = res["report"]
    assert "Pakistan" not in rep["train_events"]
    assert rep["ml"]["f1"] > 0.8 and rep["baseline"]["f1"] > 0.5
    assert 0 < rep["ml"]["water_fraction_true"] < 1


def test_nodata_labels_are_dropped():
    X, y, _ = ml.chips_to_xy(synthetic_chips())
    assert set(np.unique(y)) <= {0, 1} and len(X) == len(y)


def test_predict_probability_shape_and_nan():
    chips = synthetic_chips()
    model = ml.train_leave_one_event_out(chips, "Pakistan")["model"]
    vv, vh = chips[0]["vv"].copy(), chips[0]["vh"].copy()
    vh[0, 0] = np.nan
    p = ml.predict_probability(model, vv, vh)
    assert p.shape == vv.shape and np.isnan(p[0, 0])
    assert np.nanmin(p) >= 0 and np.nanmax(p) <= 1


def test_explain_global_importance_ranks_vh_highly(tmp_path):
    chips = synthetic_chips()
    model = ml.train_leave_one_event_out(chips, "Pakistan")["model"]
    X, y, _ = ml.chips_to_xy(chips[:6], 2000)
    res = explain.explain_pixels(model, X, y)
    top2 = sorted(res["importance"], key=res["importance"].get, reverse=True)[:2]
    assert "vh_db" in top2 or "vv_minus_vh" in top2
    png = explain.save_importance_plot(res, tmp_path / "imp.png")
    assert png.stat().st_size > 1000


# ------------------------------------------------------------------ end to end
def test_run_pipeline_end_to_end(tmp_path):
    import run_pipeline as rp
    pre, post, slope, jrc, flood, river, shadow = synthetic_scene()
    vv_post = post + 7
    pop = np.full((N, N), 500.0, "float32"); cell = np.full((N, N), 9000.0, "float32")
    wc = np.full((N, N), 40.0, "float32")
    stack = rp.Stack({"vv_pre": pre + 7, "vh_pre": pre, "vv_post": vv_post, "vh_post": post,
                      "slope_deg": slope, "jrc_occurrence": jrc, "pop_count": pop,
                      "pop_cell_m2": cell, "worldcover": wc}, 30.0, 400000.0, 3000000.0, "EPSG:32642")
    csv = tmp_path / "s.csv"
    write_gee_csv(csv, [("a", 78, "ASCENDING", 1652400000000, "pre", "VV+VH"),
                        ("b", 78, "ASCENDING", 1661750000000, "post", "VV+VH")])
    chips = synthetic_chips()
    tr = ml.train_leave_one_event_out(chips, "Pakistan")
    ml.save_model(tr["model"], tmp_path / "m.joblib", tr["report"])

    nisar_dir = Path(__file__).resolve().parent.parent / "data" / "nisar_aoi"
    s = rp.run(stack, csv, tmp_path / "out", "t", tmp_path / "m.joblib",
               nisar_dir / "nisar_dadu_summary.json", nisar_dir)

    out = tmp_path / "out" / "t"
    assert (out / "summary.json").exists() and (out / "provenance_manifest.json").exists()
    assert s["forensics"]["status"] == "CHECKS_PASSED"
    assert s["nisar"]["file_integrity"]["all_intact"]
    true_km2 = flood.sum() * 900 / 1e6
    assert s["threshold_map"]["flood_km2"] == pytest.approx(true_km2, rel=0.1)
    assert s["ml_map"]["agreement_with_threshold"]["iou"] > 0.8
    assert s["exposure"]["people_in_aoi"] == pytest.approx(N * N * 500 * 900 / 9000)
    # the manifest verifies, then fails after tampering with an output raster
    man = out / "provenance_manifest.json"
    assert fx.verify_manifest(man)["all_intact"]
    target = Path(json.loads(man.read_text())["artifact_paths"]["output_flood_threshold"])
    target.write_bytes(target.read_bytes() + b"x")
    assert fx.verify_manifest(man)["modified"] == ["output_flood_threshold"]


# ------------------------------------------------------------------ preprocess / OSM / ASF helpers
def test_to_db_and_ensure_db():
    import preprocess as pp
    lin = np.array([[1.0, 0.1, 0.0, np.nan]], "float32")
    db = pp.to_db(lin)
    assert db[0, 0] == pytest.approx(0) and db[0, 1] == pytest.approx(-10, abs=1e-4)
    assert np.isnan(db[0, 2]) and np.isnan(db[0, 3]), "zero power is no-data, not -60 dB"
    assert pp.looks_like_db(np.array([-20.0, -15.0, -8.0]))
    assert np.allclose(pp.ensure_db(lin)[0, :2], db[0, :2])          # linear gets converted
    already = np.array([[-20.0, -15.0, -10.0]], "float32")
    assert np.allclose(pp.ensure_db(already), already)               # dB untouched


def test_slope_from_dem_uses_real_pixel_size():
    import preprocess as pp
    x = np.arange(50) * 30.0
    dem = np.tile(x * np.tan(np.radians(10)), (50, 1))                # 10 degree plane, 30 m pixels
    assert pp.slope_from_dem(dem, 30.0)[25, 25] == pytest.approx(10, abs=0.01)
    assert pp.slope_from_dem(dem, 10.0)[25, 25] > 25                  # wrong spacing = wrong slope


def test_nisar_rgb_shape_range_and_mask():
    import preprocess as pp
    hh = RNG.normal(-12, 3, (40, 40)); hv = hh - 7 + RNG.normal(0, 1, (40, 40))
    low = np.zeros((40, 40), bool); low[:, :5] = True
    rgb = pp.nisar_rgb(hh, hv, low)
    assert rgb.shape == (40, 40, 3) and rgb.min() >= 0 and rgb.max() <= 1
    assert rgb[:, :5].sum() == 0


def test_web_png(tmp_path):
    import preprocess as pp
    a = RNG.normal(-15, 3, (20, 20)); a[0, 0] = np.nan
    p = pp.export_web_png(a, tmp_path / "x.png")
    from PIL import Image
    im = np.array(Image.open(p))
    assert im.shape == (20, 20, 4) and im[0, 0, 3] == 0 and im[5, 5, 3] == 255


def test_facilities_in_flood():
    m = np.zeros((10, 10), bool); m[:, 5:] = True
    pts = {"health": np.array([[50.0, -50.0], [250.0, -50.0], [9999.0, 9999.0]]),
           "school": np.array([[400.0, -400.0]])}
    r = exposure.facilities_in_flood(pts, m, 0.0, 0.0, 100.0)
    assert r["health"] == {"total_in_aoi": 2, "in_flood": 0, "outside_aoi": 1}
    assert r["school"]["in_flood"] == 0 and r["school"]["total_in_aoi"] == 1
    m2 = np.ones((10, 10), bool)
    assert exposure.facilities_in_flood(pts, m2, 0.0, 0.0, 100.0)["health"]["in_flood"] == 2


def test_asf_pair_candidates_groups_by_geometry():
    import download_asf as da
    names = [NISAR_A, NISAR_B, NISAR_A.replace("_092_D_075_", "_093_A_011_"), "garbage_name"]
    groups = da.pair_candidates(names)
    assert list(groups) == [("092", "DESCENDING", "075")]            # singleton + garbage dropped
    assert [s.cycle for s in groups[("092", "DESCENDING", "075")]] == [23, 29]


def test_run_pipeline_with_roads_and_facilities(tmp_path):
    import run_pipeline as rp
    pre, post, slope, jrc, flood, river, shadow = synthetic_scene()
    stack = rp.Stack({"vv_pre": pre + 7, "vh_pre": pre, "vv_post": post + 7, "vh_post": post,
                      "slope_deg": slope, "jrc_occurrence": jrc,
                      "pop_count": np.full((N, N), 500.0, "float32"),
                      "pop_cell_m2": np.full((N, N), 9000.0, "float32"),
                      "worldcover": np.full((N, N), 40.0, "float32")}, 30.0, 0.0, 0.0, "EPSG:32642")
    csv = tmp_path / "s.csv"
    write_gee_csv(csv, [("a", 78, "ASCENDING", 1652400000000, "pre", "x"),
                        ("b", 78, "ASCENDING", 1661750000000, "post", "x")])
    # one road straight through the flood block (rows 20-80, cols 20-90), one hospital inside it
    road = np.array([[0.0, -50 * 30.0], [N * 30.0, -50 * 30.0]])
    hosp = np.array([[50 * 30.0, -50 * 30.0]])
    s = rp.run(stack, csv, tmp_path / "o", "t", roads_xy=[road], facilities_xy={"health": hosp})
    assert s["exposure"]["roads"]["road_km_flooded"] == pytest.approx(70 * 30 / 1000, rel=0.1)
    assert s["exposure"]["facilities"]["health"]["in_flood"] == 1


# ------------------------------------------------------------------ web layers + services
def identity_warp(arr, grid, resampling="nearest"):
    return arr.astype("float32"), (67.0, 26.0, 68.0, 27.0)


def test_event_layers_written_with_leaflet_bounds(tmp_path):
    import web_layers as wl
    from PIL import Image
    pre, post, slope, jrc, flood, *_ = synthetic_scene()
    info = wl.build_event_layers(tmp_path, (0.0, 0.0, 30.0, "EPSG:32642"),
                                 {"vh_pre": pre, "vh_post": post}, {"flood_threshold": flood, "flood_ml": None},
                                 warp=identity_warp)
    assert info["bounds_leaflet"] == [[26.0, 67.0], [27.0, 68.0]]
    assert set(info["layers"]) == {"s1_vh_pre", "s1_vh_post", "flood_threshold"}   # None mask skipped
    im = np.array(Image.open(tmp_path / "layers" / "flood_threshold.png"))
    assert im.shape == (N, N, 4) and im[50, 50, 3] > 0 and im[150, 5, 3] == 0   # flood visible, land transparent


def test_nisar_layers_with_fake_reader(tmp_path):
    import web_layers as wl
    def reader(path):
        name = Path(path).name
        a = RNG.normal(-12, 3, (30, 30)).astype("float32")
        if "low_signal" in name:
            a = np.zeros((30, 30), "float32"); a[:, :4] = 1
        return a, (0.0, 0.0, 20.0, "EPSG:32642")
    info = wl.build_nisar_layers(tmp_path, "dadu", tmp_path, warp=identity_warp, reader=reader)
    assert {"nisar_rgb_before", "nisar_rgb_after", "nisar_hh_change", "nisar_hv_after"} <= set(info["layers"])
    assert len(info["layers"]) == 8      # 3 HH + 3 HV + 2 RGB
    assert (tmp_path / "layers" / "nisar_layers.json").exists()


def test_services_roundtrip_and_tamper_demo(tmp_path):
    import run_pipeline as rp
    import services as sv
    pre, post, slope, jrc, flood, *_ = synthetic_scene()
    stack = rp.Stack({"vv_pre": pre + 7, "vh_pre": pre, "vv_post": post + 7, "vh_post": post,
                      "slope_deg": slope, "jrc_occurrence": jrc,
                      "pop_count": np.full((N, N), 500.0, "float32"),
                      "pop_cell_m2": np.full((N, N), 9000.0, "float32"),
                      "worldcover": np.full((N, N), 40.0, "float32")}, 30.0, 0.0, 0.0, "EPSG:32642")
    csv = tmp_path / "s.csv"
    write_gee_csv(csv, [("a", 78, "ASCENDING", 1652400000000, "pre", "x"),
                        ("b", 78, "ASCENDING", 1661750000000, "post", "x")])
    fake_stack_file = tmp_path / "stack.tif"; fake_stack_file.write_bytes(b"pretend-geotiff")
    out = tmp_path / "out"
    rp.run(stack, csv, out, "evt", stack_path=fake_stack_file, warp=identity_warp)

    assert sv.list_events(out) == ["evt"]
    s = sv.load_summary("evt", out)
    rows = sv.funnel_rows(s)
    assert [r["pixels"] for r in rows] == sorted((r["pixels"] for r in rows), reverse=True)
    lay = sv.load_layers("evt", "event", out)
    assert Path(lay["layers"]["flood_threshold"]["path"]).exists()

    m = sv.load_manifest("evt", out)
    assert "input_gee_stack" in m["artifact_sha256"] and "input_scene_table" in m["artifact_sha256"]
    assert set(m["datasets"]) >= {"sentinel1", "nisar", "jrc_surface_water", "osm", "sen1floods11"}
    assert sv.tamper_check("evt", ".", out)["all_intact"]

    ok = sv.check_uploaded_file("evt", "input_gee_stack", b"pretend-geotiff", out)
    bad = sv.check_uploaded_file("evt", "input_gee_stack", b"pretend-geotiff!", out)
    assert ok["match"] and not bad["match"]
    with pytest.raises(KeyError):
        sv.check_uploaded_file("evt", "nope", b"x", out)
    fake_stack_file.write_bytes(b"edited")                           # tamper with an INPUT
    assert sv.tamper_check("evt", ".", out)["modified"] == ["input_gee_stack"]
