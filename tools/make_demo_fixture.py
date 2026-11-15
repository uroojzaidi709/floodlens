"""
make_demo_fixture.py - writes outputs/demo_synthetic/ so the frontend can be built TODAY, before the
real GEE stack arrives. Same file layout and keys as a real run. The numbers are FAKE.

    python tools/make_demo_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import run_pipeline as rp  # noqa: E402

N = 300
rng = np.random.default_rng(7)
pre = rng.normal(-15, 1.5, (N, N)).astype("float32")
post = pre + rng.normal(0, 0.8, (N, N)).astype("float32")
slope = np.zeros((N, N), "float32"); jrc = np.zeros((N, N), "float32")
post[40:150, 30:200] = rng.normal(-24, 1, (110, 170))              # fake flood
post[200:215, :] = pre[200:215, :] = rng.normal(-25, 1, (15, N)); jrc[200:215, :] = 100   # fake river
stack = rp.Stack({"vv_pre": pre + 7, "vh_pre": pre, "vv_post": post + 7, "vh_post": post,
                  "slope_deg": slope, "jrc_occurrence": jrc,
                  "pop_count": np.full((N, N), 800.0, "float32"),
                  "pop_cell_m2": np.full((N, N), 9000.0, "float32"),
                  "worldcover": np.where(rng.random((N, N)) > .3, 40, 50).astype("float32")},
                 30.0, 400000.0, 3000000.0, "EPSG:32642")

scenes = ROOT / "outputs" / "demo_synthetic_scenes.csv"
scenes.parent.mkdir(exist_ok=True)
scenes.write_text("scene_id,relativeOrbitNumber_start,orbitProperties_pass,time_start_ms,role,transmitterReceiverPolarisation\n"
                  "DEMO_PRE,78,ASCENDING,1652400000000,pre,VV+VH\nDEMO_POST,78,ASCENDING,1661750000000,post,VV+VH\n")
bounds = (67.62, 26.59, 67.93, 26.86)                              # fake placement near Dadu
s = rp.run(stack, scenes, ROOT / "outputs", "demo_synthetic", warp=lambda a, g, r="nearest": (a.astype("float32"), bounds))
p = ROOT / "outputs" / "demo_synthetic" / "summary.json"
s["WARNING"] = "SYNTHETIC DEMO DATA - numbers are fake. For frontend development only."
p.write_text(json.dumps(s, indent=2), encoding="utf-8")
print("wrote", p.parent)
