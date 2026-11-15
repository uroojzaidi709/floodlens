"""
download_asf.py - find and fetch NISAR L2 GCOV scenes from NASA ASF.

    # 1. what NISAR scenes cover Dadu? (pairs on the SAME track/direction/frame are listed)
    python pipeline/download_asf.py list --lon 67.7775 --lat 26.7288 --start 2026-01-01 --end 2026-11-14

    # 2. download one scene (large: several GB). Needs a token from urs.earthdata.nasa.gov -> Generate Token
    #    PowerShell:  $env:EARTHDATA_TOKEN = "..."     bash: export EARTHDATA_TOKEN=...
    python pipeline/download_asf.py get NISAR_L2_PR_GCOV_023_092_D_075_2005_QPDH_A_20260620T141344_..._001

You usually do NOT need `get`: nisar_extract.py streams just the AOI window from ASF.
Download the full file only if streaming keeps failing.

The download URL pattern below is the one nisar_extract.py already uses successfully for the
PROVISIONAL GCOV collection. If ASF moves the products to a validated collection, update URL_ROOT.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from forensics import Scene, parse_nisar_name

URL_ROOT = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GCOV_PROVISIONAL_V1"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "nisar_raw"


def pair_candidates(names: List[str]) -> Dict[tuple, List[Scene]]:
    """Group scene names by (track, direction, frame). Only groups with >= 2 dates are usable
    for before/after comparison."""
    groups: Dict[tuple, List[Scene]] = defaultdict(list)
    for n in names:
        try:
            s = parse_nisar_name(n)
        except ValueError:
            continue
        groups[(s.track, s.direction, s.frame)].append(s)
    return {k: sorted(v, key=lambda s: s.start_utc) for k, v in groups.items() if len(v) >= 2}


def search_nisar(lon: float, lat: float, start: str, end: str, max_results: int = 200) -> List[str]:
    """Scene names of NISAR GCOV products over a point. Needs `asf_search` (pip install asf_search)."""
    import asf_search as asf
    results = asf.search(dataset="NISAR", processingLevel="GCOV",
                         intersectsWith=f"POINT({lon} {lat})", start=start, end=end,
                         maxResults=max_results)
    names = []
    for r in results:
        p = r.properties
        n = p.get("sceneName") or p.get("fileID") or ""
        names.append(n.replace(".h5", ""))
    return sorted(set(n for n in names if n))


def download_scene(name: str, token: str, out_dir: Path = RAW_DIR, chunk_mb: int = 8) -> Path:
    """Stream the .h5 to disk. Resumes a partial file. Needs `requests`."""
    import requests
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.h5"
    url = f"{URL_ROOT}/{name}/{name}.h5"
    headers = {"Authorization": f"Bearer {token}"}

    have = dest.stat().st_size if dest.exists() else 0
    head = requests.head(url, headers=headers, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    if total and have == total:
        print(f"Already complete: {dest}")
        return dest
    if total and have > total:
        dest.unlink()
        have = 0
    if have:
        headers["Range"] = f"bytes={have}-"
    print(f"Downloading {name} ({total / 1e9:.2f} GB total, resuming at {have / 1e9:.2f} GB)")

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "ab" if have else "wb") as fh:
            for chunk in r.iter_content(chunk_mb << 20):
                fh.write(chunk)
    if total and dest.stat().st_size != total:
        raise IOError(f"Incomplete download {dest.stat().st_size}/{total}; run the command again to resume")
    print(f"Saved {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--lon", type=float, default=67.7775)
    ls.add_argument("--lat", type=float, default=26.7288)
    ls.add_argument("--start", default="2026-01-01")
    ls.add_argument("--end", default="2026-11-14")
    gt = sub.add_parser("get")
    gt.add_argument("scene")
    a = ap.parse_args()

    if a.cmd == "list":
        names = search_nisar(a.lon, a.lat, a.start, a.end)
        print(f"{len(names)} scenes found")
        for (track, direction, frame), scenes in pair_candidates(names).items():
            print(f"\ntrack {track} {direction} frame {frame}: {len(scenes)} dates")
            for s in scenes:
                print(f"   cycle {s.cycle}  {s.start_utc[:10]}  {s.scene_id}")
    else:
        token = os.environ.get("EARTHDATA_TOKEN")
        if not token:
            sys.exit("EARTHDATA_TOKEN is not set (urs.earthdata.nasa.gov -> Generate Token).")
        download_scene(a.scene, token)


if __name__ == "__main__":
    main()
