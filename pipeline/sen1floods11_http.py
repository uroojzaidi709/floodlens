
"""
download_sen1floods11_http.py - download the Sen1Floods11 hand-labelled chips WITHOUT gsutil.

Uses only `requests` and the public Google Cloud Storage JSON API (no login needed).

    python pipeline/download_sen1floods11_http.py --out data/sen1floods11

Downloads two folders (about 446 chips each, a few hundred MB in total):
    S1Hand/     radar chips (VV, VH in dB)
    LabelHand/  hand-drawn water labels
Re-running skips files that are already complete.
"""
import argparse
from pathlib import Path

import requests

BUCKET = "sen1floods11"
PREFIX = "v1.1/data/flood_events/HandLabeled"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}"


def list_objects(prefix):
    token = None
    while True:
        params = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(API, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        yield from data.get("items", [])
        token = data.get("nextPageToken")
        if not token:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sen1floods11")
    ap.add_argument("--folders", nargs="+", default=["S1Hand", "LabelHand"])
    a = ap.parse_args()
    out = Path(a.out)
    for folder in a.folders:
        dest_dir = out / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        items = [o for o in list_objects(f"{PREFIX}/{folder}/") if o["name"].endswith(".tif")]
        print(f"{folder}: {len(items)} files")
        for i, o in enumerate(items, 1):
            dest = dest_dir / Path(o["name"]).name
            if dest.exists() and dest.stat().st_size == int(o["size"]):
                continue
            with requests.get(f"{MEDIA}/{o['name']}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if i % 25 == 0 or i == len(items):
                print(f"  {folder}: {i}/{len(items)}")
    print("Done. Next: python pipeline/train_model.py --root data/sen1floods11 --holdout Pakistan --out ml/flood_lgbm.joblib")


if __name__ == "__main__":
    main()