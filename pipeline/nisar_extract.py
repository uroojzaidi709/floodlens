"""
nisar_extract.py  -  Extracts a small area of interest (AOI) from two NISAR L2 GCOV
scenes (before/after), masks the persistent low-signal strip (sub-swath gap), and
writes change-detection outputs.

Examples (run from the project root, e.g. D:\\floodlens):
    # default: Dadu, 20 Jun vs 31 Aug 2026
    python pipeline/nisar_extract.py

    # another place / other scenes
    python pipeline/nisar_extract.py --tag badin --lon 68.8376 --lat 24.6603 --half-km 20 \\
        --before NISAR_L2_PR_GCOV_xxx... --after NISAR_L2_PR_GCOV_yyy...

Token (PowerShell, current terminal only):
    $env:EARTHDATA_TOKEN = "your_token_here"

Data access:
  A) If data/nisar_raw/<scene>.h5 exists locally -> read from disk (no internet needed)
  B) Otherwise the file is streamed from ASF (token required, the full file is NOT downloaded)

Output: data/nisar_aoi/  ({tag}_*.tif, {tag}_change.png, nisar_{tag}_summary.json)
"""
import os, sys, json, time, hashlib, argparse
from pathlib import Path
import numpy as np
import h5py
from pyproj import Transformer
from scipy.ndimage import uniform_filter, binary_dilation
import rasterio
from rasterio.transform import from_origin

# ---------------- settings ----------------
LOOKS    = 5                     # 5x5 averaging in the linear domain to reduce speckle
STREAM_BLOCK_MB = 2              # size of each HTTP range request when streaming
RETRIES  = 6                     # retries if the connection drops
POLS     = ("HHHH", "HVHV")
BASE     = "/science/LSAR/GCOV/grids/frequencyA/"
# Persistent low-signal strip (sub-swath gap / noise floor): dark in BOTH dates and BOTH pols.
# Thresholds checked on the Dadu scenes: they cover the strip and none of the river water.
GAP_HH_DB, GAP_HV_DB, GAP_DILATE_PX = -26.0, -35.0, 10
ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "nisar_raw"
OUT_DIR  = ROOT / "data" / "nisar_aoi"
URL_ROOT = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GCOV_PROVISIONAL_V1"

DEFAULTS = dict(
    tag="dadu", lon=67.7775, lat=26.7288, half_km=15.0,
    before="NISAR_L2_PR_GCOV_023_092_D_075_2005_QPDH_A_20260620T141344_20260620T141418_P05023_N_F_J_001",
    after="NISAR_L2_PR_GCOV_029_092_D_075_2005_QPDH_A_20260831T141340_20260831T141415_P05023_N_F_J_001",
)



def open_h5(name):
    """Open the local file if it exists, otherwise stream it from ASF."""
    local = RAW_DIR / f"{name}.h5"
    if local.exists():
        print(f"  [{name[-40:]}] reading from local file")
        return h5py.File(local, "r"), None
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        sys.exit("EARTHDATA_TOKEN is not set (urs.earthdata.nasa.gov -> Generate Token).")
    import fsspec
    url = f"{URL_ROOT}/{name}/{name}.h5"
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=300)
    fs = fsspec.filesystem("https", client_kwargs={"headers": {"Authorization": f"Bearer {token}"},
                                                   "timeout": timeout})
    print(f"  [{name[-40:]}] streaming from ASF (the full file will not be downloaded)")
    raw = fs.open(url, "rb", block_size=STREAM_BLOCK_MB * 1024 * 1024)
    return h5py.File(raw, "r"), raw


def read_epsg(f):
    ds = f[BASE + "projection"]
    for key in ("epsg_code", "epsg"):
        if key in ds.attrs:
            return int(np.array(ds.attrs[key]).ravel()[0])
    return int(ds[()])


def _retryable():
    """Exception types that mean 'network problem' (worth retrying)."""
    import asyncio
    types = [OSError, RuntimeError, asyncio.TimeoutError]
    try:
        from fsspec.exceptions import FSTimeoutError
        types.append(FSTimeoutError)
    except ImportError:
        pass
    try:
        import aiohttp
        types.append(aiohttp.ClientError)
    except ImportError:
        pass
    return tuple(types)


def read_aoi(name, lon, lat, half_km):
    retryable = _retryable()
    for attempt in range(1, RETRIES + 1):
        f = raw = None
        try:
            f, raw = open_h5(name)      # opening is inside the retry block (the first read can time out too)
            g = f[BASE]
            epsg = read_epsg(f)
            x, y = g["xCoordinates"][:], g["yCoordinates"][:]
            cx, cy = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
            h = half_km * 1000
            xi = np.where((x > cx - h) & (x < cx + h))[0]
            yi = np.where((y > cy - h) & (y < cy + h))[0]
            if len(xi) == 0 or len(yi) == 0:
                sys.exit("The AOI is outside the scene footprint. Check lon/lat and the scene name.")
            xs, ys = slice(xi.min(), xi.max() + 1), slice(yi.min(), yi.max() + 1)
            data = {p: g[p][ys, xs].astype("float32") for p in POLS}   
            dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
            return data, x[xs], y[ys], epsg, dx, dy
        except retryable as e:
            print(f"  connection problem (attempt {attempt}/{RETRIES}): {type(e).__name__}")
            if attempt == RETRIES:
                print("\nStreaming keeps failing. Fallback: download the full .h5 files into "
                      f"{RAW_DIR} and run this script again (it will read them locally).")
                raise
            time.sleep(5 * attempt)
        finally:
            for obj in (f, raw):
                try:
                    if obj is not None:
                        obj.close()
                except Exception:
                    pass


def valid_mask(a):
    return np.isfinite(a) & (a > 0)


def to_db(a):
    a = np.where(valid_mask(a), a, np.nan)
  
    m = np.isfinite(a).astype("float32")
    s = uniform_filter(np.nan_to_num(a), LOOKS) / np.maximum(uniform_filter(m, LOOKS), 1e-6)
    s = np.where(uniform_filter(m, LOOKS) > 0.5, s, np.nan)
    return 10 * np.log10(s)


def write_tif(path, arr, x, y, dx, dy, epsg, dtype="float32", nodata=np.nan):
    transform = from_origin(x[0] - dx / 2, y[0] - dy / 2 if dy < 0 else y[0] + dy / 2, abs(dx), abs(dy))
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs=f"EPSG:{epsg}", transform=transform,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(arr.astype(dtype), 1)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def change_stats(chg):
    ok = np.isfinite(chg)
    if not ok.any():
        return {"mean_change_db": None, "pct_brighter_than_plus3db": None, "pct_darker_than_minus3db": None}
    c = chg[ok]
    return {"mean_change_db": round(float(c.mean()), 3),
            "pct_brighter_than_plus3db": round(float((c > 3).mean() * 100), 2),
            "pct_darker_than_minus3db": round(float((c < -3).mean() * 100), 2)}


def main():
    ap = argparse.ArgumentParser(description="Extract a NISAR GCOV AOI (before/after) and compute change maps.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    args = ap.parse_args()
    tag, lon, lat, half_km = args.tag, args.lon, args.lat, args.half_km
    scenes = {"before": args.before, "after": args.after}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    got = {}
    for label, name in scenes.items():
        print(f"{label}: {name[-60:]}")
        got[label] = read_aoi(name, lon, lat, half_km)

    (b, xb, yb, epsg, dx, dy), (a, xa, ya, epsg2, _, _) = got["before"], got["after"]
    assert epsg == epsg2, f"EPSG mismatch: {epsg} vs {epsg2}"
    ny, nx = min(b["HHHH"].shape[0], a["HHHH"].shape[0]), min(b["HHHH"].shape[1], a["HHHH"].shape[1])
    shift = max(abs(xb[0] - xa[0]), abs(yb[0] - ya[0]))
    if shift > abs(dx) / 2:
        print(f"WARNING: the two grids have origins {shift:.1f} m apart - the overlay may be shifted")
    xb, yb = xb[:nx], yb[:ny]

    db = {pol: (to_db(b[pol][:ny, :nx]), to_db(a[pol][:ny, :nx])) for pol in POLS}


    low = np.ones((ny, nx), bool)
    for pol, thr in (("HHHH", GAP_HH_DB), ("HVHV", GAP_HV_DB)):
        low &= (db[pol][0] < thr) & (db[pol][1] < thr)
    low = binary_dilation(low, iterations=GAP_DILATE_PX) if low.any() else low
    low_pct = float(low.mean() * 100)
    print(f"Low-signal strip masked out of the change maps: {low_pct:.1f}% of the AOI")
    if low_pct > 30:
        print("WARNING: more than 30% of the AOI is low-signal - consider moving the AOI.")

    summary = {"tag": tag, "aoi_center_lonlat": [lon, lat], "half_km": half_km, "epsg": epsg,
               "pixel_size_m": [abs(dx), abs(dy)], "shape": [ny, nx], "multilook": LOOKS,
               "low_signal_mask": {"hh_db_below": GAP_HH_DB, "hv_db_below": GAP_HV_DB,
                                   "dilate_px": GAP_DILATE_PX, "pct_of_aoi": round(low_pct, 2)},
               "scenes": scenes, "polarizations": {}, "files": {}}

    p = OUT_DIR / f"{tag}_low_signal_mask.tif"
    write_tif(p, low.astype("uint8"), xb, yb, dx, dy, epsg, dtype="uint8", nodata=255)
    summary["files"][p.name] = sha256(p)

    changes = {}
    for pol in POLS:
        db_b, db_a = db[pol]
        chg = np.where(low, np.nan, db_a - db_b)
        changes[pol] = chg
        summary["polarizations"][pol] = {
            "valid_fraction_before": round(float(np.isfinite(db_b).mean()), 4),
            "valid_fraction_after": round(float(np.isfinite(db_a).mean()), 4),
            **change_stats(chg),
        }
        for lab, arr in (("before", db_b), ("after", db_a), ("change", chg)):
            p = OUT_DIR / f"{tag}_{pol}_{lab}_db.tif"
            write_tif(p, arr, xb, yb, dx, dy, epsg)
            summary["files"][p.name] = sha256(p)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for i, pol in enumerate(POLS):
        db_b, db_a = db[pol]
        ax[i, 0].imshow(db_b, cmap="gray", vmin=-25, vmax=0); ax[i, 0].set_title(f"{pol} before")
        ax[i, 1].imshow(db_a, cmap="gray", vmin=-25, vmax=0); ax[i, 1].set_title(f"{pol} after")
        im = ax[i, 2].imshow(changes[pol], cmap="RdBu", vmin=-8, vmax=8); ax[i, 2].set_title(f"{pol} change (dB)")
        for k in range(3): ax[i, k].axis("off")
    fig.colorbar(im, ax=ax[:, 2], shrink=0.7)
    fig.savefig(OUT_DIR / f"{tag}_change.png", dpi=150)

    from preprocess import nisar_rgb
    for lab, idx in (("before", 0), ("after", 1)):
        p = OUT_DIR / f"{tag}_rgb_{lab}.png"
        plt.imsave(p, nisar_rgb(db["HHHH"][idx], db["HVHV"][idx], low_signal=low))
        summary["files"][p.name] = sha256(p)
    summary["rgb_recipe"] = "R=HH, G=HV, B=HH/HV ratio (dB difference); 2-98 percentile stretch"

    (OUT_DIR / f"nisar_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDone. Files written to:", OUT_DIR)
    print(json.dumps({"low_signal_mask": summary["low_signal_mask"], **summary["polarizations"]}, indent=2))


if __name__ == "__main__":
    main()