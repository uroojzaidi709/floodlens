"""
ml.py - pixel classifier for flood water, trained on Sen1Floods11 and applied to Sindh.

DESIGN (read this before the viva)
    * Sen1Floods11 has NO pre-flood image, so the model only uses post-flood radar features
      (VV, VH, VV-VH, local texture). That is what makes it transferable to Sindh.
    * Honest evaluation = leave-one-event-out: train on 10 flood events, test on the 11th
      (default: Pakistan 2017). The test event is never seen in training.
    * For Sindh 2022 there is NO ground truth. We can compare ML vs threshold maps
      (agreement), but we cannot report an accuracy for Sindh. Never claim one.
    * LightGBM is used if installed, otherwise scikit-learn's HistGradientBoosting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

FEATURES = ["vv_db", "vh_db", "vv_minus_vh", "vh_local_std", "vv_local_std"]
CHIP_FILL = -1            # Sen1Floods11 label for 'no data'
BASELINE_VH_DB = -18.0


# --------------------------------------------------------------------------- features
def _local_std(a: np.ndarray, size: int = 5) -> np.ndarray:
    a0 = np.nan_to_num(a, nan=np.nanmean(a) if np.isfinite(a).any() else 0.0)
    m = uniform_filter(a0, size)
    return np.sqrt(np.maximum(uniform_filter(a0 * a0, size) - m * m, 0.0))


def build_features(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """(H, W) dB bands -> (H, W, 5) float32 feature cube."""
    if vv_db.shape != vh_db.shape:
        raise ValueError("VV and VH must have the same shape")
    cube = np.stack([vv_db, vh_db, vv_db - vh_db, _local_std(vh_db), _local_std(vv_db)], axis=-1)
    return cube.astype("float32")


# --------------------------------------------------------------------------- model
def make_model(seed: int = 42):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                                  class_weight="balanced", random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                              class_weight="balanced", random_state=seed)


def backend_name(model) -> str:
    return type(model).__module__.split(".")[0]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    t, p = y_true.astype(bool), y_pred.astype(bool)
    tp, fp, fn, tn = (int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()), int((~t & ~p).sum()))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "iou": round(iou, 4), "accuracy": round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
            "n_pixels": tp + fp + fn + tn, "water_fraction_true": round(float(t.mean()), 4)}


# --------------------------------------------------------------------------- data
def chips_to_xy(chips: Sequence[Dict], max_per_chip: Optional[int] = None,
                seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    chips: dicts with 'event', 'vv', 'vh' (dB) and 'label' (1 water, 0 land, -1 nodata).
    Returns X (n, 5), y (n,), event (n,). Nodata labels and NaN features are dropped.
    """
    rng = np.random.default_rng(seed)
    Xs, ys, es = [], [], []
    for c in chips:
        f = build_features(c["vv"], c["vh"]).reshape(-1, len(FEATURES))
        y = c["label"].reshape(-1)
        ok = (y != CHIP_FILL) & np.isfinite(f).all(axis=1)
        idx = np.flatnonzero(ok)
        if max_per_chip and idx.size > max_per_chip:
            idx = rng.choice(idx, max_per_chip, replace=False)
        Xs.append(f[idx]); ys.append(y[idx].astype("uint8")); es.append(np.full(idx.size, c["event"]))
    if not Xs:
        raise ValueError("no chips given")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(es)


def load_sen1floods11_chips(root: str | Path, events: Optional[Sequence[str]] = None) -> List[Dict]:
    """
    Read hand-labelled chips. Expected layout (from gs://sen1floods11/v1.1/data/flood_events/HandLabeled):
        <root>/S1Hand/<Event>_<id>_S1Hand.tif      bands: VV, VH (dB)
        <root>/LabelHand/<Event>_<id>_LabelHand.tif  1 water, 0 land, -1 no data
    Requires rasterio (imported lazily so the rest of this module works without it).
    """
    import rasterio
    root = Path(root)
    chips = []
    for s1 in sorted((root / "S1Hand").glob("*_S1Hand.tif")):
        event = s1.name.split("_")[0]
        if events and event not in events:
            continue
        lab = root / "LabelHand" / s1.name.replace("_S1Hand", "_LabelHand")
        if not lab.exists():
            continue
        with rasterio.open(s1) as a, rasterio.open(lab) as b:
            vv, vh = a.read(1).astype("float32"), a.read(2).astype("float32")
            label = b.read(1).astype("int16")
        chips.append({"event": event, "name": s1.stem, "vv": vv, "vh": vh, "label": label})
    if not chips:
        raise FileNotFoundError(f"No Sen1Floods11 chips found under {root}")
    return chips


# --------------------------------------------------------------------------- train / eval
def train_leave_one_event_out(chips: Sequence[Dict], holdout_event: str = "Pakistan",
                              max_per_chip: int = 4000, seed: int = 42) -> Dict:
    """Train on every event except `holdout_event`; evaluate on all its pixels."""
    train = [c for c in chips if c["event"] != holdout_event]
    test = [c for c in chips if c["event"] == holdout_event]
    if not train or not test:
        raise ValueError(f"need chips for training and for holdout '{holdout_event}'")
    Xtr, ytr, _ = chips_to_xy(train, max_per_chip, seed)
    Xte, yte, _ = chips_to_xy(test, None, seed)

    model = make_model(seed).fit(Xtr, ytr)
    pred = model.predict(Xte)
    base = Xte[:, FEATURES.index("vh_db")] < BASELINE_VH_DB

    report = {
        "backend": backend_name(model),
        "holdout_event": holdout_event,
        "train_events": sorted({c["event"] for c in train}),
        "n_train_pixels": int(len(ytr)),
        "ml": metrics(yte, pred),
        "baseline_threshold_vh_db": BASELINE_VH_DB,
        "baseline": metrics(yte, base),
        "caveat": ("Accuracy is measured on the held-out event only. Sindh 2022 has no "
                   "ground truth, so no accuracy is claimed for it."),
    }
    return {"model": model, "report": report}


def save_model(model, path: str | Path, report: Optional[Dict] = None) -> None:
    import joblib
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    if report is not None:
        p.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_model(path: str | Path):
    import joblib
    return joblib.load(path)


# --------------------------------------------------------------------------- inference
def predict_probability(model, vv_db: np.ndarray, vh_db: np.ndarray, batch: int = 1_000_000) -> np.ndarray:
    """(H, W) probability of water; NaN where input is NaN."""
    cube = build_features(vv_db, vh_db)
    flat = cube.reshape(-1, len(FEATURES))
    ok = np.isfinite(flat).all(axis=1)
    out = np.full(flat.shape[0], np.nan, dtype="float32")
    idx = np.flatnonzero(ok)
    for i in range(0, idx.size, batch):
        j = idx[i:i + batch]
        out[j] = model.predict_proba(flat[j])[:, 1]
    return out.reshape(vv_db.shape)
