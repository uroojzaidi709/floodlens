"""
explain.py - why did the model call a pixel flood? (SHAP)

Order of methods, best first:
    1. `shap` package TreeExplainer          -> exact TreeSHAP values
    2. LightGBM built-in pred_contrib=True   -> the same TreeSHAP values, no extra package
    3. permutation importance (scikit-learn) -> GLOBAL importance only, no per-pixel values
The method actually used is returned in the result so the slide never overclaims.

A SHAP value is in log-odds: positive pushes toward 'flood', negative toward 'not flood'.
It explains the MODEL, not the physics. A pixel can be explained and still be wrong.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ml import FEATURES


def _contrib_shap(model, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        import shap
    except ImportError:
        return None
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):            # older shap: one array per class
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                    # newer shap: (n, features, classes)
        sv = sv[..., -1]
    return sv


def _contrib_lgbm(model, X: np.ndarray) -> Optional[np.ndarray]:
    if type(model).__module__.split(".")[0] != "lightgbm":
        return None
    c = np.asarray(model.predict(X, pred_contrib=True))
    return c[:, :-1]                    # last column is the bias term


def explain_pixels(model, X: np.ndarray, y: Optional[np.ndarray] = None,
                   feature_names: Sequence[str] = FEATURES, max_rows: int = 5000,
                   seed: int = 0) -> Dict:
    """X: (n, n_features) feature rows. Returns method, mean |SHAP| per feature, raw values."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        sel = rng.choice(len(X), max_rows, replace=False)
        X = X[sel]
        y = y[sel] if y is not None else None

    for method, fn in (("shap_tree", _contrib_shap), ("lightgbm_pred_contrib", _contrib_lgbm)):
        sv = fn(model, X)
        if sv is not None:
            imp = np.abs(sv).mean(axis=0)
            return {"method": method, "per_pixel": True, "X": X, "values": sv,
                    "importance": dict(zip(feature_names, map(float, imp)))}

    if y is None:
        raise ValueError("Neither shap nor lightgbm is available; permutation importance needs labels y")
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=5, random_state=seed, scoring="f1")
    return {"method": "permutation_importance", "per_pixel": False, "X": X, "values": None,
            "importance": dict(zip(feature_names, map(float, r.importances_mean)))}


def explain_one(model, x_row: np.ndarray, feature_names: Sequence[str] = FEATURES) -> Dict:
    """Per-pixel explanation for the app's 'click a pixel' feature (needs method 1 or 2)."""
    res = explain_pixels(model, x_row.reshape(1, -1), max_rows=1)
    if not res["per_pixel"]:
        raise RuntimeError("Per-pixel explanation needs `shap` or `lightgbm` installed")
    vals = res["values"][0]
    return {"method": res["method"],
            "probability_flood": float(model.predict_proba(x_row.reshape(1, -1))[0, 1]),
            "contributions": sorted(({"feature": f, "value": float(x), "shap": float(s)}
                                     for f, x, s in zip(feature_names, x_row, vals)),
                                    key=lambda d: -abs(d["shap"]))}


def save_importance_plot(result: Dict, out_png: str | Path, title: str = "What drives the flood call") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["importance"])
    vals = np.array([result["importance"][n] for n in names])
    order = np.argsort(vals)
    xlabel = ("mean |SHAP| (log-odds)" if result["method"] != "permutation_importance"
              else "permutation importance (drop in F1)")
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=160)
    ax.barh(np.array(names)[order], vals[order], color="#1f6f8b")
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}\n(method: {result['method']})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
