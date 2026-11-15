"""
train_model.py - train the flood classifier on Sen1Floods11 (leave-one-event-out).

    # 1. get the hand-labelled chips (~ a few hundred MB; verify the path in the dataset README)
    gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand    data/sen1floods11/S1Hand
    gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand data/sen1floods11/LabelHand
    # 2. train, holding out Pakistan 2017
    python pipeline/train_model.py --root data/sen1floods11 --holdout Pakistan --out ml/flood_lgbm.joblib
"""
import argparse
import json

import ml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--holdout", default="Pakistan")
    ap.add_argument("--out", default="ml/flood_lgbm.joblib")
    ap.add_argument("--max-per-chip", type=int, default=4000)
    a = ap.parse_args()

    chips = ml.load_sen1floods11_chips(a.root)
    events = sorted({c["event"] for c in chips})
    print(f"{len(chips)} chips, events: {events}")
    if a.holdout not in events:
        raise SystemExit(f"holdout '{a.holdout}' not among events {events}")
    res = ml.train_leave_one_event_out(chips, a.holdout, a.max_per_chip)
    ml.save_model(res["model"], a.out, res["report"])
    print(json.dumps(res["report"], indent=2))


if __name__ == "__main__":
    main()
