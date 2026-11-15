"""
download_sen1floods11.py - fetch the hand-labelled Sen1Floods11 chips used to train the ML model.

Needs the Google Cloud SDK (`gsutil`). The bucket is public. Hand-labelled set only
(~446 chips, 11 flood events incl. Pakistan) - NOT the 14 GB weakly-labelled set.

    python pipeline/download_sen1floods11.py --out data/sen1floods11

If the bucket layout has changed, check the dataset README (github.com/cloudtostreet/Sen1Floods11).
"""
import argparse
import shutil
import subprocess
import sys

BASE = "gs://sen1floods11/v1.1/data/flood_events/HandLabeled"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sen1floods11")
    a = ap.parse_args()
    if shutil.which("gsutil") is None:
        sys.exit("gsutil not found. Install the Google Cloud SDK, or download the two folders by hand:\n"
                 f"  {BASE}/S1Hand\n  {BASE}/LabelHand")
    for sub in ("S1Hand", "LabelHand"):
        cmd = ["gsutil", "-m", "rsync", "-r", f"{BASE}/{sub}", f"{a.out}/{sub}"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    print(f"Done. Next: python pipeline/train_model.py --root {a.out} --holdout Pakistan")


if __name__ == "__main__":
    main()
