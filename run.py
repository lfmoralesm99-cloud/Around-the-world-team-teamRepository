#!/usr/bin/env python3
"""Regenerate the submission from the provided data.

    python run.py                       # data in ./data (or $MTAH_DATA_DIR), writes possession.csv
    python run.py --data-dir /path/to/bundle --out my.csv

Uses the model committed in possession_lgbm.txt.  To rebuild that model from the open
sample games first, run `python train.py` (see README.md).
"""
import argparse
import hashlib
import os

import pipeline
from check import describe

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(HERE, "data"))
    ap.add_argument("--model", default=os.path.join(HERE, "possession_lgbm.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "possession.csv"))
    ap.add_argument("--cache-dir", default=None,
                    help="optional folder for parsed tracking (speeds up re-runs; keep it out of git)")
    args = ap.parse_args()

    labels, _ = pipeline.label_match(args.data_dir, args.model, args.cache_dir)
    pipeline.write_submission(labels, args.out)
    with open(args.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"wrote {args.out}")
    print(f"sha256 {digest}")
    describe(labels)


if __name__ == "__main__":
    main()
