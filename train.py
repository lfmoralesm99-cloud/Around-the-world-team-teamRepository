#!/usr/bin/env python3
"""Rebuild the frame classifier from the open sample games (no data of the challenge match).

    bash get_sample_data.sh
    python train.py
"""
import argparse
import os

import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", default=os.path.join(HERE, "external", "sample-data", "data"))
    ap.add_argument("--model", default=os.path.join(HERE, "possession_lgbm.txt"))
    args = ap.parse_args()
    pipeline.train(args.sample_dir, args.model)
    print(f"wrote {args.model}")


if __name__ == "__main__":
    main()
