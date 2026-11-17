#!/usr/bin/env python3
"""Validate a submission file and, optionally, score it against a frame,label ground truth.

    python check_submission.py possession.csv
    python check_submission.py possession.csv --gt path/to/public_ground_truth.csv

Scoring uses this repository's re-implementation of the metrics (metrics.py); the
organisers' evaluation/run_evaluation.py is the authority.  The public 5-minute ground truth may
only be used this way -- to check that the pipeline is correct -- never to tune or train.
"""
import argparse

import numpy as np

import check, match, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("--gt", default=None)
    args = ap.parse_args()
    labels = check.validate(args.submission)
    print("format OK:", match.N_VIDEO_FRAMES, "rows, frames 0..N-1, labels in H/A/D/X")
    check.describe(labels)
    if args.gt:
        gt, frames = check.read_labels(args.gt, match.N_VIDEO_FRAMES)
        th, ph = [], []
        for _, a, b in match.half_blocks():
            m = np.zeros(match.N_VIDEO_FRAMES, bool)
            m[frames] = True
            m &= gt != 3
            m[:a] = False
            m[b + 1:] = False
            if m.any():
                th.append(gt[m])
                ph.append(labels[m])
        print("score on the provided ground truth:", metrics.score(th, ph))


if __name__ == "__main__":
    main()
