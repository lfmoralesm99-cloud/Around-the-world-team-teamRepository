#!/usr/bin/env python3
"""Label-free checks of a submission against the provided automatic data.

None of this uses a ground truth.  It answers "does the output behave like football?":

  * how the labels relate to the presence of the ball in the automatic tracking,
  * how fast the players move in each state (they walk when the ball is dead),
  * whether the team in possession agrees with the automatic Smart Tagging, which names the team
    that is building up / progressing in a number of time windows.

    python diagnostics.py possession.csv
"""
import argparse
import os
import warnings

import numpy as np

import check, io_bundle, match

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(HERE, "data"))
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    lab = check.validate(args.submission)
    check.describe(lab)
    T = match.load(args.data_dir, args.cache_dir, verbose=False)
    graded = lab < 3
    has = T["ball_src"] > 0
    print("\nshare of graded frames by (ball tracked?, label):")
    for k, nm in ((1, "ball tracked "), (0, "ball missing ")):
        row = [np.mean((has[graded] == k) & (lab[graded] == c)) for c in range(3)]
        print(f"  {nm} H {row[0]:.3f}  A {row[1]:.3f}  D {row[2]:.3f}")
    xy = T["xy"]
    v = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    v[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=2) * 6.25
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spd = np.nanmean(v, 1)
    print(f"\nmean player speed: ball in play {np.nanmean(spd[lab < 2]):.2f} m/s, dead ball {np.nanmean(spd[lab == 2]):.2f} m/s")

    tags = io_bundle.read_smart_tagging(match.paths(args.data_dir, "tactical")["smart_tagging"])
    home, away = T["home"]["name"], T["away"]["name"]
    print("\nagreement with Smart Tagging (share of in-play frames of each window given to the team the tag implies):")
    for code, flip in (("BUILD UP", False), ("PROGRESSION", False), ("ATTACKING TRANSITION", False),
                       ("BALL IN FINAL THIRD", False), ("CREATING CHANCES", False)):
        agree = []
        for r in tags[tags["code"] == code].itertuples():
            if r.Team not in (home, away):
                continue
            t = 0 if r.Team == home else 1
            seg = lab[int(r.start * 25):int(r.end * 25)]
            seg = seg[seg < 2]
            if len(seg):
                agree.append(np.mean(seg == (1 - t if flip else t)))
        if agree:
            print(f"  {code:22s} windows {len(agree):3d}  mean {np.mean(agree):.3f}  median {np.median(agree):.3f}")


if __name__ == "__main__":
    main()
