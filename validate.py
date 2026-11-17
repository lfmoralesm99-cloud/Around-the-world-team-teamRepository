#!/usr/bin/env python3
"""Leave-one-game-out validation on the open sample games -- the numbers in the technical document.

For each of the three games: train the frame classifier on degraded copies of the other two,
degrade the held-out game with an unseen seed, predict, decode, score.  No data of the challenge
match is involved.

    bash get_sample_data.sh
    python validate.py                 # about 15 minutes on 2 cores
"""
import argparse
import os

import numpy as np

import dataset, decode, io_metrica, metrics, model, pipeline
from simulate import degrade

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", default=os.path.join(HERE, "external", "sample-data", "data"))
    args = ap.parse_args()
    games = io_metrica.load_all(args.sample_dir)
    rows = []
    for hold, g in enumerate(games):
        train = [x for i, x in enumerate(games) if i != hold]
        X, y, names = dataset.training_rows(train, seeds=pipeline.TRAIN_SEEDS)
        booster = model.fit(X, y, names)
        del X
        test = degrade(g, seed=99)
        x0, x1, yt, per, _ = dataset.game_table(test)
        P = model.predict(booster, x0, x1)
        true = [yt[per == p] for p in np.unique(per)]
        raw = metrics.score(true, [P[per == p].argmax(1) for p in np.unique(per)])
        dec = metrics.score(true, [decode.decode(P[per == p], **pipeline.DECODER) for p in np.unique(per)])
        rows.append((g["name"], raw, dec))
        print(f"{g['name']}: frame argmax spellF1 {raw['spellF1']:.4f} macroF1 {raw['macroF1']:.4f} "
              f"({raw['n_pred']} possessions) | decoded spellF1 {dec['spellF1']:.4f} macroF1 "
              f"{dec['macroF1']:.4f} ({dec['n_pred']} possessions, truth {dec['n_true']})", flush=True)
    print(f"mean decoded spellF1 {np.mean([r[2]['spellF1'] for r in rows]):.4f}  "
          f"macroF1 {np.mean([r[2]['macroF1'] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
