"""Feature tables for the labelled games (other matches) and for the challenge match."""
import numpy as np

import features as F
from simulate import degrade

SWAP = np.array([1, 0, 2])          # label seen from the other team's point of view


def game_table(game):
    """Features of a (possibly degraded) labelled game, both views, period by period.
    Returns X0, X1 (home / away view), y (home view), period id per frame, names."""
    X0, X1, Y, PER = [], [], [], []
    for p in np.unique(game["period"]):
        m = game["period"] == p
        x0, x1, names = F.period_features(game["xy"][m], game["team"], game["ball"][m])
        X0.append(x0); X1.append(x1); Y.append(game["label"][m]); PER.append(game["period"][m])
    return np.concatenate(X0), np.concatenate(X1), np.concatenate(Y).astype(np.int64), np.concatenate(PER), names


def training_rows(games, seeds, stride=5, strengths=(0.6, 1.0, 1.4), sigmas=(0.35, 0.5, 0.65),
                  smooths=(5.0, 6.0, 7.0)):
    """Stack degraded copies of the labelled games; every frame appears once per view."""
    X, Y = [], []
    names = None
    for gi, g in enumerate(games):
        for k, seed in enumerate(seeds):
            d = degrade(g, seed=1000 * (gi + 1) + seed, strength=strengths[k % len(strengths)],
                        ball_sigma=sigmas[(k + gi) % len(sigmas)],
                        ball_smooth=smooths[(k + 2 * gi) % len(smooths)])
            x0, x1, y, _, names = game_table(d)
            off = (k * 2 + gi) % stride
            X.append(x0[off::stride]); Y.append(y[off::stride])
            X.append(x1[off::stride]); Y.append(SWAP[y[off::stride]])
    return np.concatenate(X), np.concatenate(Y), names
