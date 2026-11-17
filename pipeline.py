"""End-to-end steps: train on the open sample games, label the challenge match, write the file."""
import os

import lightgbm as lgb
import numpy as np

import dataset, decode, features, io_bundle, io_metrica, kickoff, match, model

LETTERS = np.array(["H", "A", "D", "X"])

# Decoder constants, selected by leave-one-game-out validation on the open sample games
# (TECHNICAL_DOCUMENT.md, section 1).  Every setting tried scored within 0.012 spellF1 of the
# best, far inside the noise of three games; among those within 0.005 of the best this is the one
# whose number of possessions is closest to the truth.  Not tuned on the challenge match.
DECODER = dict(c_switch=4.0, c_dead_in=6.0, c_dead_out=6.0, min_spell=8, min_dead=12)
TRAIN_SEEDS = (1, 2, 3)


def train(sample_dir, model_path, verbose=True):
    """Fit the frame classifier on degraded copies of the three open sample games."""
    games = io_metrica.load_all(sample_dir)
    X, y, names = dataset.training_rows(games, seeds=TRAIN_SEEDS)
    if verbose:
        print(f"training rows: {X.shape[0]} x {X.shape[1]} features")
    booster = model.fit(X, y, names)
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    booster.save_model(model_path)
    return booster


def frame_probabilities(T, booster):
    """P(H), P(A), P(D) for every graded frame; {half: (first_frame, probabilities)}."""
    out = {}
    for half, first, last in match.half_blocks():
        sl = slice(first, last + 1)
        x0, x1, names = features.period_features(T["xy"][sl], T["team"], T["ball"][sl])
        assert names == booster.feature_name(), "feature set does not match the model"
        out[half] = (first, model.predict(booster, x0, x1))
    return out


def label_match(data_dir, model_path, cache_dir=None, verbose=True):
    """One label per frame of the tactical video: 0 H, 1 A, 2 D, 3 X."""
    T = match.load(data_dir, cache_dir, verbose)
    booster = lgb.Booster(model_file=model_path)
    tag_file = match.paths(data_dir, "tactical")["smart_tagging"]
    tags = io_bundle.read_smart_tagging(tag_file) if os.path.exists(tag_file) else None
    labels = np.full(match.N_VIDEO_FRAMES, 3, dtype=np.int64)     # X outside the two halves
    probs = frame_probabilities(T, booster)
    for half, first, last in match.half_blocks():
        P = probs[half][1]
        lab = decode.decode(P, **DECODER)
        lab, changed = kickoff.gate(lab, T["xy"][first:last + 1], T["team"],
                                    kickoff.opening_tag(tags, first, last), first)
        if verbose:
            print(f"half {half}: frames {first}..{last}, {changed} frames set to dead ball by the "
                  f"opening kick-off rules")
        labels[first:last + 1] = lab
    return labels, probs


def write_submission(labels, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.write("\n".join(f"{i},{c}" for i, c in enumerate(LETTERS[labels])))
        f.write("\n")
