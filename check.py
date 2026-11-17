"""Format checks and summary counts of a submission (labels as 0 H, 1 A, 2 D, 3 X)."""
import csv

import numpy as np

import match, metrics

CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
FPS = 25.0


def read_labels(path, n=None):
    """Read a frame,label CSV into an int array indexed by video frame (missing frames = X)."""
    frames, labs = [], []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        assert [h.strip().lower() for h in header[:2]] == ["frame", "label"], header
        for row in rd:
            if not row:
                continue
            frames.append(int(row[0]))
            labs.append(CODE[row[1].strip()])
    frames = np.array(frames)
    n = n or (frames.max() + 1)
    out = np.full(n, 3, dtype=np.int64)
    out[frames] = labs
    return out, frames


def validate(path):
    labels, frames = read_labels(path, match.N_VIDEO_FRAMES)
    assert len(frames) == match.N_VIDEO_FRAMES, f"{len(frames)} rows, expected {match.N_VIDEO_FRAMES}"
    assert (frames == np.arange(match.N_VIDEO_FRAMES)).all(), "frames must be 0..N-1 in order"
    return labels


def describe(labels):
    """Print the counts a reader wants to see: label shares and possessions per half."""
    graded = np.zeros(len(labels), bool)
    for _, a, b in match.half_blocks():
        graded[a:b + 1] = True
    g = labels[graded]
    share = np.bincount(g, minlength=4) / len(g)
    print(f"graded frames {graded.sum()}: H {share[0]:.3f}  A {share[1]:.3f}  D {share[2]:.3f}  X {share[3]:.3f}")
    outside = np.bincount(labels[~graded], minlength=4)
    print(f"frames outside the two halves: {int(outside.sum())}, of which X: {int(outside[3])}")
    for h, a, b in match.half_blocks():
        sp = metrics.spells(labels[a:b + 1])
        if len(sp):
            d = (sp[:, 1] - sp[:, 0]) / FPS
            print(f"half {h}: {len(sp)} possessions (H {(sp[:, 2] == 0).sum()}, A {(sp[:, 2] == 1).sum()}), "
                  f"median {np.median(d):.1f} s, shortest {d.min():.2f} s, longest {d.max():.1f} s")
