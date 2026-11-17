"""The challenge match: where the files are, which video frames are graded.

The frame arithmetic is the organisers' (games.py in the challenge repository): the submission is
indexed by frame of the tactical video, and the graded frames are the two blocks of the tracking
counter ("IdFrame") mapped back to video frames with one offset per half.  The numbers below are
copied from that file; nothing here was read off the video or the ground truth.
"""
import os

import numpy as np

import align
from io_bundle import read_epts

GAME = "BAR-ATH"
N_VIDEO_FRAMES = 199319
OFFSETS = {1: 1702057, 2: 1702058}                       # id_frame = video_frame + offset
HALVES_ID = {1: (1713242, 1792708), 2: (1812866, 1892255)}


def half_blocks():
    """[(half, first_video_frame, last_video_frame)] -- 11185..90651 and 110808..190197."""
    return [(h, HALVES_ID[h][0] - OFFSETS[h], HALVES_ID[h][1] - OFFSETS[h]) for h in sorted(HALVES_ID)]


def paths(data_dir, source):
    stem = os.path.join(data_dir, source, f"{GAME}_{source}")
    return {"txt": stem + "_atd.txt", "xml": stem + "_atd.xml",
            "homography": stem + "_homography.csv", "smart_tagging": stem + "_smart_tagging.xml"}


def load(data_dir, cache_dir=None, verbose=True):
    """Tactical tracking with the fused ball track. Row i = frame i of the tactical video."""
    def cache(name):
        return os.path.join(cache_dir, name) if cache_dir else None
    pt, pb = paths(data_dir, "tactical"), paths(data_dir, "broadcast")
    T = read_epts(pt["txt"], pt["xml"], cache=cache("tactical.npz"))
    assert len(T["ball"]) == N_VIDEO_FRAMES, len(T["ball"])
    fused = cache("fused.npz")
    if fused and os.path.exists(fused):
        z = np.load(fused)
        ball, src = z["ball"], z["src"]
    elif os.path.exists(pb["txt"]):
        B = read_epts(pb["txt"], pb["xml"], cache=cache("broadcast.npz"))
        if verbose:
            print("aligning the broadcast tracking to the tactical timeline ...")
        ball, src, bb, _ = align.fuse_ball(T, B, verbose)
        if fused:
            np.savez(fused, ball=ball, src=src, bb=bb)
    else:                                                 # broadcast is optional
        ball, src = T["ball"].copy(), (~np.isnan(T["ball"][:, 0])).astype(np.int8)
    T["ball_tactical"] = T["ball"]
    T["ball"], T["ball_src"] = ball, src
    return T
