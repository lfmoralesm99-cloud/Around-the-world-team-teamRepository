"""Put the broadcast tracking on the tactical timeline and fuse the two ball tracks.

The broadcast recording is "NOT aligned" to the tactical one, but both show the same match, so the
alignment can be recovered from the data itself: for the correct lag the players seen by the
broadcast camera sit on top of players of the same team in the tactical tracking.  The lag is
piecewise constant (one value per recording block -- in practice one per half), which we find with

  1. a coarse search of the lag of each block of broadcast frames, using the ball position
     (sparse but very distinctive),
  2. a +-12 frame refinement of every distinct coarse lag, using all player positions,
  3. a per-frame validity test: a broadcast frame is used only if, at its block's lag, its
     players agree with the tactical players (median nearest same-team distance below a
     threshold).  This also removes replays and close-ups whose homography is meaningless.

Nothing here is specific to the match: no frame numbers, no constants tied to this game.
"""
import numpy as np

BLOCK = 2500          # broadcast frames per block in the coarse search (100 s)
COARSE_STEP = 5
AGREE_M = 2.0         # players agree if the median nearest-neighbour distance is below this
MIN_BALL = 150        # minimum co-visible ball samples for a coarse estimate


def _ball_err(tb, bb, b0, b1, lag):
    idx = np.arange(b0, min(b1, len(bb)))
    ti = idx + lag
    m = (ti >= 0) & (ti < len(tb))
    if m.sum() == 0:
        return np.inf, 0
    dd = np.linalg.norm(tb[ti[m]] - bb[idx[m]], axis=1)
    ok = ~np.isnan(dd)
    if ok.sum() < MIN_BALL:
        return np.inf, int(ok.sum())
    return float(np.median(dd[ok])), int(ok.sum())


def _player_dist(T, B, b_frames, lag):
    """Median distance from each broadcast player to the nearest tactical player of the same
    team, per broadcast frame (NaN where it cannot be computed)."""
    out = np.full(len(b_frames), np.nan, dtype=np.float32)
    nt = len(T["xy"])
    for k, b in enumerate(b_frames):
        t = b + lag
        if t < 0 or t >= nt:
            continue
        ds = []
        for tm in (0, 1):
            pb = B["xy"][b, B["team"] == tm]
            pb = pb[~np.isnan(pb[:, 0])]
            pt = T["xy"][t, T["team"] == tm]
            pt = pt[~np.isnan(pt[:, 0])]
            if len(pb) == 0 or len(pt) == 0:
                continue
            d = np.sqrt(((pb[:, None, :] - pt[None, :, :]) ** 2).sum(-1)).min(1)
            ds.append(d)
        if ds:
            ds = np.concatenate(ds)
            if len(ds) >= 4:
                out[k] = np.median(ds)
    return out


def estimate_lags(T, B, verbose=True):
    """Returns (lag_of_broadcast_frame, valid_broadcast_frame)."""
    tb, bb = T["ball"], B["ball"]
    nb, nt = len(bb), len(tb)
    lags = np.arange(-nb, nt, COARSE_STEP)
    # 1. coarse lag per block from the ball
    coarse = []
    for b0 in range(0, nb, BLOCK):
        idx = np.arange(b0, min(b0 + BLOCK, nb))
        ok = ~np.isnan(bb[idx, 0])
        if ok.sum() < MIN_BALL:
            coarse.append(None)
            continue
        idx = idx[ok]
        best = (np.inf, None)
        for lag in lags:
            ti = idx + lag
            m = (ti >= 0) & (ti < nt)
            if m.sum() < MIN_BALL:
                continue
            dd = np.abs(tb[ti[m]] - bb[idx[m]]).sum(1)
            good = ~np.isnan(dd)
            if good.sum() < MIN_BALL:
                continue
            e = np.median(dd[good])
            if e < best[0]:
                best = (e, int(lag))
        coarse.append(best[1] if best[0] < 3.0 else None)
    # 2. distinct lags (cluster coarse values that are within 25 frames), refine with players
    cands = []
    for c in coarse:
        if c is None:
            continue
        for g in cands:
            if abs(g[0] - c) <= 25:
                g.append(c)
                break
        else:
            cands.append([c])
    cands = [int(np.median(g)) for g in cands if len(g) >= 2]
    refined = []
    for c in cands:
        blocks = [i for i, v in enumerate(coarse) if v is not None and abs(v - c) <= 25]
        frames = np.concatenate([np.arange(i * BLOCK, min((i + 1) * BLOCK, nb), 50) for i in blocks])
        best = (np.inf, c)
        for lag in range(c - 12, c + 13):
            e = np.nanmedian(_player_dist(T, B, frames, lag))
            if e < best[0]:
                best = (e, lag)
        refined.append(best[1])
        if verbose:
            print(f"  lag candidate {c} -> refined {best[1]} (median player distance {best[0]:.2f} m, "
                  f"{len(blocks)} blocks)")
    # 3. lag per block: its own (refined) cluster, else the nearest block that has one
    block_lag = []
    for v in coarse:
        if v is None:
            block_lag.append(None)
        else:
            j = int(np.argmin([abs(v - r) for r in refined]))
            block_lag.append(refined[j] if abs(v - refined[j]) <= 40 else None)
    known = [i for i, v in enumerate(block_lag) if v is not None]
    for i in range(len(block_lag)):
        if block_lag[i] is None:
            block_lag[i] = block_lag[min(known, key=lambda j: abs(j - i))]
    lag = np.repeat(np.array(block_lag), BLOCK)[:nb]
    return lag, refined


def fuse_ball(T, B, verbose=True):
    """Ball track on the tactical timeline: the tactical detection where there is one, else the
    broadcast detection from frames that pass the player-agreement test.

    Returns ball (N, 2) float32, src (N,) int8 with 0 = none, 1 = tactical, 2 = broadcast,
    and the broadcast ball mapped to tactical time (for agreement features)."""
    lag, refined = estimate_lags(T, B, verbose)
    nb, nt = len(B["ball"]), len(T["ball"])
    bidx = np.arange(nb)
    # per-frame validity from player agreement, evaluated on every frame that has a ball
    has = ~np.isnan(B["ball"][:, 0])
    valid = np.zeros(nb, bool)
    for lg in sorted(set(lag.tolist())):
        fr = bidx[has & (lag == lg)]
        d = _player_dist(T, B, fr, lg)
        valid[fr] = d < AGREE_M
    tt = bidx + lag
    m = valid & (tt >= 0) & (tt < nt)
    bb_on_t = np.full((nt, 2), np.nan, dtype=np.float32)
    # if two broadcast frames map to the same tactical frame (block seams) the later one wins;
    # this is deterministic and affects at most a handful of frames
    bb_on_t[tt[m]] = B["ball"][bidx[m]]
    ball = T["ball"].copy()
    src = (~np.isnan(ball[:, 0])).astype(np.int8)
    fill = (src == 0) & ~np.isnan(bb_on_t[:, 0])
    ball[fill] = bb_on_t[fill]
    src[fill] = 2
    if verbose:
        print(f"  broadcast frames with a ball: {has.sum()}, accepted: {m.sum()}, "
              f"used to fill tactical gaps: {fill.sum()}")
    return ball, src, bb_on_t, {"lags": refined}
