"""The opening kick-off of a half.

The graded block of a half may start well before the ball is first played: the second half of
the challenge match starts about two minutes early, with players and spare balls on the pitch.
Nothing can be in possession before the kick-off, so those frames are dead ball by definition --
but the frame classifier, which has only ever seen periods that begin at the kick-off, mistakes
the warm-up for play.  Two pieces of evidence that need no learning settle it:

  * the provided Smart Tagging marks the opening kick-off of each half ("SET PIECES", type
    "Kick Off (Start)") with a short window around it; everything in the half before that
    window is dead ball;
  * inside the window, the Laws of the Game apply: until the ball is kicked every player is in
    his own half.  A "possession" that is already over while both teams are still lined up in
    their own halves cannot have happened.

If a half has no kick-off tag, nothing is changed and the classifier's output stands.
"""
import numpy as np

from features import FPS, attack_sign

OWN_HALF_TOL = 1.5        # metres a player may stand beyond the halfway line
MIN_PLAYERS = 9           # players of each team that must be tracked
MIN_SHARE = 0.9           # ... and the share of them that must be on their own side
SLACK = 2 * FPS           # the line-up is looked for up to this long after the tag window


def own_halves(xy, team):
    """Boolean per frame: both teams are lined up in their own halves."""
    s = attack_sign(xy, team)                     # +1: team 0 defends the -x half
    ok = np.ones(len(xy), bool)
    for k, sg in ((0, s), (1, -s)):
        x = xy[:, team == k, 0] * sg              # own half is x < 0 after this flip
        seen = ~np.isnan(x)
        n = seen.sum(1)
        with np.errstate(invalid="ignore"):
            own = (x < OWN_HALF_TOL).sum(1)
        ok &= (n >= MIN_PLAYERS) & (own >= MIN_SHARE * np.maximum(n, 1))
    return ok


def opening_tag(tags, first, last):
    """(start_frame, end_frame) of the opening kick-off tag of the half spanning video frames
    first..last, or None.  tags: DataFrame from io_bundle.read_smart_tagging."""
    if tags is None or "Type" not in tags:
        return None
    k = tags[(tags["code"] == "SET PIECES") & (tags["Type"] == "Kick Off (Start)")]
    for r in k.sort_values("start").itertuples():
        f0, f1 = int(round(r.start * FPS)), int(round(r.end * FPS))
        if f1 >= first and f0 <= last:
            return f0, f1
    return None


def gate(labels, xy, team, tag, first):
    """Apply the two rules to the decoded labels of one half (array indexed from `first`).
    Returns the corrected labels and the number of frames changed."""
    if tag is None:
        return labels, 0
    out = labels.copy()
    n = len(out)
    t0 = min(max(tag[0] - first, 0), n)
    t1 = min(max(tag[1] - first + SLACK, 0), n)
    out[:t0] = 2                                           # rule 1: dead before the kick-off window
    if t1 > t0:
        cfg = own_halves(xy[t0:t1], team)
        if cfg[:FPS].mean() > 0.5:                         # the teams are lined up in the window
            t_e = t0 + int(np.flatnonzero(cfg)[-1])        # last frame of the line-up
            ch = np.flatnonzero(np.diff(out) != 0)
            st = np.r_[0, ch + 1]
            en = np.r_[ch + 1, n]
            for s, e in zip(st, en):
                if out[s] != 2 and e - 1 <= t_e:           # rule 2: over before the line-up broke
                    out[s:e] = 2
    return out, int((out != labels).sum())
