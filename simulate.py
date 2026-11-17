"""Make the corrected tracking of the open sample games look like automatic tracking.

The sample games are *corrected* data: every player is there on every frame and the ball has a
position exactly when it is in play.  The challenge match is automatic data, where

  * the ball track is cut in long pieces: it is usually lost when play stops, but it is also lost
    for a while in open play, and it is often still tracked while the ball is dead -- rolling out,
    in the hands of the player who walks to the restart, passed between team-mates before a goal
    kick;
  * the ball track is low-pass filtered (about a third as many sharp changes of direction per
    minute as in corrected data) and sits further from the player who controls it (median
    distance to the nearest player 1.8 m, against 0.5 m in corrected data);
  * a ball in the air is projected onto the pitch metres away from where it is (the two cameras
    of the challenge match disagree by a median 0.7 m on a slow ball and 2.7 m on a fast one);
  * players come and go (fragmented tracks), and there is a little positional noise.

A model trained on the clean version would learn "ball position present == ball in play" and
"ball next to a player == that player's team has it", which are both false on automatic data.
This module re-creates those artefacts at random, with a seeded generator, so that the model has
to rely on what the ball and the players *do*.

Every rate below was set from label-free statistics of the challenge tracking (share of frames
without ball, length of ball tracks, how often a track runs through an excursion beyond a line,
speed of the ball on the first frame of a track, disagreement between the two cameras, players
per frame).  None was fitted to a label.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from io_metrica import _runs

FPS = 25
HALF_L, HALF_W = 52.5, 34.0


def _drift(rng, n, k, corr):
    """Smooth unit-variance noise, (n, k)."""
    w = gaussian_filter1d(rng.normal(0, 1, (n, k)), corr, axis=0, mode="nearest")
    sd = w.std(0, keepdims=True)
    return w / np.where(sd > 0, sd, 1.0)


def _smooth_runs(ball, sigma):
    """Low-pass every uninterrupted piece of the ball track.

    A piece is continued backwards at its initial velocity before filtering, because that is how
    the automatic tracks look: on the first frame of a track the ball is already at full speed
    (median 12 m/s on the challenge match), it does not ramp up from rest."""
    out = ball.copy()
    pad = int(3 * sigma)
    st, en, v = _runs((~np.isnan(ball[:, 0])).astype(np.int8))
    for s, e, has in zip(st, en, v):
        if not has or e - s <= 2:
            continue
        seg = ball[s:e].astype(np.float64)
        k = min(4, e - s - 1)
        vel = (seg[k] - seg[min(1, k - 1)]) / max(k - min(1, k - 1), 1)
        pre = seg[0] + vel * np.arange(-pad, 0)[:, None]
        out[s:e] = gaussian_filter1d(np.concatenate([pre, seg]), sigma, axis=0, mode="nearest")[pad:]
    return out


def _smoothstep(m):
    w = np.linspace(0, 1, m + 2)[1:-1, None]
    return w * w * (3 - 2 * w)


def _dead_ball_path(rng, ball, xy, team, s, e, mode):
    """Positions of a ball that stays tracked during the stoppage s..e-1 (None where it is not).

    mode: "full"  tracked from end to end -- rolls out, is fetched, carried to the restart
          "trail" tracked for a moment after play stopped
          "lead"  picked up before the restart, in the hands / at the feet of the taker
    While it is carried the ball follows the real trajectory of a player (the one who takes the
    restart, sometimes a team-mate first, who then passes it on), so it moves the way a ball moves
    in a stoppage: at walking pace, next to a player, sometimes changing hands."""
    L = e - s
    path = np.full((L, 2), np.nan)
    spot = ball[e]
    v0 = (ball[s - 1] - ball[max(s - 3, 0)]) * (FPS / 2.0) if s >= 3 else np.zeros(2)
    tau = 0.6

    def rollout(V):
        t = (np.arange(1, V + 1) / FPS)[:, None]
        return ball[s - 1] + v0 * tau * (1 - np.exp(-t / tau)) + rng.normal(0, 0.05, (V, 2))

    # the taker: nearest player to the ball on the frame it is played
    d = np.linalg.norm(xy[e] - spot, axis=1)
    d = np.where(np.isnan(d), np.inf, d)
    taker = int(np.argmin(d))

    def follow(p, a, b):
        """Ball carried by player p over stoppage-relative frames a..b-1, or None."""
        pos = xy[s + a:s + b, p].astype(np.float64)
        if b <= a or np.isnan(pos).any():
            return None
        off = _drift(rng, b - a, 2, corr=8) * 0.25 if b - a >= 12 else rng.normal(0, 0.2, (1, 2))
        return pos + off

    if mode == "trail":
        V = int(min(L - 1, rng.integers(8, 120)))
        path[:V] = rollout(V)
        return path
    if mode == "lead":
        U = int(min(L - 1, rng.integers(10, 175)))
        seg = follow(taker, L - U, L) if rng.random() < 0.7 else None
        if seg is None:                                   # resting on the spot
            seg = spot + rng.normal(0, 0.05, (U, 2)) + np.cumsum(rng.normal(0, 0.01, (U, 2)), 0)
        path[L - U:] = seg
        return path
    # full
    V = int(min(L // 4, rng.integers(8, 50)))
    path[:V] = rollout(V)
    cur = path[V - 1].copy()
    t = V
    # sometimes a team-mate has it first and passes it to the taker (keeper to defender, ...)
    mates = np.flatnonzero((team == team[taker]) & np.isfinite(d))
    mates = mates[mates != taker]
    if len(mates) and L - V >= 100 and rng.random() < 0.35:
        helper = int(mates[np.argmin(d[mates])])
        t1 = t + int(rng.integers(12, 40))                # fetched by the helper
        t2 = t1 + int(rng.integers(15, max(16, (L - t1) // 2)))
        seg = follow(helper, t1, t2)
        if seg is not None and t2 < L - 20:
            path[t:t1] = cur * (1 - _smoothstep(t1 - t)) + seg[0] * _smoothstep(t1 - t)
            path[t1:t2] = seg
            cur, t = seg[-1].copy(), t2
            hop = int(rng.integers(6, 20))                # the pass to the taker
        else:
            hop = int(rng.integers(12, 60))
    else:
        hop = int(rng.integers(12, 60))
    hop = int(min(hop, max(L - t - 5, 1)))
    seg = follow(taker, t + hop, L)
    if seg is None:
        seg = np.repeat(spot[None, :], L - t - hop, 0) + rng.normal(0, 0.05, (L - t - hop, 2))
    w = _smoothstep(hop)
    path[t:t + hop] = cur * (1 - w) + seg[0] * w + rng.normal(0, 0.05, (hop, 2))
    path[t + hop:] = seg
    return path


def _lift(rng, ball, alive, sign):
    """A ball in the air is seen metres away from the point of the pitch it is above.

    Long fast stretches of ball movement are treated as flights; each gets a bump away from the
    camera side, zero at both ends, whose size grows with the flight time."""
    out = ball.copy()
    spd = np.zeros(len(ball))
    spd[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=1) * FPS
    fast = alive & (np.nan_to_num(spd) > 12.0)
    st, en, v = _runs(fast.astype(np.int8))
    for s, e, f in zip(st, en, v):
        T = (e - s) / FPS
        if not f or T < 0.5 or rng.random() < 0.4:        # short, or a pass along the ground
            continue
        amp = min(12.0, 3.7 * T * T) * rng.uniform(0.2, 1.0)
        out[s:e, 1] += sign * amp * np.sin(np.pi * (np.arange(e - s) + 0.5) / (e - s))
    return out


def degrade(game, seed, strength=1.0, ball_sigma=0.5, ball_smooth=6.0):
    rng = np.random.default_rng(seed)
    xy0 = game["xy"]
    xy = xy0.copy()
    ball = game["ball"].copy()
    team = game["team"]
    n = len(ball)
    alive = ~np.isnan(ball[:, 0])

    # ---------------- ball in the air (before anything else touches the track)
    ball = _lift(rng, ball, alive, sign=1.0 if rng.random() < 0.5 else -1.0)
    clean = ball.copy()

    # ---------------- ball while dead
    st, en, v = _runs((~alive).astype(np.int8))
    for s, e, dead in zip(st, en, v):
        if not dead or s == 0 or e >= n:
            continue
        L = e - s
        if L < 10:
            continue
        # A stoppage at a line (throw-in, corner, goal kick) keeps its ball on screen far more
        # often than a foul in midfield.  On the challenge match a track that leaves the pitch
        # over a touchline comes back in without a break in 25 cases, ends outside in 13 and
        # starts outside in 18; long stoppages lose the ball.
        last, spot = clean[s - 1], clean[e]
        at_line = min(HALF_L - abs(last[0]), HALF_W - abs(last[1]),
                      HALF_L - abs(spot[0]), HALF_W - abs(spot[1])) < 2.5
        if L >= 750:
            p_full, p_trail, p_lead = 0.03, 0.25, 0.25
        elif at_line:
            p_full, p_trail, p_lead = (0.35, 0.22, 0.33) if L < 250 else (0.20, 0.25, 0.35)
        else:
            p_full, p_trail, p_lead = (0.25, 0.20, 0.20) if L < 250 else (0.10, 0.20, 0.25)
        r = rng.random() / strength
        mode = None
        if r < p_full and L >= 40:
            mode = "full"
        elif r < p_full + p_trail and s >= 3:
            mode = "trail"
        elif r < p_full + p_trail + p_lead:
            mode = "lead"
        if mode:
            path = _dead_ball_path(rng, clean, xy0, team, s, e, mode)
            ok = ~np.isnan(path[:, 0])
            ball[s:e][ok] = path[ok]

    # ---------------- ball while alive
    st, en, v = _runs(alive.astype(np.int8))
    for s, e, a in zip(st, en, v):
        if not a:
            continue
        L = e - s
        # late pick-up after the restart / early loss before the stoppage
        if rng.random() < 0.20 * strength and L > 30:
            ball[s:s + int(rng.integers(3, min(60, L // 2)))] = np.nan
        if rng.random() < 0.20 * strength and L > 30:
            ball[e - int(rng.integers(3, min(60, L // 2))):e] = np.nan
        # holes in open play: about one every two minutes of live ball
        k = rng.poisson(L / (120.0 * FPS) * strength)
        for _ in range(k):
            dur = int(np.clip(rng.lognormal(np.log(50), 1.1), 3, 600))
            a0 = int(rng.integers(s, max(s + 1, e - 1)))
            ball[a0:min(e, a0 + dur)] = np.nan
    # The automatic ball track is much smoother than the corrected one: a pass shows up as a bend
    # spread over half a second, not as a corner.  It also sits a little away from the player in
    # control.  Both were measured without labels on the challenge tracking (velocity change over
    # 0.24 s, acceleration, distance to the nearest player) and are matched here by a Gaussian
    # low-pass of every piece of track plus a slow random drift.
    ball = _smooth_runs(ball, ball_smooth)
    ball += (_drift(rng, n, 2, corr=20) * ball_sigma + rng.normal(0, 0.02, ball.shape)).astype(np.float32)

    # ---------------- bridged gaps: the tracker loses the ball and draws a straight line to where
    # it finds it again.  The challenge track has about 15 stretches per 10 minutes where the ball
    # moves for a second or more without any change of velocity, on average 3.3 m from the
    # nearest player; corrected data has one.  (Done last: these lines are perfectly straight.)
    st, en, v = _runs((alive & ~np.isnan(ball[:, 0])).astype(np.int8))
    for s, e, a in zip(st, en, v):
        if not a or e - s < 60:
            continue
        k = rng.poisson((e - s) / (35.0 * FPS) * strength)
        for _ in range(k):
            dur = int(np.clip(rng.lognormal(np.log(38), 0.55), 20, 220))
            if e - s - dur - 2 <= 1:
                continue
            a0 = int(rng.integers(s + 1, e - dur - 1))
            b0 = a0 + dur
            w = (np.arange(1, dur + 1) / (dur + 1.0))[:, None]
            ball[a0:b0] = ball[a0 - 1] * (1 - w) + ball[b0] * w

    # ---------------- players: fragmented tracks and noise
    P = xy.shape[1]
    for p in range(P):
        k = rng.poisson(n / (150.0 * FPS) * strength)
        for _ in range(k):
            dur = int(np.clip(rng.exponential(8 * FPS), 5, 60 * FPS))
            a0 = int(rng.integers(0, n))
            xy[a0:a0 + dur, p] = np.nan
    xy += (_drift(rng, n, P * 2, corr=12).reshape(n, P, 2) * 0.20 + rng.normal(0, 0.02, xy.shape)).astype(np.float32)
    out = dict(game)
    out["xy"], out["ball"] = xy, ball.astype(np.float32)
    return out
