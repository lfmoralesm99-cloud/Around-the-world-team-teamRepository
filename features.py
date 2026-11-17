"""Per-frame features from anonymous tracking: a set of player positions per team plus a ball that
may be missing.  The same code runs on the open sample games (training) and on the challenge match
(inference), and nothing in it relies on player identities, so fragmented tracks and identity
switches do not matter.

Every feature is expressed from the point of view of one team ("us") playing towards +x, so the
model is symmetric by construction: each frame is seen twice, once per team, and the two views
are combined at prediction time.
"""
import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

HALF_L, HALF_W = 52.5, 34.0
FPS = 25
NEAR_M = 2.0            # a player this close to the ball is "on the ball"
CAP = 250               # cap (frames) for "time since / until" features
KICK_DV = 3.5           # m/s change of ball velocity within 0.4 s that counts as a kick


def _ffill_idx(valid):
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx)


def _bfill_idx(valid):
    n = len(valid)
    idx = np.where(valid, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _runs_bool(m):
    ch = np.flatnonzero(m[1:] != m[:-1])
    st = np.r_[0, ch + 1]
    en = np.r_[ch + 1, len(m)]
    return st, en, m[st]


def _roll(x, a, b):
    """nan-mean of x over frames [t+a, t+b] (inclusive), clipped at the ends of the array."""
    n = len(x)
    ok = ~np.isnan(x)
    cs = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    cn = np.r_[0, np.cumsum(ok)]
    t = np.arange(n)
    lo = np.clip(t + a, 0, n)
    hi = np.clip(t + b + 1, 0, n)
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[hi] - cs[lo]) / cnt
    out[cnt == 0] = np.nan
    return out.astype(np.float32)


def _shift(x, k):
    """x at frame t+k (NaN outside the array)."""
    out = np.full(len(x), np.nan, dtype=np.float32)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def _vel(p):
    """Centred velocity over +-2 frames, m/s. p: (N, ..., 2)."""
    v = np.full_like(p, np.nan)
    v[2:-2] = (p[4:] - p[:-4]) * (FPS / 4.0)
    return v


def attack_sign(xy, team):
    """+1 if team 0 plays towards +x in this period: a team's players sit, on average, on the
    side of the pitch it defends."""
    m0 = np.nanmean(xy[:, team == 0, 0])
    m1 = np.nanmean(xy[:, team == 1, 0])
    return 1.0 if m0 < m1 else -1.0


def _team_stats(xy, vel, spd, mask):
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # frames with no player of a team
        x, y = xy[:, mask, 0], xy[:, mask, 1]
        out = {
            "n": (~np.isnan(x)).sum(1).astype(np.float32),
            "cx": np.nanmean(x, 1), "cy": np.nanmean(y, 1),
            "sx": np.nanstd(x, 1), "sy": np.nanstd(y, 1),
            "minx": np.nanmin(x, 1), "maxx": np.nanmax(x, 1),
            "spd": np.nanmean(spd[:, mask], 1),
            "vx": np.nanmean(vel[:, mask, 0], 1),
        }
    return out


def base(xy, team, ball):
    """Orientation-free quantities for one period."""
    n = len(xy)
    vel = _vel(xy)
    spd = np.linalg.norm(vel, axis=2)
    b = {"n": n}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b["spd_all"] = np.nanmean(spd, 1)
        b["spd_p90"] = np.nanpercentile(spd, 90, axis=1)
        b["spd_p25"] = np.nanpercentile(spd, 25, axis=1)
    b["team"] = [_team_stats(xy, vel, spd, team == k) for k in (0, 1)]
    for k in (0, 1):
        x = xy[:, team == k, 0]
        b["team"][k]["x"] = x
    # players in each penalty area (set-piece shapes), by side
    inbox = (np.abs(xy[:, :, 1]) < 20.16) & (np.abs(xy[:, :, 0]) > HALF_L - 16.5)
    b["box_neg"] = (inbox & (xy[:, :, 0] < 0)).sum(1).astype(np.float32)
    b["box_pos"] = (inbox & (xy[:, :, 0] > 0)).sum(1).astype(np.float32)

    # ---- ball
    has = ~np.isnan(ball[:, 0])
    b["has"] = has
    bv = _vel(ball)
    # fall back to a one-sided difference at the ends of a ball track
    fwd = np.full_like(ball, np.nan); fwd[:-1] = (ball[1:] - ball[:-1]) * FPS
    bwd = np.full_like(ball, np.nan); bwd[1:] = (ball[1:] - ball[:-1]) * FPS
    for alt in (fwd, bwd):
        m = np.isnan(bv[:, 0]) & ~np.isnan(alt[:, 0])
        bv[m] = alt[m]
    b["ball"], b["bv"] = ball, bv
    b["bspd"] = np.linalg.norm(bv, axis=1)
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)                # (N, P)
    for k in (0, 1):
        dk = np.where(np.isnan(d[:, team == k]), np.inf, d[:, team == k])
        part = np.sort(dk, axis=1)[:, :2]
        d1, d2 = part[:, 0].copy(), part[:, 1].copy()
        j = np.argmin(dk, axis=1)
        ps = spd[:, team == k][np.arange(n), j]
        pv = vel[:, team == k][np.arange(n), j]
        rel = np.linalg.norm(pv - bv, axis=1)
        bad = ~np.isfinite(d1)
        d1[bad] = np.nan; ps[bad] = np.nan; rel[bad] = np.nan
        d2[~np.isfinite(d2)] = np.nan
        b["team"][k].update({"d1": d1.astype(np.float32), "d2": d2.astype(np.float32),
                             "near_spd": ps, "near_rel": rel,
                             "n3": (dk < 3.0).sum(1).astype(np.float32),
                             "n8": (dk < 8.0).sum(1).astype(np.float32)})
        b["team"][k]["n3"][~has] = np.nan
        b["team"][k]["n8"][~has] = np.nan
    d0, d1_ = b["team"][0]["d1"], b["team"][1]["d1"]
    with np.errstate(invalid="ignore"):
        touch = np.zeros(n, dtype=np.int8)                           # 0 nobody, 1 team 0, 2 team 1
        t0 = (d0 < NEAR_M) & ~(d1_ < d0)
        t1 = (d1_ < NEAR_M) & ~(d0 <= d1_)
        touch[t0] = 1
        touch[t1] = 2
    b["touch"] = touch

    # ---- kicks: sharp changes of the ball's velocity, attributed to the team closest to the
    #      ball around that instant.  Unlike plain proximity this survives a ball that is
    #      detected a metre away from the player, because the *shape* of its path is unaffected.
    dv = np.full(n, np.nan, dtype=np.float32)
    dv[5:-5] = np.linalg.norm(bv[10:] - bv[:-10], axis=1)
    dvz = np.where(np.isnan(dv), -1.0, dv)
    peak = (dvz >= KICK_DV) & (dvz >= maximum_filter1d(dvz, 13, mode="nearest"))
    ev = np.flatnonzero(peak)
    if len(ev):                                   # one event per plateau
        ev = ev[np.r_[True, np.diff(ev) > 6]]
    m0 = _roll(np.where(np.isfinite(d0), d0, np.nan), -3, 3)
    m1 = _roll(np.where(np.isfinite(d1_), d1_, np.nan), -3, 3)
    margin = (np.nan_to_num(m1[ev], nan=50.0) - np.nan_to_num(m0[ev], nan=50.0)).astype(np.float32)   # >0: team 0 closer
    b["ev"], b["ev_margin"] = ev, margin
    b["ev_dv"] = dv[ev]
    b["dv"] = dv

    # ---- gaps in the ball track
    t = np.arange(n)
    li, ni = _ffill_idx(has), _bfill_idx(has)
    since = np.where(li >= 0, t - li, CAP).clip(0, CAP).astype(np.float32)
    until = np.where(ni < n, ni - t, CAP).clip(0, CAP).astype(np.float32)
    b["since"], b["until"] = since, until
    b["gap"] = np.where(has, 0, np.where((li >= 0) & (ni < n), ni - li, 2 * CAP)).clip(0, 2 * CAP).astype(np.float32)
    lic, nic = np.clip(li, 0, n - 1), np.clip(ni, 0, n - 1)
    last = ball[lic].copy(); last[li < 0] = np.nan
    nxt = ball[nic].copy(); nxt[ni >= n] = np.nan
    b["last"], b["next"] = last, nxt
    b["last_spd"] = np.where(li >= 0, b["bspd"][lic], np.nan)
    b["last_bv"] = bv[lic].copy(); b["last_bv"][li < 0] = np.nan
    b["next_spd"] = np.where(ni < n, b["bspd"][nic], np.nan)
    b["jump"] = np.linalg.norm(nxt - last, axis=1)
    # how long the current ball track has been running / will keep running
    start = _ffill_idx(~has)          # last frame without ball
    end = _bfill_idx(~has)
    b["track_age"] = np.where(has, t - start, 0).clip(0, 2 * CAP).astype(np.float32)
    b["track_left"] = np.where(has, end - t, 0).clip(0, 2 * CAP).astype(np.float32)

    # ---- evidence that play stopped (or did not): a slow ball with slow players around it is a
    #      stoppage only if something stopped play -- the ball left the pitch, or it came to rest.
    margin = np.minimum(HALF_L - np.abs(ball[:, 0]), HALF_W - np.abs(ball[:, 1]))
    b["margin"] = margin
    with np.errstate(invalid="ignore"):
        outside = np.nan_to_num(margin, nan=1.0) < -0.3
        slow = np.nan_to_num(b["bspd"], nan=99.0) < 1.0
    rest = uniform_filter1d(slow.astype(np.float32), 11, mode="nearest") > 0.999   # 0.44 s still
    for name, flag in (("out", outside), ("rest", rest)):
        li_, ni_ = _ffill_idx(flag), _bfill_idx(flag)
        b[f"t_since_{name}"] = np.where(li_ >= 0, t - li_, 2 * CAP).clip(0, 2 * CAP).astype(np.float32)
        b[f"t_until_{name}"] = np.where(ni_ < n, ni_ - t, 2 * CAP).clip(0, 2 * CAP).astype(np.float32)
    big = np.where(np.isnan(margin), 99.0, margin).astype(np.float32)
    for w in (50, 125, 250):
        b[f"margin_min_p{w}"] = minimum_filter1d(big, w + 1, origin=w // 2, mode="nearest")
        b[f"margin_min_f{w}"] = minimum_filter1d(big, w + 1, origin=-(w // 2), mode="nearest")

    # ---- coasting: the ball moves without any change of velocity.  For a second or more this is
    #      not a ball being played but a gap the tracker bridged with a straight line, and what the
    #      track says about who is near the ball during it is not to be trusted.
    with np.errstate(invalid="ignore"):
        coast = has & (np.nan_to_num(dv, nan=99.0) < 0.6) & (np.nan_to_num(b["bspd"]) > 1.5)
    cst, cen, cv = _runs_bool(coast)
    clen = np.zeros(n, dtype=np.float32)
    for s_, e_, c_ in zip(cst, cen, cv):
        if c_:
            clen[s_:e_] = e_ - s_
    b["coast_len"] = np.minimum(clen, 2 * CAP)
    b["coast"] = np.where(has, coast.astype(np.float32), np.nan)

    # ---- touches: who was last / will be next on the ball, and for how long
    tv = touch > 0
    lt, nt = _ffill_idx(tv), _bfill_idx(tv)
    b["last_touch"] = np.where(lt >= 0, touch[np.clip(lt, 0, n - 1)], 0)
    b["next_touch"] = np.where(nt < n, touch[np.clip(nt, 0, n - 1)], 0)
    b["t_last_touch"] = np.where(lt >= 0, t - lt, CAP).clip(0, CAP).astype(np.float32)
    b["t_next_touch"] = np.where(nt < n, nt - t, CAP).clip(0, CAP).astype(np.float32)
    # "holder": last team on the ball, carried through loose phases; length of its current hold
    holder = b["last_touch"]
    chg = np.r_[True, holder[1:] != holder[:-1]]
    run_start = np.maximum.accumulate(np.where(chg, t, 0))
    nxt_chg = np.r_[holder[1:] != holder[:-1], True]
    run_end = np.minimum.accumulate(np.where(nxt_chg, t, n)[::-1])[::-1]
    b["hold_age"] = (t - run_start).clip(0, 4 * CAP).astype(np.float32)
    b["hold_left"] = (run_end - t).clip(0, 4 * CAP).astype(np.float32)
    # number of frames with an actual touch by the holder inside the current hold
    ct = np.r_[0, np.cumsum(tv)]
    b["hold_touches"] = (ct[np.clip(run_end + 1, 0, n)] - ct[run_start]).clip(0, 4 * CAP).astype(np.float32)
    return b


WINS = ((-12, 12), (-37, 37), (-100, 100), (-250, 250))
PAST = ((-12, 0), (-25, 0), (-50, 0), (-100, 0), (-200, 0))
FUT = ((0, 12), (0, 25), (0, 50), (0, 100), (0, 200))
LAGS = (-100, -50, -25, -12, 12, 25, 50, 100)


def view(b, v, sign):
    """Feature matrix of one period seen by team v (0/1) playing towards sign * (+x)."""
    us, th = b["team"][v], b["team"][1 - v]
    s = sign if v == 0 else -sign                 # s = +1 if "us" plays towards +x
    f = {}
    ball, bv, has = b["ball"], b["bv"], b["has"]
    hasf = has.astype(np.float32)

    # --- ball state
    f["has"] = hasf
    f["bx"] = ball[:, 0] * s
    f["by_abs"] = np.abs(ball[:, 1])
    f["mx"] = HALF_L - np.abs(ball[:, 0])
    f["my"] = HALF_W - np.abs(ball[:, 1])
    f["bspd"] = b["bspd"]
    f["bvx"] = bv[:, 0] * s
    f["bvy_out"] = bv[:, 1] * np.sign(ball[:, 1])          # speed towards the nearer touchline
    f["bvx_out"] = bv[:, 0] * np.sign(ball[:, 0])          # ... towards the nearer goal line
    f["since"], f["until"], f["gap"] = b["since"], b["until"], b["gap"]
    f["track_age"], f["track_left"] = b["track_age"], b["track_left"]
    f["last_bx"] = b["last"][:, 0] * s
    f["last_mx"] = HALF_L - np.abs(b["last"][:, 0])
    f["last_my"] = HALF_W - np.abs(b["last"][:, 1])
    f["next_bx"] = b["next"][:, 0] * s
    f["next_mx"] = HALF_L - np.abs(b["next"][:, 0])
    f["next_my"] = HALF_W - np.abs(b["next"][:, 1])
    f["last_spd"], f["next_spd"], f["jump"] = b["last_spd"], b["next_spd"], b["jump"]
    f["last_vy_out"] = b["last_bv"][:, 1] * np.sign(b["last"][:, 1])
    f["last_vx_out"] = b["last_bv"][:, 0] * np.sign(b["last"][:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        f["jump_rate"] = b["jump"] / np.maximum(b["gap"], 1.0) * FPS

    for name in ("t_since_out", "t_until_out", "t_since_rest", "t_until_rest",
                 "margin_min_p50", "margin_min_p125", "margin_min_p250",
                 "margin_min_f50", "margin_min_f125", "margin_min_f250", "coast_len"):
        f[name] = b[name]

    # --- ball vs players
    for nm, tm in (("us", us), ("th", th)):
        f[f"d1_{nm}"] = tm["d1"]
        f[f"d2_{nm}"] = tm["d2"]
        f[f"n3_{nm}"] = tm["n3"]
        f[f"n8_{nm}"] = tm["n8"]
        f[f"near_spd_{nm}"] = tm["near_spd"]
        f[f"near_rel_{nm}"] = tm["near_rel"]
    f["d1_diff"] = us["d1"] - th["d1"]
    f["d1_min"] = np.fmin(us["d1"], th["d1"])
    code = {0: 0.0, 1: 1.0 if v == 0 else -1.0, 2: -1.0 if v == 0 else 1.0}
    lut = np.array([code[0], code[1], code[2]], dtype=np.float32)
    touch = lut[b["touch"]]
    f["touch"] = np.where(has, touch, np.nan)
    f["last_touch"] = lut[b["last_touch"]]
    f["next_touch"] = lut[b["next_touch"]]
    f["t_last_touch"], f["t_next_touch"] = b["t_last_touch"], b["t_next_touch"]
    f["hold_age"], f["hold_left"], f["hold_touches"] = b["hold_age"], b["hold_left"], b["hold_touches"]
    tu = (touch > 0).astype(np.float32); tu[~has] = np.nan
    tt = (touch < 0).astype(np.float32); tt[~has] = np.nan
    with np.errstate(invalid="ignore"):
        nearer = np.where(np.isnan(f["d1_diff"]), np.nan, (f["d1_diff"] < 0).astype(np.float32))

    # --- kicks (see base): who played the ball last / next, and how surely
    n = b["n"]
    t = np.arange(n)
    ev = b["ev"]
    sgn = 1.0 if v == 0 else -1.0
    em = b["ev_margin"] * sgn                          # > 0: "us" closer to the ball at the kick
    et = np.sign(em).astype(np.float32)
    k_last = np.searchsorted(ev, t, side="right") - 1  # index of the last kick at or before t
    k_next = k_last + 1

    def pick(arr, k, fill=0.0):
        ok = (k >= 0) & (k < len(ev))
        out = np.full(n, fill, dtype=np.float32)
        if len(ev):
            out[ok] = arr[np.clip(k, 0, len(ev) - 1)][ok]
        return out

    evf = ev.astype(np.float32)
    f["ev_last_team"] = pick(et, k_last)
    f["ev_last_margin"] = pick(em, k_last)
    f["ev_prev_team"] = pick(et, k_last - 1)
    f["ev_prev2_team"] = pick(et, k_last - 2)
    f["ev_next_team"] = pick(et, k_next)
    f["ev_next_margin"] = pick(em, k_next)
    f["ev_next2_team"] = pick(et, k_next + 1)
    f["ev_next3_team"] = pick(et, k_next + 2)
    f["t_ev_last"] = np.clip(t - pick(evf, k_last, -1e6), 0, CAP).astype(np.float32)
    f["t_ev_prev"] = np.clip(t - pick(evf, k_last - 1, -1e6), 0, 2 * CAP).astype(np.float32)
    f["t_ev_next"] = np.clip(pick(evf, k_next, 1e6) - t, 0, CAP).astype(np.float32)
    f["t_ev_next2"] = np.clip(pick(evf, k_next + 1, 1e6) - t, 0, 2 * CAP).astype(np.float32)
    f["ev_last_dv"] = pick(b["ev_dv"], k_last)
    # streaks: consecutive kicks by the same team ending at the last kick / starting at the next
    if len(ev):
        chg = np.r_[True, et[1:] != et[:-1]]
        grp = np.cumsum(chg)
        pos = np.arange(len(ev)) - np.maximum.accumulate(np.where(chg, np.arange(len(ev)), 0))
        size = np.bincount(grp)[grp]
        f["ev_streak_past"] = pick((pos + 1).astype(np.float32), k_last)
        f["ev_streak_next"] = pick((size - pos).astype(np.float32), k_next)
    else:
        f["ev_streak_past"] = np.zeros(n, dtype=np.float32)
        f["ev_streak_next"] = np.zeros(n, dtype=np.float32)
    ku = np.zeros(n, dtype=np.float32); ku[ev[et > 0]] = 1.0
    kt = np.zeros(n, dtype=np.float32); kt[ev[et < 0]] = 1.0
    f["dv"] = b["dv"]

    # --- team shape and movement
    for nm, tm, sg in (("us", us, s), ("th", th, s)):
        f[f"n_{nm}"] = tm["n"]
        f[f"cx_{nm}"] = tm["cx"] * sg
        f[f"sx_{nm}"] = tm["sx"]
        f[f"sy_{nm}"] = tm["sy"]
        f[f"spd_{nm}"] = tm["spd"]
        f[f"vx_{nm}"] = tm["vx"] * sg
    f["deep_us"] = us["minx"] * s if s > 0 else -us["maxx"]
    f["deep_th"] = th["maxx"] * s if s > 0 else -th["minx"]
    f["front_us"] = us["maxx"] * s if s > 0 else -us["minx"]
    f["front_th"] = th["minx"] * s if s > 0 else -th["maxx"]
    f["sx_diff"] = us["sx"] - th["sx"]
    f["sy_diff"] = us["sy"] - th["sy"]
    f["spd_diff"] = us["spd"] - th["spd"]
    f["cx_gap"] = (us["cx"] - th["cx"]) * s
    f["b_cx_us"] = (ball[:, 0] - us["cx"]) * s
    f["b_cx_th"] = (ball[:, 0] - th["cx"]) * s
    f["last_b_cx_us"] = (b["last"][:, 0] - us["cx"]) * s
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ahead_us = np.nanmean(np.where(np.isnan(us["x"]), np.nan, (us["x"] * s > (ball[:, 0] * s)[:, None]).astype(np.float32)), 1)
        behind_th = np.nanmean(np.where(np.isnan(th["x"]), np.nan, (th["x"] * s > (ball[:, 0] * s)[:, None]).astype(np.float32)), 1)
    f["ahead_us"] = np.where(has, ahead_us, np.nan)
    f["goalside_th"] = np.where(has, behind_th, np.nan)
    f["spd_all"], f["spd_p90"], f["spd_p25"] = b["spd_all"], b["spd_p90"], b["spd_p25"]
    f["box_att"] = b["box_pos"] if s > 0 else b["box_neg"]         # players in the box "us" attacks
    f["box_def"] = b["box_neg"] if s > 0 else b["box_pos"]

    # --- temporal context
    ctx = {"has": hasf, "bspd": f["bspd"], "spd_all": f["spd_all"], "spd_p90": f["spd_p90"],
           "tu": tu, "tt": tt, "d1_diff": f["d1_diff"], "d1_min": f["d1_min"],
           "vx_us": f["vx_us"], "vx_th": f["vx_th"], "sx_diff": f["sx_diff"], "spd_diff": f["spd_diff"],
           "cx_us": f["cx_us"], "nearer": nearer, "coast": b["coast"], "dv": b["dv"]}
    for name, x in ctx.items():
        for a, c in WINS:
            f[f"{name}_c{c}"] = _roll(x, a, c)
    for name in ("has", "bspd", "spd_all", "tu", "tt", "d1_diff", "vx_us", "sx_diff", "nearer"):
        x = ctx[name]
        for a, c in PAST:
            f[f"{name}_p{-a}"] = _roll(x, a, c)
        for a, c in FUT:
            f[f"{name}_f{c}"] = _roll(x, a, c)
    for name in ("spd_all", "bspd", "mx", "my", "d1_diff", "has", "bx"):
        x = ctx[name] if name in ctx else f[name]
        for k in LAGS:
            f[f"{name}_l{k}"] = _shift(x, k)
    for a, c in ((-50, 0), (-125, 0), (-250, 0), (1, 50), (1, 125), (1, 250)):
        tag = f"p{-a}" if a < 0 else f"f{c}"
        f[f"ku_{tag}"] = _roll(ku, a, c) * (c - a + 1)
        f[f"kt_{tag}"] = _roll(kt, a, c) * (c - a + 1)
        f[f"kdiff_{tag}"] = f[f"ku_{tag}"] - f[f"kt_{tag}"]
    # change of mean player speed and of ball presence across "now"
    f["spd_all_df50"] = f["spd_all_f50"] - f["spd_all_p50"]
    f["spd_all_df200"] = f["spd_all_f200"] - f["spd_all_p200"]
    f["has_df50"] = f["has_f50"] - f["has_p50"]
    f["tu_minus_tt_p50"] = f["tu_p50"] - f["tt_p50"]
    f["tu_minus_tt_f50"] = f["tu_f50"] - f["tt_f50"]
    f["tu_minus_tt_p200"] = f["tu_p200"] - f["tt_p200"]
    f["tu_minus_tt_f200"] = f["tu_f200"] - f["tt_f200"]
    f["t_frac"] = (np.arange(b["n"]) / max(b["n"] - 1, 1)).astype(np.float32)
    f["t_start"] = np.minimum(np.arange(b["n"]), 4 * CAP).astype(np.float32)
    names = list(f.keys())
    return np.stack([np.asarray(f[k], dtype=np.float32) for k in names], 1), names


def period_features(xy, team, ball):
    """Both views of one period: (X_home_view, X_away_view, names)."""
    sign = attack_sign(xy, team)
    b = base(xy, team, ball)
    x0, names = view(b, 0, sign)
    x1, _ = view(b, 1, sign)
    return x0, x1, names
