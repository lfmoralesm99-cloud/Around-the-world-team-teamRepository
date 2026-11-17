"""Readers for the open Metrica Sports sample games (other matches, hand-made event data).

These three games are the only *labelled* data the pipeline uses.  For each one we return the
tracking in the same layout as the challenge bundle (metres, origin at the pitch centre) together
with a per-frame H / A / D label derived mechanically from the provider's event stream:

    D  -- the ball is out of play (the corrected tracking carries no ball position), and
    H/A -- the team of the most recent on-ball event (pass, carry, shot, recovery, set piece, ball
           lost).  Challenges are ignored: winning a duel without controlling the ball afterwards is
           not a change of possession under the challenge's definition.
"""
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PITCH = (105.0, 68.0)
GRANTING = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}
H, A, D = 0, 1, 2
MIN_DEAD = 5          # frames; shorter holes in the ball track are tracking glitches, not stoppages
MIN_SPELL = 8         # frames; a shorter spell between two spells of the opponent is a touch, not control


def _to_m(a):
    out = np.empty_like(a, dtype=np.float32)
    out[..., 0] = (a[..., 0] - 0.5) * PITCH[0]
    out[..., 1] = (a[..., 1] - 0.5) * PITCH[1]
    return out


def _runs(lab):
    ch = np.flatnonzero(np.diff(lab) != 0)
    st = np.r_[0, ch + 1]
    en = np.r_[ch + 1, len(lab)]
    return st, en, lab[st]


def _labels(n, ball, ev, min_spell=MIN_SPELL):
    """Per-frame H/A/D from the event stream.  ev: DataFrame with team (0/1), type, frame."""
    # 1. owner = team of the latest possession-granting event.  BALL LOST and CHALLENGE never
    #    grant possession: the loser already had the ball, and a duel won without control after
    #    it is explicitly not a change of possession.
    ev = ev[ev.type.isin(GRANTING)].sort_values("frame", kind="stable")
    fr = ev.frame.to_numpy()
    tm = ev.team.to_numpy()
    idx = np.searchsorted(fr, np.arange(n), side="right") - 1
    owner = np.where(idx >= 0, tm[np.maximum(idx, 0)], -1).astype(np.int8)
    # 2. dead ball = the corrected tracking has no ball, except holes of a few frames
    dead = np.isnan(ball[:, 0])
    st, en, v = _runs(dead.astype(np.int8))
    for s, e, d in zip(st, en, v):
        if d and e - s < MIN_DEAD and s > 0 and e < n:
            dead[s:e] = False
    lab = owner.copy()
    lab[dead | (owner < 0)] = D
    # 3. a very short spell squeezed after a spell of the other team is an uncontrolled touch:
    #    possession stays with the last team that had it
    changed = True
    while changed:
        changed = False
        st, en, v = _runs(lab)
        for i in range(1, len(st)):
            if v[i] != D and en[i] - st[i] < min_spell and v[i - 1] != D and v[i - 1] != v[i]:
                lab[st[i]:en[i]] = v[i - 1]
                changed = True
                break
    return lab


def read_csv_game(folder, name):
    """Sample games 1 and 2 (CSV)."""
    def trk(side):
        df = pd.read_csv(os.path.join(folder, f"{name}_RawTrackingData_{side}_Team.csv"), skiprows=2)
        a = df.to_numpy(dtype=np.float64)
        return a[:, 0].astype(int), a[:, 3:]
    per, h = trk("Home")
    _, a = trk("Away")
    ball = h[:, -2:]
    hp = h[:, :-2].reshape(len(h), -1, 2)
    ap = a[:, :-2].reshape(len(a), -1, 2)
    xy = np.concatenate([hp, ap], 1)
    team = np.array([0] * hp.shape[1] + [1] * ap.shape[1], dtype=np.int8)
    ev = pd.read_csv(os.path.join(folder, f"{name}_RawEventsData.csv"))
    ev = pd.DataFrame({"team": (ev["Team"] == "Away").astype(int), "type": ev["Type"],
                       "frame": ev["Start Frame"] - 1})
    ball = _to_m(ball)
    return {"xy": _to_m(xy), "ball": ball, "team": team, "period": per,
            "label": _labels(len(ball), ball, ev), "name": name}


def read_epts_game(folder, name):
    """Sample game 3 (FIFA EPTS txt + xml, events as json)."""
    root = ET.parse(os.path.join(folder, f"{name}_metadata.xml")).getroot()
    strip = lambda t: t.split("}")[-1]
    home_id = None
    pteam, order, params = {}, [], {}
    for el in root.iter():
        tag = strip(el.tag)
        if tag == "Score":
            home_id = el.get("idLocalTeam")
        elif tag == "Player":
            pteam[el.get("id")] = el.get("teamId")
        elif tag == "PlayerChannelRef":
            m = re.match(r"player(.+)_x$", el.get("playerChannelId"))
            if m:
                order.append(m.group(1))
        elif tag == "ProviderParameter":
            kv = {strip(c.tag): c.text for c in el}
            params[kv.get("Name")] = kv.get("Value")
    chan_player = {el.get("id"): el.get("playerId") for el in root.iter()
                   if strip(el.tag) == "PlayerChannel"}
    # The column layout changes at every substitution: one DataFormatSpecification per frame range.
    specs = []
    for el in root.iter():
        if strip(el.tag) == "DataFormatSpecification":
            refs = [c.get("playerChannelId") for c in el.iter() if strip(c.tag) == "PlayerChannelRef"]
            ids = [chan_player[r] for r in refs if r.endswith("_x")]
            specs.append((int(el.get("startFrame")), int(el.get("endFrame")),
                          [0 if pteam[p] == home_id else 1 for p in ids]))
    rows = []
    with open(os.path.join(folder, f"{name}_tracking.txt")) as f:
        for line in f:
            head, players, ball_s = line.rstrip("\n").split(":")
            rows.append((int(head), players, ball_s))
    n = len(rows)
    frames = np.array([r[0] for r in rows])
    n_home = max(sum(1 for t in s[2] if t == 0) for s in specs)
    n_away = max(sum(1 for t in s[2] if t == 1) for s in specs)
    xy = np.full((n, n_home + n_away, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    team = np.array([0] * n_home + [1] * n_away, dtype=np.int8)
    for i, (fr, players, ball_s) in enumerate(rows):
        spec = next(s for s in specs if s[0] <= fr <= s[1])
        vals = np.array([p.split(",") for p in players.split(";")], dtype=np.float64)
        tt = np.array(spec[2])
        h = vals[tt == 0]
        a = vals[tt == 1]
        xy[i, :len(h)] = h
        xy[i, n_home:n_home + len(a)] = a
        ball[i] = [float(v) for v in ball_s.split(",")[:2]]
    ball = _to_m(ball)
    per = np.where(frames <= int(params["first_half_end"]), 1, 2)
    evs = json.load(open(os.path.join(folder, f"{name}_events.json")))["data"]
    f0 = frames[0]
    ev = pd.DataFrame({"team": [0 if e["team"]["id"] == home_id else 1 for e in evs],
                       "type": [e["type"]["name"] for e in evs],
                       "frame": [e["start"]["frame"] - f0 for e in evs]})
    return {"xy": _to_m(xy), "ball": ball, "team": team, "period": per,
            "label": _labels(n, ball, ev), "name": name}


def load_all(root):
    games = [read_csv_game(os.path.join(root, "Sample_Game_1"), "Sample_Game_1"),
             read_csv_game(os.path.join(root, "Sample_Game_2"), "Sample_Game_2"),
             read_epts_game(os.path.join(root, "Sample_Game_3"), "Sample_Game_3")]
    return games
