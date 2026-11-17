"""Readers for the match bundle: FIFA EPTS tracking (txt + xml) and the Smart Tagging XML.

Everything returned is indexed by *video frame* of the source it was read from: row i of the
tracking arrays is line i+1 of the txt file, i.e. frame i of that source's video. Coordinates
are converted to metres with the origin at the pitch centre (x along the length, y across).
"""
import io
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PITCH = (105.0, 68.0)          # size the normalised tracking coordinates refer to (xml FieldSize)


def _strip_ns(tag):
    return tag.split("}")[-1]


def read_epts_meta(xml_path):
    """Team of every track slot, in the column order of the txt file. 0 = home, 1 = away."""
    root = ET.parse(xml_path).getroot()
    home_id = away_id = None
    teams, player_team, order, fps = {}, {}, [], 25.0
    for el in root.iter():
        tag = _strip_ns(el.tag)
        if tag == "Score":
            home_id, away_id = el.get("idLocalTeam"), el.get("idVisitingTeam")
        elif tag == "Team":
            name = next((c.text for c in el if _strip_ns(c.tag) == "Name"), None)
            direction = None
            for p in el.iter():
                if _strip_ns(p.tag) == "ProviderParameter":
                    kv = {_strip_ns(c.tag): c.text for c in p}
                    if kv.get("Name") == "attack_direction_first_half":
                        direction = kv.get("Value")
            teams[el.get("id")] = {"name": name, "attack_direction_first_half": direction}
        elif tag == "Player":
            player_team[el.get("id")] = el.get("teamId")
        elif tag == "FrameRate":
            fps = float(el.text)
        elif tag == "PlayerChannelRef":
            m = re.match(r"player(\d+)_x$", el.get("playerChannelId"))
            if m:
                order.append(m.group(1))
    team = np.array([0 if player_team[p] == home_id else 1 for p in order], dtype=np.int8)
    return {"team": team, "fps": fps, "home": teams[home_id], "away": teams[away_id],
            "n_tracks": len(order)}


def read_epts(txt_path, xml_path, cache=None):
    """Tracking of one source.

    Returns a dict with
      xy    (N, P, 2) float32, metres, NaN where the slot is empty
      spd   (N, P)    float32, the provider's third channel
      ball  (N, 2)    float32, metres, NaN where the ball was not detected
      team  (P,)      int8, 0 = home, 1 = away
    """
    meta = read_epts_meta(xml_path)
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return {**meta, "xy": z["xy"], "spd": z["spd"], "ball": z["ball"]}
    with open(txt_path, "rb") as f:
        raw = f.read()
    raw = raw.replace(b":", b",").replace(b";", b",")
    df = pd.read_csv(io.BytesIO(raw), header=None, dtype=np.float64, na_values=["NaN"],
                     engine="c")
    del raw
    a = df.to_numpy()
    del df
    n_tracks = meta["n_tracks"]
    assert a.shape[1] == 1 + 3 * n_tracks + 3, a.shape
    assert (a[:, 0] == np.arange(1, len(a) + 1)).all(), "txt frame counter is not 1..N"
    pl = a[:, 1:1 + 3 * n_tracks].reshape(len(a), n_tracks, 3)
    xy = np.stack([(pl[:, :, 0] - 0.5) * PITCH[0], (pl[:, :, 1] - 0.5) * PITCH[1]], -1)
    b = a[:, 1 + 3 * n_tracks:]
    ball = np.stack([(b[:, 0] - 0.5) * PITCH[0], (b[:, 1] - 0.5) * PITCH[1]], -1)
    out = {"xy": xy.astype(np.float32), "spd": pl[:, :, 2].astype(np.float32),
           "ball": ball.astype(np.float32)}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **out)
    return {**meta, **out}


def read_smart_tagging(xml_path):
    """All tagged instances as a DataFrame: start/end in seconds of the tactical video, the
    code, and one column per label group."""
    rows = []
    for inst in ET.parse(xml_path).getroot().iter("instance"):
        row = {"id": int(inst.findtext("ID")), "start": float(inst.findtext("start")),
               "end": float(inst.findtext("end")), "code": inst.findtext("code")}
        for lab in inst.iter("label"):
            row[lab.findtext("group")] = lab.findtext("text")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
