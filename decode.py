"""From per-frame probabilities to a label sequence made of whole possessions.

The ranking metric scores possessions as objects, so one spurious flip in the middle of a long
possession costs two possessions.  The frame classifier is therefore followed by

  1. a Viterbi pass over the three states with a price on every change of state (a hidden Markov
     model with uniform emissions weight and "sticky" transitions), and
  2. a clean-up that removes what the definition of the task rules out: a spell too short to be
     controlled possession is merged with the spell of the opponent next to it, and a stoppage
     too short to be a stoppage is absorbed by the possession around it.

All constants were selected on the open sample games (leave-one-game-out), never on the
challenge match.
"""
import numpy as np

H, A, D = 0, 1, 2


def viterbi(logp, c_switch, c_dead_in, c_dead_out):
    """logp: (T, 3) log-probabilities.  Costs: H<->A, alive->D, D->alive."""
    T = len(logp)
    cost = np.array([[0.0, c_switch, c_dead_in],
                     [c_switch, 0.0, c_dead_in],
                     [c_dead_out, c_dead_out, 0.0]])
    back = np.zeros((T, 3), dtype=np.int8)
    s0, s1, s2 = (float(v) for v in logp[0])
    lp = logp.tolist()
    c01, c02, c20 = cost[0, 1], cost[0, 2], cost[2, 0]
    bk = back.tolist()
    for t in range(1, T):
        e0, e1, e2 = lp[t]
        # into H
        a, b, c = s0, s1 - c01, s2 - c20
        if a >= b and a >= c:
            n0, k0 = a, 0
        elif b >= c:
            n0, k0 = b, 1
        else:
            n0, k0 = c, 2
        a, b, c = s0 - c01, s1, s2 - c20
        if b >= a and b >= c:
            n1, k1 = b, 1
        elif a >= c:
            n1, k1 = a, 0
        else:
            n1, k1 = c, 2
        a, b, c = s0 - c02, s1 - c02, s2
        if c >= a and c >= b:
            n2, k2 = c, 2
        elif a >= b:
            n2, k2 = a, 0
        else:
            n2, k2 = b, 1
        s0, s1, s2 = n0 + e0, n1 + e1, n2 + e2
        bk[t] = (k0, k1, k2)
    out = np.empty(T, dtype=np.int64)
    k = int(np.argmax([s0, s1, s2]))
    for t in range(T - 1, -1, -1):
        out[t] = k
        k = bk[t][k]
    return out


def _runs(lab):
    ch = np.flatnonzero(np.diff(lab) != 0)
    st = np.r_[0, ch + 1]
    en = np.r_[ch + 1, len(lab)]
    return st, en, lab[st]


def cleanup(lab, min_spell, min_dead):
    lab = lab.copy()
    for _ in range(50):
        st, en, v = _runs(lab)
        done = True
        for i in range(len(st)):
            L = en[i] - st[i]
            prev = v[i - 1] if i > 0 else None
            nxt = v[i + 1] if i + 1 < len(st) else None
            if v[i] == D and L < min_dead and prev is not None and nxt is not None:
                # a hole in a possession; if the two sides disagree, split it in the middle
                if prev == nxt:
                    lab[st[i]:en[i]] = prev
                else:
                    mid = (st[i] + en[i]) // 2
                    lab[st[i]:mid] = prev
                    lab[mid:en[i]] = nxt
                done = False
                break
            if v[i] != D and L < min_spell and prev is not None and prev != D and prev != v[i]:
                lab[st[i]:en[i]] = prev
                done = False
                break
            if v[i] != D and L < min_spell and (prev is None or prev == D) and nxt is not None \
                    and nxt != D and nxt != v[i]:
                # a touch right after a restart, lost at once: the spell belongs to who follows
                lab[st[i]:en[i]] = nxt
                done = False
                break
        if done:
            break
    return lab


def decode(P, c_switch, c_dead_in, c_dead_out, min_spell, min_dead, power=1.0):
    logp = np.log(np.clip(P.astype(np.float64), 1e-6, 1.0)) * power
    return cleanup(viterbi(logp, c_switch, c_dead_in, c_dead_out), min_spell, min_dead)
