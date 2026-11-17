"""Local re-implementation of the two headline metrics, written from EVALUATION.md.

The official scorer (evaluation/possession_metrics.py in the organisers' repository) is the
specification; this module exists only so the model can be selected on *other* games without that
repository at hand.  It follows the published description:

  macroF1  mean of the per-class F1 of H, A and D over graded frames; an abstention (X) on a graded
           frame is a miss.
  spellF1  abstentions are forward-filled within a half (back-filled at its head); a possession is a
           maximal run of H or of A; predicted and true possessions of the same team are matched
           one-to-one by temporal IoU; the F1 of that matching is averaged over the IoU thresholds
           0.3, 0.4, 0.5, 0.6 and 0.7.

Labels are small integers: 0 = H, 1 = A, 2 = D, 3 = X.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

H, A, D, X = 0, 1, 2, 3
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def macro_f1(true, pred):
    f = []
    for c in (H, A, D):
        tp = np.sum((true == c) & (pred == c))
        fp = np.sum((true != c) & (pred == c))
        fn = np.sum((true == c) & (pred != c))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))


def fill_abstentions(pred):
    """Forward-fill X with the last non-X label, back-fill at the head. One half at a time."""
    pred = pred.copy()
    idx = np.where(pred != X, np.arange(len(pred)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(pred != X) if (pred != X).any() else None
    if first is None:
        return np.full_like(pred, D)
    idx[idx < 0] = first
    return pred[idx]


def spells(lab):
    """[(start, end_exclusive, team)] for every maximal run of H or A."""
    n = len(lab)
    if n == 0:
        return np.zeros((0, 3), dtype=np.int64)
    ch = np.flatnonzero(np.diff(lab) != 0)
    st = np.r_[0, ch + 1]
    en = np.r_[ch + 1, n]
    v = lab[st]
    m = v < D
    return np.stack([st[m], en[m], v[m]], 1)


def _match_count(ts, ps, thr):
    if len(ts) == 0 or len(ps) == 0:
        return 0
    rows, cols = [], []
    j0 = 0
    for i, (s, e, t) in enumerate(ts):
        while j0 < len(ps) and ps[j0, 1] <= s:
            j0 += 1
        j = j0
        while j < len(ps) and ps[j, 0] < e:
            if ps[j, 2] == t:
                inter = min(e, ps[j, 1]) - max(s, ps[j, 0])
                union = max(e, ps[j, 1]) - min(s, ps[j, 0])
                if inter > 0 and inter / union >= thr:
                    rows.append(i)
                    cols.append(j)
            j += 1
    if not rows:
        return 0
    g = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(ts), len(ps)))
    return int((maximum_bipartite_matching(g, perm_type="column") >= 0).sum())


def spell_f1(true_halves, pred_halves):
    """true_halves / pred_halves: lists of label arrays, one per half (graded frames only)."""
    ts, ps, off = [], [], 0
    for t, p in zip(true_halves, pred_halves):
        p = fill_abstentions(np.asarray(p))
        a, b = spells(np.asarray(t)), spells(p)
        a[:, :2] += off
        b[:, :2] += off
        ts.append(a)
        ps.append(b)
        off += len(t) + 10          # keep halves apart
    ts, ps = np.concatenate(ts), np.concatenate(ps)
    f = []
    for thr in THRESHOLDS:
        m = _match_count(ts, ps, thr)
        f.append(2 * m / (len(ts) + len(ps)) if (len(ts) + len(ps)) else 0.0)
    return float(np.mean(f)), len(ts), len(ps)


def score(true_halves, pred_halves):
    t = np.concatenate(true_halves)
    p = np.concatenate(pred_halves)
    s, nt, npred = spell_f1(true_halves, pred_halves)
    return {"spellF1": s, "macroF1": macro_f1(t, p), "n_true": nt, "n_pred": npred}
