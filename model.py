"""Frame classifier: gradient-boosted trees over the per-frame features, 3 classes (us / them / dead)."""
import lightgbm as lgb
import numpy as np

PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=1.0, max_bin=127, verbose=-1, seed=7, deterministic=True,
              force_row_wise=True, num_threads=2)
ROUNDS = 500


def fit(X, y, names, rounds=ROUNDS, params=None):
    ds = lgb.Dataset(X, label=y, feature_name=list(names), free_raw_data=True)
    return lgb.train(params or PARAMS, ds, num_boost_round=rounds)


def predict(model, X0, X1):
    """Average of the two views, expressed as P(H), P(A), P(D)."""
    p0 = model.predict(X0, num_threads=2)
    p1 = model.predict(X1, num_threads=2)[:, [1, 0, 2]]
    return ((p0 + p1) / 2).astype(np.float32)
