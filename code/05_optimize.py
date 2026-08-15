# -*- coding: utf-8 -*-
"""
Stage 5 — per-subject optimisation, strictly within subject.

Design rationale (this is the part a reviewer will read):

  * Every subject has their own sensorimotor rhythm peak.  The channel x
    frequency r^2 maps in stage 0 show the Stand/Walk contrast sitting on a
    ~2 Hz-wide ridge whose centre moves between 9 and 16 Hz across subjects.
    A single 8-30 Hz band averages that ridge away, so the model is given a
    FINE filter bank (4 Hz wide, 2 Hz hop) and has to pick.

  * Picking is done by mutual information *inside the training fold only*.
    Band selection, spatial filters, tangent-space reference, scaler and
    classifier are all refit per fold; nothing is chosen on test data.

  * The reported number is NESTED: an inner 4-fold CV chooses the
    configuration, the outer fold only scores it.  With ~50 trials/class,
    choosing "the best of N pipelines" on the reported CV inflates accuracy
    by several points, which is the single most common reason a small-n
    BCI result fails to replicate.

Outputs: results/opt_scores.csv, fig_opt_confusion.png, fig_opt_summary.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CLASSES = sio.CLASS_NAMES
INK, GRID = "#1f2328", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})

# 4 Hz wide, 2 Hz hop — resolves a 2 Hz ridge while keeping each band wide
# enough for a stable 8x8 covariance from 4 s of data.
#
# PRIMARY analysis stops at 30 Hz on purpose.  Screening single channel x
# frequency effects showed subject S06's strongest Stand-vs-Walk feature at
# 36 Hz with an essentially FLAT topography (r = 0.47-0.61 on all eight
# electrodes).  Genuine sensorimotor rhythm is spatially graded; a uniform
# >30 Hz effect is the signature of scalp/neck EMG, which differs between an
# imagined postural transition and imagined gait for reasons that have
# nothing to do with cortex.  Reviewers of any MI paper look for exactly
# this.  The 30-40 Hz extension is available as a secondary analysis via
# --fmax 40 and must be reported with that caveat.
def fine_bands(fmax: float = 30.0) -> tuple[tuple[float, float], ...]:
    return tuple((lo, lo + 4) for lo in range(8, int(fmax) - 3, 2))


FINE_BANDS = fine_bands(30.0)


# ====================== feature blocks (fold-safe) ===================
class BankTangent(BaseEstimator, TransformerMixin):
    """(n, B, c, t) -> per-band tangent-space vectors, concatenated."""

    def __init__(self, metric="riemann", estimator="oas"):
        self.metric = metric
        self.estimator = estimator

    def _mk(self):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.pipeline import make_pipeline
        return make_pipeline(Covariances(self.estimator), TangentSpace(self.metric))

    def fit(self, X, y=None):
        self.p_ = [self._mk().fit(X[:, b]) for b in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.concatenate([p.transform(X[:, b]) for b, p in enumerate(self.p_)], axis=1)


class BankLogVar(BaseEstimator, TransformerMixin):
    """(n, B, c, t) -> log band-power, B*c features. Cheap, very interpretable."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X.var(axis=-1) + 1e-30).reshape(len(X), -1)


class BankCSP(BaseEstimator, TransformerMixin):
    """Per-band CSP log-variance features (FBCSP front end)."""

    def __init__(self, nfilter=4):
        self.nfilter = nfilter

    def _mk(self):
        from pyriemann.estimation import Covariances
        from pyriemann.spatialfilters import CSP
        from sklearn.pipeline import make_pipeline
        return make_pipeline(Covariances("oas"), CSP(nfilter=self.nfilter, log=True))

    def fit(self, X, y=None):
        self.p_ = [self._mk().fit(X[:, b], y) for b in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.concatenate([p.transform(X[:, b]) for b, p in enumerate(self.p_)], axis=1)


class EnvelopeModSpectrum(BaseEstimator, TransformerMixin):
    """
    Gait-cadence feature block.

    Walking imagery is *rhythmic* — the subject imagines two alternating
    steps inside the 5 s cue — whereas standing-up imagery is a single
    postural transition.  Trial-averaged band power cannot see that
    difference: both are "sustained sensorimotor ERD".  What differs is the
    slow modulation of the sensorimotor rhythm's amplitude envelope.

    So: band-pass -> Hilbert envelope -> spectrum of the envelope, pooled
    into 5 coarse modulation bins between 0.25 and 4 Hz (a 4 s epoch gives
    0.25 Hz resolution).  Step cadence for two imagined steps in ~4 s lands
    near 0.5 Hz; arm-swing coupling near 1 Hz.

    Features are the *shape* of that spectrum (each bin divided by the total
    0.25-4 Hz power), so this block is not just a restatement of band power.

    STATUS — tested, NOT supported as a standalone effect.  Screening all ten
    subjects, the largest normalised-shape correlation with Stand-vs-Walk was
    |r| = 0.13-0.31, peaking at a *different* modulation frequency in every
    subject (0.4, 1.2, 1.4, 1.6, 2.2, 3.4, 3.8, 4.0 Hz).  With ~40 bins
    screened and n≈100 trials the 5 % critical value is |r| ≈ 0.20, so that
    is what multiple comparisons alone would produce.  There is no evidence
    of a consistent imagined-gait cadence signature in this dataset.  The
    block is kept as a *candidate* only — the inner CV drops it when it does
    not pay — and the negative result is worth one paragraph in the paper.

    Input (n, B, c, t) -> output (n, B' * c * 5) log relative modulation power.
    """

    MOD_EDGES = (0.25, 0.75, 1.25, 2.0, 3.0, 4.0)

    def __init__(self, band_stride=2, fs=256.0):
        self.band_stride = band_stride
        self.fs = fs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy.signal import hilbert, welch

        Xs = X[:, ::self.band_stride]                       # (n, B', c, t)
        env = np.abs(hilbert(Xs, axis=-1))
        env = env - env.mean(axis=-1, keepdims=True)        # drop mean power
        f, P = welch(env, fs=self.fs, nperseg=min(env.shape[-1], 1024),
                     axis=-1, detrend=False)
        keep = (f >= self.MOD_EDGES[0]) & (f < self.MOD_EDGES[-1])
        tot = P[..., keep].sum(axis=-1, keepdims=True) + 1e-30
        feats = []
        for lo, hi in zip(self.MOD_EDGES[:-1], self.MOD_EDGES[1:]):
            m = (f >= lo) & (f < hi)
            feats.append(np.log(P[..., m].sum(axis=-1, keepdims=True) / tot + 1e-30)[..., 0])
        return np.stack(feats, axis=-1).reshape(len(X), -1)


def _mi(X, y):
    return mutual_info_classif(X, y, random_state=0)


def candidate_grid(n_trials: int):
    """(name, estimator, param_grid) triples; all take the fine bank."""
    lr = LogisticRegression(max_iter=5000, solver="lbfgs")   # multinomial-capable
    k_opts = [k for k in (16, 32, 64, 128) if k < n_trials * 0.8 * 3]

    return [
        ("fbts", Pipeline([("f", BankTangent()), ("sc", StandardScaler()),
                           ("sel", SelectKBest(_mi)), ("clf", clone(lr))]),
         {"sel__k": k_opts, "clf__C": [0.03, 0.1, 0.3, 1.0]}),

        ("fblv", Pipeline([("f", BankLogVar()), ("sc", StandardScaler()),
                           ("sel", SelectKBest(_mi)),
                           ("clf", LDA(solver="lsqr", shrinkage="auto"))]),
         {"sel__k": [k for k in (8, 16, 32, 64) if k < n_trials * 0.8]}),

        ("fbcsp", Pipeline([("f", BankCSP()), ("sc", StandardScaler()),
                            ("sel", SelectKBest(_mi)), ("clf", clone(lr))]),
         {"f__nfilter": [2, 4], "sel__k": [8, 16, 32], "clf__C": [0.1, 0.3, 1.0]}),

        ("ems", Pipeline([("f", EnvelopeModSpectrum()), ("sc", StandardScaler()),
                          ("sel", SelectKBest(_mi)), ("clf", clone(lr))]),
         {"sel__k": [8, 16, 32, 64], "clf__C": [0.03, 0.1, 0.3, 1.0]}),
    ]


def make_ensemble(fitted):
    """Soft-vote the tuned members; equal weight, no extra fitting decisions."""
    return VotingClassifier([(n, m) for n, m in fitted], voting="soft")


# ============================ evaluation =============================
def nested_subject(sid, Xb, y, n_outer=5, seed=0, verbose=True):
    """
    Outer CV scores; inner 4-fold GridSearchCV chooses band count, feature
    block and regularisation.  Also scores the soft-vote ensemble of the
    three tuned blocks — the ensemble members are tuned on the inner folds
    only, so the outer estimate stays clean.
    """
    outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    oof = {k: np.empty_like(y) for k in ("best", "ens")}
    chosen = []

    for tr, te in outer.split(np.zeros(len(y)), y):
        inner = StratifiedKFold(4, shuffle=True, random_state=seed + 7)
        fitted, scores = [], {}
        for name, est, grid in candidate_grid(len(tr)):
            gs = GridSearchCV(est, grid, cv=inner, scoring="accuracy",
                              n_jobs=1, refit=True)
            gs.fit(Xb[tr], y[tr])
            fitted.append((name, gs.best_estimator_))
            scores[name] = gs.best_score_
        best_name = max(scores, key=scores.get)
        chosen.append(f"{best_name}")
        oof["best"][te] = dict(fitted)[best_name].predict(Xb[te])

        # members keep the hyper-parameters the inner CV chose, but are refit
        # on the full outer-training set by VotingClassifier
        ens = make_ensemble([(n, clone(m)) for n, m in fitted]).fit(Xb[tr], y[tr])
        oof["ens"][te] = ens.predict(Xb[te])

    out = {k: float((v == y).mean()) for k, v in oof.items()}
    if verbose:
        print(f"  {sid:14s} best={out['best']*100:5.1f}%  ens={out['ens']*100:5.1f}%  "
              f"picked={'/'.join(chosen)}", flush=True)
    return sid, out, oof, y


def run_task(task, cfg, jobs, bands):
    from importlib import import_module
    TASKS = import_module("03_transfer").TASKS

    data = []
    for date, subj in sio.list_subjects():
        _, Xb, y = sio.get_task_arrays(date, subj, bands=bands, **cfg)
        if TASKS[task] is not None:
            a, b = TASKS[task]
            ia, ib = CLASSES.index(a), CLASSES.index(b)
            m = np.isin(y, [ia, ib])
            Xb, y = Xb[m], (y[m] == ib).astype(int)
        data.append((f"{date}/{subj}", Xb.astype(np.float64), y))

    res = Parallel(n_jobs=jobs, verbose=0)(
        delayed(nested_subject)(sid, Xb, y) for sid, Xb, y in data)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=4.5)
    ap.add_argument("--ref", default="none")
    ap.add_argument("--tasks", default="3cls,SvW")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--fmax", type=float, default=30.0,
                    help="upper edge of the filter bank; >30 admits the "
                         "EMG-suspect range (secondary analysis only)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = dict(lo=8.0, hi=min(30.0, a.fmax), tmin=a.tmin, tmax=a.tmax, ref=a.ref)
    bands = fine_bands(a.fmax)

    rows, conf = [], {}
    for task in a.tasks.split(","):
        chance = 1 / (3 if task == "3cls" else 2)
        print(f"\n===== {task}  (fine bank {len(bands)} bands up to {a.fmax:.0f} Hz, "
              f"t∈[{a.tmin},{a.tmax}], chance {chance*100:.0f}%) =====", flush=True)
        t0 = time.time()
        res = run_task(task, cfg, a.jobs, bands)
        for sid, out, oof, y in res:
            for k, v in out.items():
                rows.append(dict(subject=sid, task=task, model=k, acc=v, n=len(y)))
        conf[task] = [(sid, y, oof["ens"], oof["best"]) for sid, _, oof, y in res]
        m = np.mean([o["best"] for _, o, _, _ in res])
        me = np.mean([o["ens"] for _, o, _, _ in res])
        print(f"  MEAN nested: best-block {m*100:.1f}%   ensemble {me*100:.1f}%   "
              f"({time.time()-t0:.0f}s)", flush=True)

    tag = a.tag or f"fmax{int(a.fmax)}"
    df = pd.DataFrame(rows)
    df["fmax"] = a.fmax
    df.to_csv(os.path.join(R_DIR, f"opt_scores_{tag}.csv"), index=False)
    np.save(os.path.join(R_DIR, f"opt_conf_{tag}.npy"),
            np.array(conf, dtype=object), allow_pickle=True)
    print("\nwrote", os.path.join(R_DIR, f"opt_scores_{tag}.csv"))


if __name__ == "__main__":
    main()
