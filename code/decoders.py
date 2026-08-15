# -*- coding: utf-8 -*-
"""
Decoding pipelines for the Bitbrain Rest/Stand/Walk MI data.

Everything here is a plain scikit-learn estimator, so every fit (CSP filters,
tangent-space reference, scalers, feature selection) happens *inside* the CV
fold — no leakage.

Two input conventions:
  * plain pipelines   take X of shape (n_trials, n_ch, n_times)
  * filter-bank ones  take X of shape (n_trials, n_bands, n_ch, n_times)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances, TimeDelayCovariances
from pyriemann.spatialfilters import CSP as RiemCSP
from pyriemann.tangentspace import TangentSpace


# ============================ Transformers ===========================
class LogVar(BaseEstimator, TransformerMixin):
    """(n, c, t) -> log band-power per channel."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X.var(axis=-1) + 1e-30)


class WindowedCov(BaseEstimator, TransformerMixin):
    """
    Split each trial into `n_win` equal sub-windows, take the covariance of
    each, and hand back a block-diagonal-free stack of shape (n, n_win, c, c).

    Rationale: gait imagery is rhythmic, so *when* power changes inside the
    5 s cue may separate Walk from Stand even when the trial average does not.
    """

    def __init__(self, n_win=2, estimator="oas"):
        self.n_win = n_win
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n, c, t = X.shape
        w = t // self.n_win
        cov = Covariances(estimator=self.estimator)
        outs = [cov.fit_transform(X[:, :, i * w:(i + 1) * w]) for i in range(self.n_win)]
        return np.stack(outs, axis=1)


class PerBlockTangent(BaseEstimator, TransformerMixin):
    """(n, B, c, c) SPD blocks -> concatenated tangent-space vectors."""

    def __init__(self, metric="riemann"):
        self.metric = metric

    def fit(self, X, y=None):
        self.ts_ = [TangentSpace(metric=self.metric).fit(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.concatenate([ts.transform(X[:, i]) for i, ts in enumerate(self.ts_)], axis=1)


class FilterBank(BaseEstimator, TransformerMixin):
    """
    Apply one sub-pipeline per frequency band and concatenate the features.

    Input  : (n_trials, n_bands, n_ch, n_times)
    Output : (n_trials, n_bands * n_features_per_band)
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, y=None):
        self.pipes_ = [clone(self.base).fit(X[:, b], y) for b in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.concatenate([p.transform(X[:, b]) for b, p in enumerate(self.pipes_)], axis=1)

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y).transform(X)


# ============================= Pipelines =============================
def _lr(C=1.0):
    # lbfgs, not liblinear: sklearn >= 1.8 refuses liblinear for n_classes >= 3,
    # and the 3-class Rest/Stand/Walk problem needs a multinomial fit.
    return LogisticRegression(C=C, max_iter=5000, solver="lbfgs")


def _make_plain() -> dict[str, Pipeline]:
    """Pipelines that take (n, c, t)."""
    return {
        # --- classic baselines (what the lab already had) ---
        "logvar_lda": make_pipeline(LogVar(), LDA(solver="lsqr", shrinkage="auto")),
        "csp_lda": make_pipeline(
            Covariances("oas"), RiemCSP(nfilter=6, log=True),
            LDA(solver="lsqr", shrinkage="auto")),
        "csp_svm": make_pipeline(
            Covariances("oas"), RiemCSP(nfilter=6, log=True),
            StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),

        # --- Riemannian geometry (never tried in this project) ---
        "mdm": make_pipeline(Covariances("oas"), MDM(metric="riemann")),
        "fgmdm": make_pipeline(Covariances("oas"), FgMDM(metric="riemann")),
        "ts_lr": make_pipeline(Covariances("oas"), TangentSpace("riemann"),
                               StandardScaler(), _lr(1.0)),
        "ts_svm": make_pipeline(Covariances("oas"), TangentSpace("riemann"),
                                StandardScaler(), SVC(kernel="rbf", C=1.0)),

        # --- augmented (time-delay embedded) covariances ---
        # delays are in SAMPLES at 256 Hz; 5 samples ~ 20 ms ~ a quarter period
        # of the mu rhythm, which is where CSSP-style lags pay off.
        "acm_d5_ts_lr": make_pipeline(TimeDelayCovariances([1, 5], estimator="oas"),
                                      TangentSpace("riemann"), StandardScaler(), _lr(0.3)),
        "acm_d5_10_ts_lr": make_pipeline(TimeDelayCovariances([1, 5, 10], estimator="oas"),
                                         TangentSpace("riemann"), StandardScaler(), _lr(0.1)),

        # --- time-resolved covariance ---
        "wcov2_ts_lr": make_pipeline(WindowedCov(2), PerBlockTangent(),
                                     StandardScaler(), _lr(0.3)),
        "wcov4_ts_lr": make_pipeline(WindowedCov(4), PerBlockTangent(),
                                     StandardScaler(), _lr(0.1)),
    }


def _make_bank() -> dict[str, Pipeline]:
    """Pipelines that take (n, B, c, t)."""
    fbcsp_feat = make_pipeline(Covariances("oas"), RiemCSP(nfilter=4, log=True))
    ts_feat = make_pipeline(Covariances("oas"), TangentSpace("riemann"))
    return {
        # Ang et al. FBCSP: per-band CSP -> mutual-information feature select -> clf
        "fbcsp_mibif_lda": Pipeline([
            ("fb", FilterBank(fbcsp_feat)),
            ("sel", SelectKBest(mutual_info_classif, k=8)),
            ("clf", LDA(solver="lsqr", shrinkage="auto"))]),
        "fbcsp_mibif_svm": Pipeline([
            ("fb", FilterBank(fbcsp_feat)),
            ("sel", SelectKBest(mutual_info_classif, k=8)),
            ("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0))]),
        "fbcsp_rf": Pipeline([
            ("fb", FilterBank(fbcsp_feat)),
            ("clf", RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1))]),

        # filter-bank Riemannian tangent space
        "fb_ts_lr": Pipeline([
            ("fb", FilterBank(ts_feat)), ("sc", StandardScaler()), ("clf", _lr(0.1))]),
        "fb_ts_lr_sel": Pipeline([
            ("fb", FilterBank(ts_feat)), ("sc", StandardScaler()),
            ("sel", SelectKBest(mutual_info_classif, k=40)), ("clf", _lr(0.3))]),
        "fb_logvar_lda": Pipeline([
            ("fb", FilterBank(LogVar())),
            ("clf", LDA(solver="lsqr", shrinkage="auto"))]),
    }


class CropEnsemble(BaseEstimator):
    """
    Cropped training (Schirrmeister et al., HBM 2017) for any base estimator.

    Each trial is cut into overlapping sub-windows.  Training sees every crop
    as its own labelled example, which regularises a model that would
    otherwise see only ~50 trials per class; at test time the crop
    probabilities of a trial are averaged back into one decision.

    Crops never cross the trial boundary, so a trial is wholly inside either
    the training or the test fold — no leakage. Works for (n, c, t) and
    (n, B, c, t) inputs.
    """

    def __init__(self, base, crop=512, stride=128):
        self.base = base
        self.crop = crop
        self.stride = stride

    def _cut(self, X):
        t = X.shape[-1]
        starts = list(range(0, max(t - self.crop, 0) + 1, self.stride))
        if not starts:
            starts = [0]
        crops = [X[..., s:s + self.crop] for s in starts]
        return np.concatenate(crops, axis=0), len(starts)

    def fit(self, X, y):
        Xc, k = self._cut(X)
        self.model_ = clone(self.base).fit(Xc, np.tile(y, k))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        Xc, k = self._cut(X)
        p = self.model_.predict_proba(Xc)
        return p.reshape(k, len(X), -1).mean(axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


PLAIN = _make_plain()
BANK = _make_bank()

# crop-augmented variants of the two strongest blocks (1 s hop, 2 s crops)
PLAIN["crop_ts_lr"] = CropEnsemble(
    make_pipeline(Covariances("oas"), TangentSpace("riemann"),
                  StandardScaler(), _lr(0.3)), crop=512, stride=256)
BANK["crop_fb_ts_lr"] = CropEnsemble(
    Pipeline([("fb", FilterBank(make_pipeline(Covariances("oas"),
                                              TangentSpace("riemann")))),
              ("sc", StandardScaler()), ("clf", _lr(0.1))]),
    crop=512, stride=256)

ALL = {**PLAIN, **BANK}


def get(name: str):
    return clone(ALL[name])


def is_bank(name: str) -> bool:
    return name in BANK
