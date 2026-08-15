# -*- coding: utf-8 -*-
"""
Stage 18 — does a PHYSIOLOGICALLY CONSTRAINED classifier beat a generic one?

Everything up to here used a generic pipeline: a fixed 8-30 Hz filter bank
over all eight electrodes, with the spatial and spectral weighting left to
the learner.  That is well motivated for oscillatory MI, but it is not a
physiologically *constrained* model — nothing in it knows that the foot
representation is medial, or where this particular participant's sensorimotor
rhythm sits.

Five constraints are added here, and — critically — a control that uses the
WRONG physiology, so that "the constraint helps" can be distinguished from
"any constraint helps".

  1  INDIVIDUAL SMR FREQUENCY.  Each participant's mu and beta peaks are
     located from their own spectrum at central electrodes and the bands are
     centred there, instead of using textbook 8-13 / 13-30 for everyone.
     Estimated LABEL-FREE (spectral peak, not a class contrast), so it is
     leakage-proof by construction rather than by careful fold handling.

  2  FOOT-AREA LAPLACIAN.  Cz - mean(C3, C4, CPz) and the corresponding
     derivations at CPz, C3, C4.  A surface Laplacian is a high-pass spatial
     filter that sharpens focal radial sources and suppresses the spatially
     broad components that Sections III-E and III-F identified as arousal and
     muscle.  This montage has no FCz, so the Cz ring is three-sided.

  3  MEDIAL-CONSTRAINED SPATIAL FILTERING.  CSP restricted to the
     Laplacian-derived central channels, i.e. the spatial filter cannot use a
     lateral or global pattern even if one happens to separate the classes.

  4  EXPLICIT ERD FEATURES.  Log power at Cz and CPz in the participant's own
     mu and beta bands, as named interpretable features rather than latent
     covariance entries.

  5  CONSTRAINED vs UNCONSTRAINED, head to head, same folds.

  C  WRONG-PHYSIOLOGY CONTROL.  Identical machinery, but the Laplacian ring
     is built around the LATERAL frontal electrodes (FC3/FC4) instead of the
     foot area.  Lower-limb MI has no reason to be lateral-frontal.  If the
     foot-area model beats the generic one but the lateral model does too,
     then the gain is from spatial high-pass filtering in general and not
     from the anatomy, and the "physiologically grounded" claim is not
     earned.

Output: results/physio_classifier.csv, fig_physio_classifier.png
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ttest_rel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from pyriemann.estimation import Covariances          # noqa: E402
from pyriemann.spatialfilters import CSP as RiemCSP   # noqa: E402
from pyriemann.tangentspace import TangentSpace       # noqa: E402

import steam_io as sio                                # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
TMIN, TMAX = 0.5, 4.5

# Laplacian rings.  Each entry: centre -> its neighbours in THIS montage.
FOOT_RING = {"Cz": ["C3", "C4", "CPz"], "CPz": ["CP3", "CP4", "Cz"],
             "C3": ["FC3", "Cz", "CP3"], "C4": ["FC4", "Cz", "CP4"]}
# Wrong-physiology control: lateral frontal, where lower-limb MI has no reason
# to live.
LAT_RING = {"FC3": ["C3", "CP3", "FC4"], "FC4": ["C4", "CP4", "FC3"],
            "CP3": ["C3", "FC3", "CPz"], "CP4": ["C4", "FC4", "CPz"]}

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GREEN, GREY = "#1f6feb", "#bc4c00", "#116329", "#9aa4b2"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 170,
})


# ------------------------- physiological front ends -------------------------
def laplacian_matrix(ring):
    """(n_out, 8) matrix implementing centre minus mean-of-neighbours."""
    W = np.zeros((len(ring), len(CH)))
    for i, (c, nb) in enumerate(ring.items()):
        W[i, CH.index(c)] = 1.0
        for n in nb:
            W[i, CH.index(n)] = -1.0 / len(nb)
    return W


def individual_peaks(X_broad, fs):
    """
    Participant's own mu and beta peak frequencies, from the central-electrode
    spectrum.  Label-free: this is where the rhythm IS, not where it differs
    between classes, so no fold handling is needed.

    Returns ((mu_lo, mu_hi), (beta_lo, beta_hi)).
    """
    ci = [CH.index(c) for c in ("Cz", "CPz", "C3", "C4")]
    f, P = welch(X_broad[:, ci], fs=fs, nperseg=int(fs * 2), axis=-1)
    spec = np.log(P.mean(axis=(0, 1)) + 1e-30)
    # remove the 1/f trend so the peak is the rhythm, not the low-frequency edge
    keep = (f >= 4) & (f <= 35)
    ff, ss = f[keep], spec[keep]
    trend = np.polyval(np.polyfit(np.log(ff), ss, 1), np.log(ff))
    resid = ss - trend
    mu_band = (ff >= 6) & (ff <= 15)
    be_band = (ff >= 15) & (ff <= 30)
    mu_pk = float(ff[mu_band][resid[mu_band].argmax()])
    be_pk = float(ff[be_band][resid[be_band].argmax()])
    return (mu_pk - 2.0, mu_pk + 2.0), (be_pk - 4.0, be_pk + 4.0), mu_pk, be_pk


class SpatialFilter(BaseEstimator, TransformerMixin):
    """Fixed linear spatial filter (identity, or a Laplacian ring)."""

    def __init__(self, W=None):
        self.W = W

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X if self.W is None else np.einsum("oc,nct->not", self.W, X)


class BandLogPower(BaseEstimator, TransformerMixin):
    """Explicit, named log band-power features — one per channel per band."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X: (n, n_band, n_ch, n_time)
        return np.log(X.var(axis=-1) + 1e-30).reshape(len(X), -1)


class BankTS(BaseEstimator, TransformerMixin):
    """Per-band covariance -> tangent space, concatenated."""

    def fit(self, X, y=None):
        self.p_ = [make_pipeline(Covariances("oas"), TangentSpace("riemann"))
                   .fit(X[:, b]) for b in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.concatenate([p.transform(X[:, b])
                               for b, p in enumerate(self.p_)], axis=1)


class BankCSP(BaseEstimator, TransformerMixin):
    def __init__(self, nfilter=2):
        self.nfilter = nfilter

    def fit(self, X, y=None):
        self.p_ = []
        for b in range(X.shape[1]):
            k = min(self.nfilter, X.shape[2])
            self.p_.append(make_pipeline(Covariances("oas"),
                                         RiemCSP(nfilter=k, log=True))
                           .fit(X[:, b], y))
        return self

    def transform(self, X):
        return np.concatenate([p.transform(X[:, b])
                               for b, p in enumerate(self.p_)], axis=1)


# ------------------------------- experiment --------------------------------
def build(date, subj, bands, W):
    """(n_trials, n_band, n_out_ch, n_time) after the fixed spatial filter."""
    _, Xb, y = sio.get_task_arrays(date, subj, lo=8.0, hi=30.0,
                                   tmin=TMIN, tmax=TMAX, ref="none",
                                   bands=tuple(bands))
    if W is not None:
        Xb = np.einsum("oc,nbct->nbot", W, Xb)
    return Xb, y


GENERIC_BANDS = tuple((lo, lo + 4) for lo in range(8, 27, 2))   # 8-30, 2 Hz hop
LR = lambda C=0.3: LogisticRegression(max_iter=5000, C=C)


def main():
    subs = sio.list_subjects()
    tasks = {"SvW": (1, 2), "RvS": (0, 1), "RvW": (0, 2)}
    W_foot = laplacian_matrix(FOOT_RING)
    W_lat = laplacian_matrix(LAT_RING)
    rows = []
    peaks = []

    for date, subj in subs:
        ep = sio.get_epochs(date, subj, l_freq=1.0, h_freq=45.0,
                            tmin=TMIN, tmax=TMAX, ref="none")
        (mu, be, mu_pk, be_pk) = individual_peaks(ep["X"], ep["fs"])
        peaks.append(dict(subject=f"{date}/{subj}", mu_peak=mu_pk, beta_peak=be_pk))
        ind_bands = (mu, be)

        # five feature pipelines, all scored on identical folds
        variants = {
            "A generic (current)":        (GENERIC_BANDS, None,  "ts"),
            "B + foot Laplacian":         (GENERIC_BANDS, W_foot, "ts"),
            "C + individual SMR bands":   (ind_bands,     None,  "ts"),
            "D full physio-constrained":  (ind_bands,     W_foot, "mix"),
            "E WRONG physio (control)":   (ind_bands,     W_lat,  "mix"),
        }
        cache = {}
        for name, (bands, W, kind) in variants.items():
            key = (tuple(bands), None if W is None else name[0])
            if key not in cache:
                cache[key] = build(date, subj, bands, W)
            Xb, y = cache[key]

            for task, (ia, ib) in tasks.items():
                m = np.isin(y, [ia, ib])
                yy = (y[m] == ib).astype(int)
                Xt = Xb[m]
                if kind == "ts":
                    clf = make_pipeline(BankTS(), StandardScaler(), LR(0.1))
                else:
                    # constrained model: medial CSP + explicit named band power
                    from sklearn.pipeline import FeatureUnion
                    clf = make_pipeline(
                        FeatureUnion([("csp", BankCSP(2)),
                                      ("pow", BandLogPower())]),
                        StandardScaler(),
                        LDA(solver="lsqr", shrinkage="auto"))
                acc = cross_val_score(
                    clf, Xt, yy,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    n_jobs=1).mean()
                rows.append(dict(subject=f"{date}/{subj}", task=task,
                                 model=name, acc=float(acc),
                                 n_feat_ch=Xt.shape[2], n_band=Xt.shape[1]))
        print(f"  {date}/{subj:4s} mu peak {mu_pk:4.1f} Hz, beta peak "
              f"{be_pk:4.1f} Hz", flush=True)

    df = pd.DataFrame(rows)
    pd.DataFrame(peaks).to_csv(os.path.join(R_DIR, "individual_peaks.csv"),
                               index=False)
    df.to_csv(os.path.join(R_DIR, "physio_classifier.csv"), index=False)

    order = list(dict.fromkeys(df.model))
    print("\n" + "=" * 82)
    print("PHYSIOLOGICALLY CONSTRAINED vs GENERIC   (5-fold, same folds)")
    print("=" * 82)
    for task in ("RvS", "RvW", "SvW"):
        d = df[df.task == task]
        base = d[d.model == order[0]].set_index("subject")["acc"]
        print(f"\n  {task}")
        for m in order:
            v = d[d.model == m].set_index("subject")["acc"]
            if m == order[0]:
                print(f"    {m:28s} {v.mean()*100:5.1f} %")
                continue
            t, p = ttest_rel(v, base.reindex(v.index))
            diff = (v - base.reindex(v.index)).mean() * 100
            mark = "  *" if p < .05 else ""
            print(f"    {m:28s} {v.mean()*100:5.1f} %   vs A {diff:+5.1f} pt "
                  f"(p={p:.3f}){mark}")

    print("\n  --- the control that decides whether 'physiological' is earned ---")
    for task in ("RvS", "RvW", "SvW"):
        d = df[df.task == task]
        foot = d[d.model == order[3]].set_index("subject")["acc"]
        lat = d[d.model == order[4]].set_index("subject")["acc"]
        t, p = ttest_rel(foot, lat)
        print(f"    {task:4s} foot-area {foot.mean()*100:5.1f} %  vs  "
              f"lateral {lat.mean()*100:5.1f} %   diff {(foot-lat).mean()*100:+5.1f} pt"
              f"  p={p:.3f}"
              f"{'   <-- anatomy matters' if p < .05 and foot.mean() > lat.mean() else ''}")

    # ------------------------------- figure -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), constrained_layout=True,
                             sharey=True)
    cols = {order[0]: GREY, order[1]: BLUE, order[2]: BLUE,
            order[3]: GREEN, order[4]: ORANGE}
    for ax, task in zip(axes, ("RvS", "RvW", "SvW")):
        d = df[df.task == task]
        chance = 50
        for i, m in enumerate(order):
            v = d[d.model == m]["acc"].values * 100
            ax.bar(i, v.mean(), 0.62, color=cols[m], zorder=2)
            ax.scatter(np.full(len(v), i), v, s=11, color="white",
                       edgecolor=INK, linewidth=0.5, zorder=3)
        ax.axhline(chance, color=INK, lw=1.2, ls=":")
        ax.set_xticks(range(len(order)),
                      ["A\ngeneric", "B\n+Lap", "C\n+ind band",
                       "D\nphysio", "E\nWRONG"], fontsize=7.6)
        ax.set_title({"RvS": "Rest vs Stand", "RvW": "Rest vs Walk",
                      "SvW": "Stand vs Walk"}[task], loc="left", fontsize=10)
    axes[0].set_ylabel("5-fold accuracy (%)")
    fig.suptitle("Physiologically constrained classifier vs generic pipeline — "
                 "E is the same machinery pointed at the wrong anatomy",
                 ha="left", x=0.008, fontsize=11.5, fontweight="bold")
    fig.savefig(os.path.join(R_DIR, "fig_physio_classifier.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_physio_classifier.png"))


if __name__ == "__main__":
    main()
