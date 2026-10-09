"""Exploratory eight-channel temporal, covariance and transfer reanalysis.

Continuous caches and derived feature arrays stay local. Every learned transform
is fitted within its training fold. Fixed per-trial feature extraction uses no
labels; model selection is nested. This is not independent new-data validation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[2]
# All generated signals stay in the ignored cache, separate from public results.
HERE = Path(os.environ.get("EEG_RECHECK_WORK", REPO_ROOT / "cache" / "stand_walk_recheck"))
CACHE_DIR = Path(os.environ.get("EEG_CACHE_DIR", REPO_ROOT / "cache"))
sys.path.insert(0, str(SOURCE_DIR.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import t, ttest_1samp
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import oas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import pyriemann
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace

from reanalyse_cached_eeg import CODE_TO_Y, CUE_CODE, keep_position, participant_files

CHANNELS = ("FC3", "FC4", "C3", "C4", "Cz", "CP3", "CP4", "CPz")
CLASS_NAMES = ("Rest", "Stand", "Walk")
BANDS = ((4, 8), (8, 13), (13, 20), (20, 30))
BAND_NAMES = ("theta", "mu", "low_beta", "high_beta")
METRICS = ("mean", "sd", "range", "best_split", "crossings", "modulation",
           "half_abs", "mean_abs_change")
CURVES = CHANNELS + ("LI_FC", "LI_C", "LI_CP", "LI_pooled", "mid_lateral", "front_back")
PAIR_CONTRASTS = ((1, 2, "Walk_minus_Stand"), (0, 1, "Stand_minus_Rest"),
                  (0, 2, "Walk_minus_Rest"))
FS = 256
SEED = 20261009


def logvar(x):
    return 10 * np.log10(np.var(x, axis=-1) + 1e-30)


def temporal_descriptors(curves):
    """Input (..., time), fixed phase/sign tolerant summaries except mean."""
    n = curves.shape[-1]
    cumulative = np.cumsum(curves, axis=-1)
    tau = np.arange(3, n - 2)
    delta = (cumulative[..., tau - 1] / tau
             - (cumulative[..., -1, None] - cumulative[..., tau - 1]) / (n - tau))
    centered = curves - curves.mean(axis=-1, keepdims=True)
    crossings = np.sum(np.diff(centered > 0, axis=-1), axis=-1)
    freq, psd = welch(centered, fs=4, nperseg=n, axis=-1)
    modulation = np.log10(psd[..., (freq >= .2) & (freq <= 1)].mean(axis=-1) + 1e-12)
    return np.stack((curves.mean(axis=-1), curves.std(axis=-1),
                     np.ptp(curves, axis=-1), np.max(np.abs(delta), axis=-1),
                     crossings, modulation,
                     np.abs(curves[..., :n//2].mean(axis=-1)
                            - curves[..., n//2:].mean(axis=-1)),
                     np.mean(np.abs(np.diff(curves, axis=-1)), axis=-1)), axis=-1)


def make_curves(power):
    """Add log-power contrasts; pooled LI uses geometric mean band power."""
    extra = [power[..., r, :] - power[..., l, :] for l, r in ((0, 1), (2, 3), (5, 6))]
    extra.append(power[..., [1, 3, 6], :].mean(axis=-2)
                 - power[..., [0, 2, 5], :].mean(axis=-2))
    extra.append(power[..., [4, 7], :].mean(axis=-2)
                 - power[..., [0, 1, 2, 3, 5, 6], :].mean(axis=-2))
    extra.append(power[..., [0, 1], :].mean(axis=-2)
                 - power[..., [5, 6, 7], :].mean(axis=-2))
    return np.concatenate((power, np.stack(extra, axis=-2)), axis=-2)


def covariances(epochs):
    return np.asarray([oas(epoch.T, assume_centered=False)[0] for epoch in epochs])


def extract_participant(date, runs):
    collected = {}
    labels, run_ids, positions, quality = [], [], [], []
    for run_index, (tag, _, path) in enumerate(runs):
        with np.load(path, allow_pickle=False) as z:
            data = z["data"].astype(float)
            codes, samples = z["codes"], z["samples"]
            ns, fs = int(z["n_samples"]), float(z["fs"])
            clips = np.unpackbits(z["clip"], axis=1, count=ns)
            assert tuple(z["ch_names"]) == CHANNELS and fs == FS
        order = np.argsort(samples, kind="stable")
        codes, samples = codes[order], samples[order]
        selected = []
        invalid = clipped = kept = 0
        for number, pos in enumerate(np.flatnonzero(np.isin(codes, list(CODE_TO_Y))), 1):
            if not keep_position(date, tag, number):
                continue
            kept += 1
            sample, code = int(samples[pos]), int(codes[pos])
            if (pos < 2 or codes[pos - 1] != CUE_CODE[code]
                    or abs((sample - samples[pos - 1]) / FS - 2) > .3
                    or sample - 5*FS < 0 or sample + 5*FS > ns):
                invalid += 1
                continue
            if np.max(clips[:, sample:sample + 5*FS].mean(axis=1)) > .005:
                clipped += 1
                continue
            selected.append(sample)
            labels.append(CODE_TO_Y[code])
            run_ids.append(run_index)
            positions.append(number)
        quality.append(dict(run=tag, kept_by_rule=kept, invalid=invalid,
                            clipped=clipped, analysed=len(selected)))
        anchors = np.asarray(selected)
        for reference in ("ear", "car"):
            source = data if reference == "ear" else data - data.mean(axis=0, keepdims=True)
            task_bins, base_bins, trajectory, covs, mean_power = [], [], [], [], []
            for band in BANDS:
                filtered = sosfiltfilt(butter(4, band, btype="bandpass", fs=FS, output="sos"),
                                      source, axis=-1)
                task_bins.append(np.asarray([[logvar(filtered[:, a + j*FS:a + (j+1)*FS])
                                               for j in range(5)] for a in anchors]))
                base_bins.append(np.asarray([[logvar(filtered[:, a + int(start*FS):a + int((start+1)*FS)])
                                               for start in (-4.5, -3.5)] for a in anchors]))
                trajectory.append(np.asarray([[logvar(filtered[:, a + round(start*FS):
                                                               a + round((start+.5)*FS)])
                                                for start in np.arange(.25, 4.251, .25)]
                                               for a in anchors]))
                epochs = np.asarray([filtered[:, a + FS//2:a + int(4.5*FS)] for a in anchors])
                covs.append(covariances(epochs))
                mean_power.append(np.asarray([[logvar(filtered[:, a + round(start*FS):
                                                                a + round((start+1)*FS)])
                                                for start in (.5, 1.5, 2.5, 3.5)] for a in anchors]).mean(axis=1))
            # Shapes: trial x band x channel x time.
            task = np.transpose(np.asarray(task_bins), (1, 0, 3, 2))
            baseline = np.transpose(np.asarray(base_bins), (1, 0, 3, 2)).mean(axis=-1)
            trajectory = np.transpose(np.asarray(trajectory), (1, 0, 3, 2))
            curves = make_curves(trajectory)
            desc = temporal_descriptors(curves)
            features = {
                "power": np.transpose(np.asarray(mean_power), (1, 0, 2)).reshape(len(anchors), -1),
                "change": (task[..., 1:].mean(axis=-1) - baseline).reshape(len(anchors), -1),
                "baseline": baseline.reshape(len(anchors), -1),
                "trajectory": trajectory.reshape(len(anchors), -1),
                "dynamic": desc.reshape(len(anchors), -1),
                "cov": np.transpose(np.asarray(covs), (1, 0, 2, 3)),
                "task_change": task - baseline[..., None],
                "descriptors": desc,
                "curves": curves,
            }
            features["combined"] = np.concatenate((features["power"], features["change"],
                                                     features["dynamic"]), axis=1)
            # Deterministic per-trial/channel amplitude clipping is a sensitivity,
            # not validated EOG/EMG removal. It estimates nothing across trials.
            broad = sosfiltfilt(butter(4, (8, 30), btype="bandpass", fs=FS, output="sos"),
                                source, axis=-1)
            epochs = np.asarray([broad[:, a + FS//2:a + int(4.5*FS)] for a in anchors])
            center = np.median(epochs, axis=-1, keepdims=True)
            scale = 1.4826 * np.median(np.abs(epochs - center), axis=-1, keepdims=True) + 1e-15
            robust = np.clip(epochs, center - 6*scale, center + 6*scale)
            features["robust_cov"] = covariances(robust)[:, None]
            features["broad_cov"] = covariances(epochs)[:, None]
            features["winsor_fraction"] = np.mean(np.abs(epochs - center) > 6*scale, axis=(1, 2))
            for key, value in features.items():
                collected.setdefault(reference + "_" + key, []).append(value)
    result = {key: np.concatenate(parts) for key, parts in collected.items()}
    result.update(y=np.asarray(labels), run=np.asarray(run_ids), position=np.asarray(positions))
    assert np.array_equal(np.bincount(result["y"], minlength=3), [50, 50, 50])
    assert all(np.isfinite(value).all() for value in result.values())
    return result, quality


class PerBandTangent(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.maps_ = [TangentSpace(metric="riemann").fit(x[:, b]) for b in range(x.shape[1])]
        return self

    def transform(self, x):
        return np.concatenate([model.transform(x[:, b]) for b, model in enumerate(self.maps_)], axis=1)


class PerBandCSP(BaseEstimator, TransformerMixin):
    def fit(self, x, y):
        self.maps_ = [CSP(nfilter=4, log=True).fit(x[:, b], y) for b in range(x.shape[1])]
        return self

    def transform(self, x):
        return np.concatenate([model.transform(x[:, b]) for b, model in enumerate(self.maps_)], axis=1)


def candidates():
    out = []
    for view in ("power", "change", "trajectory", "dynamic", "combined"):
        for c in (.01, .1):
            out.append((f"ear_{view}_lr{c}", "ear_" + view,
                        make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))))
        from sklearn.svm import SVC
        out.append((f"ear_{view}_rbf", "ear_" + view,
                    make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf"))))
    for reference in ("ear", "car"):
        for view in ("cov", "robust_cov"):
            out.append((f"{reference}_{view}_ts", reference + "_" + view,
                        make_pipeline(PerBandTangent(), StandardScaler(),
                                      LogisticRegression(C=.1, max_iter=1000))))
        out.append((reference + "_fbcsp", reference + "_cov",
                    make_pipeline(PerBandCSP(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))))
    out.append(("car_combined_lr", "car_combined", make_pipeline(StandardScaler(),
                 LogisticRegression(C=.1, max_iter=1000))))
    out.append(("ear_combined_trees", "ear_combined",
                ExtraTreesClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=5,
                                     random_state=SEED, n_jobs=1)))
    return out


def splits(y, mode, n_splits=5, groups=None):
    if groups is not None:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    if mode == "random":
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    result = []
    for test in np.array_split(np.arange(len(y)), n_splits):
        forbidden = np.arange(max(0, test[0]-1), min(len(y), test[-1]+2))
        train = np.setdiff1d(np.arange(len(y)), forbidden)
        assert len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2
        result.append((train, test))
    return result


def nested_predict(data, y, outer, mode, groups=None, bank=None):
    bank = candidates() if bank is None else bank
    predictions = np.full((len(bank), len(y)), -1, int)
    chosen = np.full(len(y), -1, int)
    choices = []
    for fold, (train, test) in enumerate(outer):
        inner = splits(y[train], mode, 3, None if groups is None else groups[train])
        scores = []
        for ci, (name, view, estimator) in enumerate(bank):
            x = data[view]
            inner_prediction = np.full(len(train), -1, int)
            for tr, te in inner:
                model = clone(estimator).fit(x[train[tr]], y[train[tr]])
                inner_prediction[te] = model.predict(x[train[te]])
            score = balanced_accuracy_score(y[train], inner_prediction)
            scores.append(score)
            model = clone(estimator).fit(x[train], y[train])
            predictions[ci, test] = model.predict(x[test])
        best = int(np.argmax(scores))
        chosen[test] = predictions[best, test]
        choices.append(dict(fold=fold, selected=bank[best][0], inner_balanced=scores[best],
                            n_train=len(train), n_test=len(test)))
    assert (predictions >= 0).all() and (chosen >= 0).all()
    return chosen, predictions, choices


def summarize(frame, keys, value):
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value].to_numpy()
        half = t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))
        rows.append(dict(zip(keys, key), n=len(values), mean=values.mean(),
                         sd=values.std(ddof=1), ci_low=values.mean()-half, ci_high=values.mean()+half))
    return pd.DataFrame(rows)


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order])
    out = np.empty_like(pvalues)
    out[order] = np.minimum(adjusted, 1)
    return out


def signal_analysis(dataset, out):
    rows, effects = [], []
    for sid, data in dataset.items():
        y = data["y"]
        for ref in ("ear", "car"):
            desc = data[ref + "_descriptors"]
            change = data[ref + "_task_change"][..., 1:].mean(axis=-1)
            # Freeze the same split separately within each class in time order.
            for band, curve, metric in itertools.product(range(4), range(14), range(8)):
                for a, b, contrast in PAIR_CONTRASTS:
                    aa, bb = desc[y == a, band, curve, metric], desc[y == b, band, curve, metric]
                    rows.append(dict(subject=sid, reference=ref, band=BAND_NAMES[band],
                                     curve=CURVES[curve], metric=METRICS[metric], contrast=contrast,
                                     difference=bb.mean()-aa.mean(),
                                     first_half=bb[:25].mean()-aa[:25].mean(),
                                     second_half=bb[25:].mean()-aa[25:].mean()))
            for band, channel in itertools.product(range(4), range(8)):
                for a, b, contrast in PAIR_CONTRASTS:
                    rows.append(dict(subject=sid, reference=ref, band=BAND_NAMES[band],
                                     curve=CHANNELS[channel], metric="baseline_change", contrast=contrast,
                                     difference=change[y == b, band, channel].mean()-change[y == a, band, channel].mean(),
                                     first_half=np.nan, second_half=np.nan))
            for cls in range(3):
                for band, channel in itertools.product(range(4), range(8)):
                    effects.append(dict(subject=sid, reference=ref, condition=CLASS_NAMES[cls],
                                        band=BAND_NAMES[band], channel=CHANNELS[channel],
                                        change_db=change[y == cls, band, channel].mean()))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "signal_participant_differences.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "within_condition_change.csv", index=False)
    keys = ["reference", "band", "curve", "metric", "contrast"]
    summary = summarize(frame, keys, "difference")
    extras = []
    for key, group in frame.groupby(keys, sort=False):
        values = group["difference"].to_numpy()
        p = ttest_1samp(values, 0).pvalue if np.std(values) > 0 else 1.
        signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
        permutation = np.abs((signs * values).mean(axis=1))
        exact_p = np.mean(permutation >= abs(values.mean()) - 1e-12)
        extras.append(dict(zip(keys, key), p_raw=p, p_signflip=exact_p,
                           dz=values.mean()/values.std(ddof=1) if np.std(values) > 0 else 0,
                           positive=int(np.sum(values > 0)),
                           halves_same_positive=int(np.sum((group.first_half > 0) & (group.second_half > 0)))))
    summary = summary.merge(pd.DataFrame(extras), on=keys, validate="one_to_one")
    summary["p_holm"] = np.nan
    summary["p_signflip_holm"] = np.nan
    for _, indices in summary.groupby(["reference", "contrast"]).groups.items():
        summary.loc[indices, "p_holm"] = holm(summary.loc[indices, "p_raw"])
        summary.loc[indices, "p_signflip_holm"] = holm(summary.loc[indices, "p_signflip"])
    summary.to_csv(out / "signal_summary.csv", index=False)


def within_person(dataset, out):
    rows, fixed_rows, choices, predictions = [], [], [], []
    bank = candidates()
    for sid, data in dataset.items():
        take = data["y"] > 0
        y = data["y"][take] - 1
        local = {key: value[take] for key, value in data.items()}
        for mode in ("random", "contiguous"):
            started = time.monotonic()
            chosen, fixed, selected = nested_predict(local, y, splits(y, mode), mode)
            rows.append(dict(subject=sid, evaluation=mode, model="nested_selection",
                             accuracy=accuracy_score(y, chosen), balanced=balanced_accuracy_score(y, chosen)))
            for ci, (name, _, _) in enumerate(bank):
                fixed_rows.append(dict(subject=sid, evaluation=mode, model=name,
                                       accuracy=accuracy_score(y, fixed[ci]),
                                       balanced=balanced_accuracy_score(y, fixed[ci])))
            for choice in selected:
                choices.append(dict(subject=sid, evaluation=mode, **choice))
            for i, actual in enumerate(y):
                predictions.append(dict(subject=sid, evaluation=mode, trial=i, actual=int(actual),
                                        prediction=int(chosen[i])))
            # Baseline control: fixed classifier, no model search.
            control = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
            pred = np.full(len(y), -1)
            for train, test in splits(y, mode):
                pred[test] = clone(control).fit(local["ear_baseline"][train], y[train]).predict(local["ear_baseline"][test])
            rows.append(dict(subject=sid, evaluation=mode, model="pre_cue_fixed_control",
                             accuracy=accuracy_score(y, pred), balanced=balanced_accuracy_score(y, pred)))
            pd.DataFrame(rows).to_csv(out / "within_person_scores.csv", index=False)
            pd.DataFrame(fixed_rows).to_csv(out / "fixed_candidate_scores.csv", index=False)
            pd.DataFrame(choices).to_csv(out / "nested_choices.csv", index=False)
            pd.DataFrame(predictions).to_csv(out / "within_person_predictions.csv", index=False)
            print(f"{sid} {mode}: nested={rows[-2]['accuracy']:.3f}; {time.monotonic()-started:.1f}s", flush=True)
    summarize(pd.DataFrame(rows), ["evaluation", "model"], "accuracy").to_csv(out / "within_person_summary.csv", index=False)
    summarize(pd.DataFrame(fixed_rows), ["evaluation", "model"], "accuracy").to_csv(out / "fixed_candidate_summary.csv", index=False)


def cross_person(dataset, out):
    # Group-exclusive selection among a bounded subset; no test-person calibration.
    bank = [candidate for candidate in candidates()
            if candidate[0] in ("ear_power_lr0.1", "ear_change_lr0.1", "ear_dynamic_lr0.01",
                               "ear_combined_lr0.01", "ear_power_rbf", "ear_change_rbf",
                               "ear_cov_ts", "car_cov_ts", "ear_fbcsp")]
    subjects = list(dataset)
    keys = sorted({view for _, view, _ in bank})
    full = {key: np.concatenate([dataset[sid][key][dataset[sid]["y"] > 0] for sid in subjects]) for key in keys}
    y = np.concatenate([dataset[sid]["y"][dataset[sid]["y"] > 0] - 1 for sid in subjects])
    groups = np.repeat(np.arange(10), 100)
    rows, choice_rows = [], []
    for group, sid in enumerate(subjects):
        train, test = np.flatnonzero(groups != group), np.flatnonzero(groups == group)
        inner = splits(y[train], "random", 3, groups[train])
        scores = []
        for name, view, estimator in bank:
            fold_scores = []
            for tr, te in inner:
                model = clone(estimator).fit(full[view][train[tr]], y[train[tr]])
                pred = model.predict(full[view][train[te]])
                # Participant-weighted scores, not trial-weighted when sizes vary.
                fold_scores.extend(accuracy_score(y[train[te]][groups[train[te]] == g],
                                                  pred[groups[train[te]] == g])
                                   for g in np.unique(groups[train[te]]))
            scores.append(np.mean(fold_scores))
        best = int(np.argmax(scores))
        name, view, estimator = bank[best]
        model = clone(estimator).fit(full[view][train], y[train])
        pred = model.predict(full[view][test])
        rows.append(dict(subject=sid, evaluation="LOSO", model="nested_group_selection",
                         accuracy=accuracy_score(y[test], pred), balanced=balanced_accuracy_score(y[test], pred)))
        choice_rows.append(dict(subject=sid, selected=name, inner_accuracy=scores[best]))
        pd.DataFrame(rows).to_csv(out / "cross_person_scores.csv", index=False)
        pd.DataFrame(choice_rows).to_csv(out / "cross_person_choices.csv", index=False)
        print(f"{sid} LOSO: {rows[-1]['accuracy']:.3f} selected={name}", flush=True)
    summarize(pd.DataFrame(rows), ["evaluation", "model"], "accuracy").to_csv(out / "cross_person_summary.csv", index=False)


def figures(dataset, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    times = np.arange(.5, 4.501, .25)
    for row, band in enumerate((1, 2)):
        for col, curve in enumerate(("LI_C", "LI_CP", "mid_lateral")):
            ax = axes[row, col]
            for cls, color in ((1, "#008478"), (2, "#bd3556")):
                values = np.asarray([data["ear_curves"][data["y"] == cls, band, CURVES.index(curve)].mean(axis=0)
                                     for data in dataset.values()])
                mean = values.mean(axis=0)
                se = values.std(axis=0, ddof=1) / np.sqrt(10)
                ax.plot(times, mean, label=CLASS_NAMES[cls], color=color)
                ax.fill_between(times, mean-se, mean+se, alpha=.15, color=color)
            ax.axhline(0, color=".7", linewidth=.8)
            ax.set(title=BAND_NAMES[band] + " / " + curve, xlabel="Task time (s)", ylabel="Log-power contrast (dB)")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.savefig(out / "time_laterality.png", dpi=160)
    plt.close(fig)
    if (out / "within_person_scores.csv").exists():
        frame = pd.read_csv(out / "within_person_scores.csv")
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        for mode, color, offset in (("random", "#008478", -.08), ("contiguous", "#bd3556", .08)):
            subset = frame[(frame.model == "nested_selection") & (frame.evaluation == mode)]
            ax.scatter(np.arange(len(subset)) + offset, 100*subset.accuracy, color=color, label=mode)
        ax.axhline(50, color=".5", linestyle="--")
        ax.set(xticks=np.arange(10), xticklabels=list(dataset), ylabel="Held-out accuracy (%)", ylim=(25, 85))
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(out / "held_out_accuracy.png", dpi=160)
        plt.close(fig)


def self_test():
    rng = np.random.default_rng(SEED)
    n = 17
    stationary = rng.normal(0, .08, (100, n))
    switched = stationary.copy()
    for i in range(100):
        tau = rng.integers(4, 13)
        switched[i, :tau] += 1 if i % 2 else -1
        switched[i, tau:] -= 1 if i % 2 else -1
    repeated = stationary + np.sin(np.arange(n)[None] * 2*np.pi/5 + rng.uniform(0, 2*np.pi, (100, 1)))
    a, b, c = map(temporal_descriptors, (stationary, switched, repeated))
    assert b[:, METRICS.index("best_split")].mean() > a[:, METRICS.index("best_split")].mean() + 1
    assert c[:, METRICS.index("modulation")].mean() > a[:, METRICS.index("modulation")].mean() + 1
    y = np.tile([0, 1], 50)
    for train, test in splits(y, "contiguous"):
        assert not set(train) & set(test)
        assert np.min(np.abs(train[:, None] - test[None, :])) >= 2
    # A test-only perturbation cannot change fitted training tangent references.
    x = rng.normal(size=(20, 8, 150))
    cov = covariances(x)[:, None]
    model = PerBandTangent().fit(cov[:10])
    before = model.maps_[0].reference_.copy()
    model.transform(cov[10:] * 100)
    assert np.array_equal(before, model.maps_[0].reference_)
    return dict(variable_switch_detected=True, phase_free_repetition_detected=True,
                purged_block_splits_disjoint=True, tangent_test_transform_does_not_refit=True,
                simulation_scope="feature sensitivity, not sensitivity to real imagined steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--stage", choices=("extract", "within", "cross", "figures", "all", "test"), default="all")
    args = parser.parse_args()
    out, private = HERE / "results", HERE / "private_features"
    out.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    tests = self_test()
    (out / "self_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    if args.stage == "test":
        print(tests)
        return
    dataset, quality = {}, []
    for sid, date, runs in participant_files(args.cache):
        path = private / (sid + ".npz")
        if args.stage in ("extract", "all") or not path.exists():
            data, checks = extract_participant(date, runs)
            np.savez_compressed(path, **data)
            quality.extend(dict(subject=sid, **row) for row in checks)
            print(sid + " extracted: " + str(np.bincount(data["y"]).tolist()), flush=True)
        else:
            with np.load(path, allow_pickle=False) as z:
                data = {key: z[key] for key in z.files}
        dataset[sid] = data
    if quality:
        pd.DataFrame(quality).to_csv(out / "quality.csv", index=False)
    if args.stage in ("extract", "all"):
        signal_analysis(dataset, out)
    if args.stage in ("within", "all"):
        within_person(dataset, out)
    if args.stage in ("cross", "all"):
        cross_person(dataset, out)
    figures(dataset, out)
    manifest = dict(seed=SEED, numpy=np.__version__, scipy=scipy.__version__,
                    sklearn=sklearn.__version__, pyriemann=pyriemann.__version__,
                    channels=CHANNELS, bands=BANDS, candidates=[name for name, _, _ in candidates()],
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    evaluation="exploratory; same previously inspected dataset", stage=args.stage)
    (out / ("manifest_" + args.stage + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
