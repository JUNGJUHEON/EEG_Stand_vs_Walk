"""Within-condition EEG modulation and repeated-half stability checks.

All comparisons use equal-length 1 s windows. See the repository README.
The output describes recorded EEG, not an identified motor-cortical source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import t

from reanalyse_cached_eeg import (CLASSES, CODE_TO_Y, CUE_CODE, keep_position,
                                  participant_files)


BANDS = {"mu_8_13": (8, 13), "beta_13_30": (13, 30)}
ROIS = {"midline": ("Cz", "CPz"), "lateral": ("C3", "C4")}
WINDOWS = {"base1": (-4.5, -3.5), "base2": (-3.5, -2.5),
           "cue1": (-2, -1), "task1": (0, 1), "task2": (1, 2),
           "task3": (2, 3), "task4": (3, 4), "task5": (4, 5)}
MEASURE = tuple(k for k in WINDOWS if k not in ("base1", "base2"))
SCALE = 10 / np.log(10)


def collect_participant(participant: str, date: str, runs: list) -> tuple[list[dict], list[dict], dict]:
    rows, quality, baseline_spectra = [], [], []
    for run_index, (tag, _, path) in enumerate(runs):
        with np.load(path, allow_pickle=False) as z:
            data = z["data"].astype(np.float64)
            fs = float(z["fs"])
            codes, samples = z["codes"], z["samples"]
            n_samples = int(z["n_samples"])
            clip = np.unpackbits(z["clip"], axis=1, count=n_samples).astype(bool)
            channels = [str(c) for c in z["ch_names"]]
        if fs != 256 or data.shape != (8, n_samples):
            raise ValueError(f"Unexpected cache geometry: {path.name}")
        roi_idx = {name: [channels.index(ch) for ch in chs]
                   for name, chs in ROIS.items()}
        order = np.argsort(samples, kind="stable")
        codes, samples = codes[order], samples[order]
        positions = np.flatnonzero(np.isin(codes, tuple(CODE_TO_Y)))
        filtered = {name: sosfiltfilt(
            butter(4, edges, btype="bandpass", fs=fs, output="sos"), data, axis=1)
            for name, edges in BANDS.items()}
        accepted, excluded = 0, 0
        for trial, pos in enumerate(positions, start=1):
            if not keep_position(date, tag, trial):
                continue
            code, sample = int(codes[pos]), int(samples[pos])
            if (pos < 2 or int(codes[pos - 1]) != CUE_CODE[code]
                    or int(codes[pos - 2]) != 0
                    or abs((sample - int(samples[pos - 1])) / fs - 2) > 0.3
                    or abs((sample - int(samples[pos - 2])) / fs - 5) > 0.3):
                excluded += 1
                continue
            slices = {key: slice(sample + round(start * fs),
                                 sample + round(stop * fs))
                      for key, (start, stop) in WINDOWS.items()}
            if (any(sl.start < 0 or sl.stop > n_samples for sl in slices.values())
                    or any(np.max(clip[:, sl].mean(axis=1)) > 0.005
                           for sl in slices.values())):
                excluded += 1
                continue
            accepted += 1
            condition = CLASSES[CODE_TO_Y[code]]
            fixation = slice(slices["base1"].start, slices["base2"].stop)
            freqs, spectrum = welch(data[roi_idx["midline"], fixation],
                                    fs=fs, nperseg=512, axis=-1)
            baseline_spectra.append(spectrum.mean(axis=0))
            for band, original in filtered.items():
                for reference in ("earlobe", "car"):
                    signal = (original if reference == "earlobe" else
                              original - original.mean(axis=0, keepdims=True))
                    for roi, indices in roi_idx.items():
                        powers = {key: np.log(np.var(signal[indices, sl], axis=1)
                                                   + 1e-30).mean()
                                  for key, sl in slices.items()}
                        baseline = (powers["base1"] + powers["base2"]) / 2
                        for window in MEASURE:
                            rows.append({"participant": participant, "run_index": run_index,
                                         "trial_in_run": trial, "condition": condition,
                                         "band": band, "roi": roi, "reference": reference,
                                         "window": window,
                                         "change_db": float((powers[window] - baseline) * SCALE)})
        quality.append({"participant": participant, "run": tag,
                        "retained": accepted, "excluded": excluded})

    if not baseline_spectra:
        raise ValueError(f"No fixation spectra for {participant}")
    mean_spectrum = np.mean(baseline_spectra, axis=0)
    spectrum_db = 10 * np.log10(mean_spectrum + 1e-30)
    low = (freqs >= 7) & (freqs <= 8)
    high = (freqs >= 14) & (freqs <= 15)
    mu = (freqs >= 8) & (freqs <= 13)
    low_freq, high_freq = freqs[low].mean(), freqs[high].mean()
    low_db, high_db = spectrum_db[low].mean(), spectrum_db[high].mean()
    background = np.interp(freqs[mu], [low_freq, high_freq], [low_db, high_db])
    prominence = spectrum_db[mu] - background
    peak_at = int(np.argmax(prominence))
    peak = {"participant": participant, "n_fixation_windows": len(baseline_spectra),
            "peak_hz": float(freqs[mu][peak_at]),
            "peak_excess_db": float(prominence[peak_at]),
            "mean_mu_excess_db": float(prominence.mean())}
    return rows, quality, peak


def summarize(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial_ids = (trials[["participant", "condition", "run_index", "trial_in_run"]]
                 .drop_duplicates().sort_values(["participant", "condition",
                                                 "run_index", "trial_in_run"]))
    trial_ids["within_condition_order"] = trial_ids.groupby(
        ["participant", "condition"]).cumcount()
    counts = trial_ids.groupby(["participant", "condition"]).size()
    if not (counts == 50).all():
        raise ValueError(f"Expected 50 trials per class: {counts.to_dict()}")
    trial_ids["half"] = np.where(trial_ids.within_condition_order < 25,
                                 "first", "last")
    trials = trials.merge(trial_ids, on=["participant", "condition",
                                        "run_index", "trial_in_run"], validate="many_to_one")
    keys = ["participant", "condition", "band", "roi", "reference", "window"]
    participant = (trials.groupby(keys).change_db
                   .agg(mean_db="mean", median_trial_db="median", sd_trial_db="std")
                   .reset_index())
    halves = (trials.groupby(keys + ["half"]).change_db.mean()
              .unstack("half").reset_index())
    participant = participant.merge(halves, on=keys, validate="one_to_one")
    participant["same_sign_halves"] = (np.sign(participant["first"])
                                       == np.sign(participant["last"]))

    group = []
    for (condition, band, roi, reference, window), sub in participant.groupby(keys[1:]):
        values = sub.mean_db.to_numpy()
        width = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        group.append({"condition": condition, "band": band, "roi": roi,
                      "reference": reference,
                      "window": window, "n_participants": len(values),
                      "mean_db": float(values.mean()),
                      "median_participant_db": float(np.median(values)),
                      "ci95_low_db": float(values.mean() - width),
                      "ci95_high_db": float(values.mean() + width),
                      "n_negative": int((values < 0).sum()),
                      "n_same_sign_halves": int(sub.same_sign_halves.sum())})

    contrasts = []
    for (participant_id, band, roi, reference, window), sub in participant.groupby(
            ["participant", "band", "roi", "reference", "window"]):
        by_condition = sub.set_index("condition")
        for condition in ("Stand", "Walk"):
            contrasts.append({"participant": participant_id,
                              "condition_minus_rest": condition, "band": band,
                              "roi": roi, "reference": reference, "window": window,
                              "delta_db": float(by_condition.loc[condition, "mean_db"]
                                                - by_condition.loc["Rest", "mean_db"]),
                              "first_delta_db": float(by_condition.loc[condition, "first"]
                                                      - by_condition.loc["Rest", "first"]),
                              "last_delta_db": float(by_condition.loc[condition, "last"]
                                                     - by_condition.loc["Rest", "last"])})
    return participant, pd.DataFrame(group), pd.DataFrame(contrasts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, quality, peaks = [], [], []
    for participant, date, runs in participant_files(args.cache):
        trial_rows, quality_rows, peak = collect_participant(participant, date, runs)
        rows.extend(trial_rows)
        quality.extend(quality_rows)
        peaks.append(peak)
        print(participant, "retained", sum(q["retained"] for q in quality_rows), flush=True)
    participant, group, contrasts = summarize(pd.DataFrame(rows))
    participant.to_csv(args.out / "within_class_participant.csv", index=False)
    group.to_csv(args.out / "within_class_group.csv", index=False)
    contrasts.to_csv(args.out / "within_class_vs_rest.csv", index=False)
    pd.DataFrame(quality).to_csv(args.out / "within_class_quality.csv", index=False)
    pd.DataFrame(peaks).to_csv(args.out / "within_class_peaks.csv", index=False)
    print("Wrote five within-class summary CSV files")


if __name__ == "__main__":
    main()
