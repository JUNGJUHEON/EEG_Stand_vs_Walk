"""Stationary-noise control for the fixation-baselined ERD calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mne.time_frequency import tfr_array_morlet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260928)
    fs = 256
    frequencies = np.arange(4, 41, 1.5)
    frequencies = frequencies[(frequencies >= 8) & (frequencies <= 13)]
    power = tfr_array_morlet(
        rng.standard_normal((500, 1, 10 * fs)), fs, frequencies,
        n_cycles=np.clip(frequencies / 2.0, 3, 12), output="power",
        decim=8, n_jobs=1, verbose=False,
    )
    times = np.arange(power.shape[-1]) * 8 / fs - 5.0
    baseline = (times >= -4.5) & (times <= -2.5)
    task = (times >= 0.5) & (times <= 4.5)
    baseline_mean = power[:, :, :, baseline].mean(axis=-1)
    old = 10 * np.log10(
        power[:, :, :, task] / (baseline_mean[:, :, :, None] + 1e-30) + 1e-30
    )
    symmetric = 10 * np.log10(
        (power[:, :, :, task].mean(axis=-1) + 1e-30) /
        (baseline_mean + 1e-30)
    )
    result = {
        "signal": "stationary zero-mean Gaussian noise, no task effect",
        "seed": 20260928,
        "n_simulated_trials": 500,
        "mu_frequencies_hz": frequencies.tolist(),
        "original_mean_instantaneous_log_ratio_db": float(old.mean()),
        "symmetric_log_mean_power_ratio_db": float(symmetric.mean()),
    }
    (args.out / "erd_estimator_stationary_null.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
