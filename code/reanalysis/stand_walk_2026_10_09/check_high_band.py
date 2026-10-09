"""30-45 Hz sensitivity proxy, not an EMG measurement or source diagnosis."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from threadpoolctl import threadpool_limits

from recheck_stand_walk import HERE, CACHE_DIR, CHANNELS, CLASS_NAMES, CODE_TO_Y, FS, logvar
from reanalyse_cached_eeg import keep_position, participant_files
from pathlib import Path


def main():
    cache = CACHE_DIR
    rows = []
    for sid, date, runs in participant_files(cache):
        for tag, _, path in runs:
            with np.load(path, allow_pickle=False) as z:
                data, codes, samples = z["data"].astype(float), z["codes"], z["samples"]
            order = np.argsort(samples, kind="stable")
            codes, samples = codes[order], samples[order]
            for ref in ("ear", "car"):
                source = data if ref == "ear" else data - data.mean(axis=0, keepdims=True)
                filtered = {name: sosfiltfilt(butter(4, band, btype="bandpass", fs=FS, output="sos"), source)
                            for name, band in (("smr_8_30", (8, 30)), ("high_30_45", (30, 45)))}
                for number, pos in enumerate(np.flatnonzero(np.isin(codes, list(CODE_TO_Y))), 1):
                    if not keep_position(date, tag, number):
                        continue
                    sample, condition = int(samples[pos]), CLASS_NAMES[CODE_TO_Y[int(codes[pos])]]
                    powers = {}
                    for name, signal in filtered.items():
                        powers[name] = np.mean([logvar(signal[:, sample+round(a*FS):sample+round((a+1)*FS)])
                                                for a in (.5, 1.5, 2.5, 3.5)], axis=0)
                    for ci, channel in enumerate(CHANNELS):
                        rows.append(dict(subject=sid, run=tag, position=number, reference=ref,
                                         condition=condition, channel=channel,
                                         high_to_smr_db=float(powers["high_30_45"][ci]-powers["smr_8_30"][ci])))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "results/high_band_proxy_trials.csv", index=False)
    participant = frame.groupby(["subject", "reference", "condition"], as_index=False).high_to_smr_db.mean()
    participant.to_csv(HERE / "results/high_band_proxy_participant.csv", index=False)
    print(participant.groupby(["reference", "condition"]).high_to_smr_db.mean().to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
