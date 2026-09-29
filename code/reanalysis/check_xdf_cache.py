"""Compare every motor-imagery XDF's events and clipping mask to its cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf


def file_sets(raw_root, cache_root):
    by_date = {}
    for cache in cache_root.glob("*_MI*.npz"):
        date, subject, tag = cache.stem.split("_")
        if tag not in ("MI", "MI2"):
            continue
        raw = raw_root / date / subject / f"{subject}_{tag}.xdf"
        if not raw.is_file():
            raise FileNotFoundError(raw)
        by_date.setdefault(date, []).append((tag, cache, raw))
    if len(by_date) != 10:
        raise ValueError("Expected ten MI participants")
    for i, date in enumerate(sorted(by_date), start=1):
        for tag, cache, raw in sorted(by_date[date]):
            yield f"S{i:02d}", tag, cache, raw


def check_one(participant, tag, cache_path, raw_path):
    streams, _ = pyxdf.load_xdf(str(raw_path))
    eeg_streams = [s for s in streams if s["info"]["type"][0].lower() == "eeg"]
    if len(eeg_streams) != 1:
        raise ValueError(f"Expected one EEG stream: {raw_path.name}")
    eeg = eeg_streams[0]
    marker = next(s for s in streams if s["info"]["name"][0] == "MarkerStream")
    ch_names = [str(c["label"][0]) for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    keep = [i for i, name in enumerate(ch_names) if name != "9"]
    raw_uv = np.asarray(eeg["time_series"], dtype=np.float64)[:, keep].T
    eeg_ts = np.asarray(eeg["time_stamps"], dtype=np.float64)
    codes = np.asarray([int(float(m[0])) for m in marker["time_series"]], dtype=np.int64)
    marker_ts = np.asarray(marker["time_stamps"], dtype=np.float64)
    sample = np.searchsorted(eeg_ts, marker_ts)
    sample = np.clip(sample, 0, len(eeg_ts) - 1)
    left = np.clip(sample - 1, 0, len(eeg_ts) - 1)
    sample = np.where(np.abs(eeg_ts[left] - marker_ts) <
                      np.abs(eeg_ts[sample] - marker_ts), left, sample)
    clip = np.abs(raw_uv) > 1e5

    with np.load(cache_path, allow_pickle=False) as z:
        cache_clip = np.unpackbits(z["clip"], axis=1, count=int(z["n_samples"]))
        counts = {
            "n_cached_samples": int(z["n_samples"]),
            "n_cached_markers": len(z["codes"]),
            "channel_match": list(z["ch_names"]) == [ch_names[i] for i in keep],
            "code_match": np.array_equal(codes, z["codes"]),
            "sample_index_match": np.array_equal(sample, z["samples"]),
            "clip_mask_match": np.array_equal(clip, cache_clip),
            "nominal_fs_match": float(z["fs"]) == float(eeg["info"]["nominal_srate"][0]),
        }
    return {
        "participant": participant, "run": tag,
        "n_raw_samples": raw_uv.shape[1], "n_raw_markers": len(codes),
        "n_raw_task_markers": int(np.isin(codes, [2, 5, 8]).sum()),
        "n_raw_clipped_samples": int(clip.sum()),
        "n_sample_difference": abs(raw_uv.shape[1] - counts["n_cached_samples"]),
        "n_marker_difference": abs(len(codes) - counts["n_cached_markers"]),
        **{k: v for k, v in counts.items() if k.endswith("_match")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for participant, tag, cache, raw in file_sets(args.raw, args.cache):
        row = check_one(participant, tag, cache, raw)
        rows.append(row)
        hashes[f"{participant}_{tag}.xdf"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        print(participant, tag, "events", row["n_raw_markers"], "match", row["code_match"]
              and row["sample_index_match"] and row["clip_mask_match"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "xdf_cache_checks.csv", index=False)
    (args.out / "xdf_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    checks = [c for c in df.columns if c.endswith("_match")]
    if not df[checks].all().all() or df["n_sample_difference"].sum() or df["n_marker_difference"].sum():
        raise SystemExit("Raw-cache mismatch found; inspect xdf_cache_checks.csv")
    print("All", len(df), "MI raw/cache comparisons passed")


if __name__ == "__main__":
    main()
