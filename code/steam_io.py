# -*- coding: utf-8 -*-
"""
STEAM / Bitbrain MI — corrected loader + preprocessing backbone
================================================================

Replaces Import/STEAM_KEEP_RULE_xdf_v2.py.

CLASS MAP
---------
Taken from Import/STEAM_KEEP_RULE_xdf_v2.py, which is the project's authority:

        2 = Rest,  5 = Stand,  8 = Walk

Verified independently against every *_MI*.xdf, the PsychoPy marker stream is
emitted as repeating quadruplets, one per trial, in randomised class order:

        0 (fixation, 3.0 s)
        1 (cue, 2.0 s)  ->  2 (task screen, 5.0 s)  ->  3 (end)
        0 (fixation, 3.0 s)
        4 (cue, 2.0 s)  ->  5 (task screen, 5.0 s)  ->  6 (end)
        0 (fixation, 3.0 s)
        7 (cue, 2.0 s)  ->  8 (task screen, 5.0 s)  ->  9 (end)

Counts per run are 150 x code 0 and 50 x each of codes 1-9, i.e. 50 trials of
each class.  Note this does NOT match the marker table in the handover deck
(slide 17), which assigns one code per screen and would give 50 x code 0 — so
that table is stale; the timing structure above is what the data contains.
The one thing the timing alone cannot fix is which of {2,5,8} is which class,
since all three have identical structure; the map above is the experimenter's.

WHAT CHANGED VS THE OLD LOADER
------------------------------
1) EPOCH WINDOW.  t = 0 at a task marker is the *onset of the MI period*.
   The old default tmin = -1.0 therefore pulled in the last second of the
   *cue* screen, which shows a class-specific visual stimulus (up arrow /
   right arrow / rest text).  That is a visual-evoked confound, not MI.
   Default here is [0.5, 4.5] — inside the MI period, past the 1 s beep.
   Measured cost of the leak: the cue-only window [-2, 0] still decodes at
   52-56 %, so the old setting was importing a few points of visual response.

2) ARTIFACT HANDLING.  The Bitbrain amp rails at ~±102 mV and several
   subjects have channels saturated for 0.4–3.4 % of the recording with
   std 10–45 mV (vs ~0.2 mV for a clean channel).  The old pipeline fed
   those straight into the classifier.  Here we detect clipping on the raw
   signal, mark/interpolate dead channels, and reject outlier epochs.
   (Measured: that saturation falls entirely in the between-block breaks —
   zero clipped samples land inside a task epoch — so what actually bites is
   the variance-outlier rejection, which drops 0-8 trials per subject.)

3) CACHING.  XDF decode is slow (~20 MB/file).  Continuous data is decoded
   once, notch-filtered, and cached as .npz so band/window sweeps are cheap.

USAGE
-----
    import steam_io as sio
    sio.build_cache()                       # one time, ~2 min
    subs = sio.list_subjects()              # [('250905','S01'), ...]
    ep   = sio.get_epochs('250905', 'S01', l_freq=8, h_freq=30,
                          tmin=0.5, tmax=4.5)
    X, y = ep['X'], ep['y']                 # (n_trials, 8, n_times), labels
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from functools import lru_cache

import numpy as np

warnings.filterwarnings("ignore")

# ============================== Paths ================================
HERE = os.path.dirname(os.path.abspath(__file__))
# Scripts live in code/ and write one level up, so results/ and cache/ sit at
# the repository root next to the manuscript that cites them.
ROOT = os.path.dirname(HERE)
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")

# Raw XDF recordings.  Not distributed with this repository: they are human
# subjects data whose release is governed by the study's ethics approval.
# Point STEAM_DATA_DIR at the directory holding the per-date recording folders,
# or place them under <repo>/data/.
BASE_DIR = os.environ.get("STEAM_DATA_DIR",
                          os.path.join(ROOT, "data", "1. BBRaw"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATE_PREFIXES = ("2509", "2511")

# ========================= Protocol constants ========================
# marker code -> class name, per Import/STEAM_KEEP_RULE_xdf_v2.py
CLASS_MAP = {2: "Rest", 5: "Stand", 8: "Walk"}
# canonical label order used everywhere downstream
CLASS_NAMES = ["Rest", "Stand", "Walk"]
CODE_TO_Y = {2: 0, 5: 1, 8: 2}          # marker -> 0/1/2 index into CLASS_NAMES
CUE_OF_TASK = {2: 1, 5: 4, 8: 7}        # task marker -> its preceding cue marker

# *_ME.xdf runs use a DIFFERENT marker layout, verified identical in all ten
# files:  1 = 8 s inter-trial rest, 2 -> 3 = 5 s executed movement A,
# 4 -> 5 = 5 s executed movement B, 6 = end.  Note that code 5 means "Stand"
# in an MI run but "executed movement B" here, so the two layouts must never
# be mixed — always pass kind= explicitly.
#
# A and B are deliberately left unnamed.  ME is used in this project only as
# an independent validation set for a directional hypothesis, and which
# executed movement is which is a separate question that the marker timing
# cannot settle (both tasks have identical structure).
ME_CLASS_MAP = {1: "RestME", 3: "TaskA", 5: "TaskB"}
ME_CLASS_NAMES = ["RestME", "TaskA", "TaskB"]
ME_CODE_TO_Y = {1: 0, 3: 1, 5: 2}


def maps_for(kind: str):
    """(code -> y index, ordered class names) for a 'MI' or 'ME' run."""
    if kind.upper().startswith("ME"):
        return ME_CODE_TO_Y, ME_CLASS_NAMES
    return CODE_TO_Y, CLASS_NAMES

EEG_CHANNELS = ["FC3", "FC4", "C3", "C4", "Cz", "CP3", "CP4", "CPz"]
COUNTER_CHANNEL = "9"                   # 16-bit sample counter, not EEG

SFREQ = 256.0
LINE_FREQ = 60.0                        # Korea mains
CLIP_UV = 1.0e5                         # amp rails sit at ~1.0207e5 uV

# Subject-specific keep rules, carried over from STEAM_KEEP_RULE_xdf_v2.py.
# Indices are 1-based positions in the *time-ordered list of task markers*
# within that run (this is index-stable regardless of epoch window).
# Keyed by recording date alone: each date holds exactly one participant, so
# the date is a sufficient key and no participant identifier need appear here.
KEEP_RULES = {
    "250911": {"MI": "1-30,61-120", "MI2": "1-60"},
    "250922": {"MI": "1-30,61-150", "MI2": "1-30"},
    "250924": {"MI": "1-30,61-150", "MI2": "1-30"},
}


def subject_for(date: str) -> str:
    """On-disk directory name for a recording date (dates are unique)."""
    for d, s in list_subjects():
        if d == date:
            return s
    raise KeyError(f"no recording for {date}")


def anon(date: str) -> str:
    """Publication label S01..S10, ordered by recording date."""
    return f"S{[d for d, _ in list_subjects()].index(date) + 1:02d}"


# ============================== Helpers ==============================
def _expand_ranges(spec: str, max_n: int) -> list[int]:
    """"1-30,61-120" or "*" -> sorted 0-based indices, clipped to [0, max_n)."""
    spec = (spec or "*").strip()
    if spec == "*":
        return list(range(max_n))
    out: set[int] = set()
    for tok in re.split(r"[,\s]+", spec):
        if not tok:
            continue
        if "-" in tok:
            a, b = (int(v) for v in tok.split("-", 1))
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(tok))
    return [i - 1 for i in sorted(out) if 1 <= i <= max_n]


def list_subjects() -> list[tuple[str, str]]:
    """All (date, subject) pairs under BASE_DIR matching DATE_PREFIXES."""
    subs = []
    for date_dir in sorted(glob.glob(os.path.join(BASE_DIR, "*"))):
        date = os.path.basename(date_dir)
        if not os.path.isdir(date_dir) or not date.startswith(DATE_PREFIXES):
            continue
        for sd in sorted(glob.glob(os.path.join(date_dir, "*"))):
            if os.path.isdir(sd):
                subs.append((date, os.path.basename(sd)))
    return subs


def run_paths(date: str, subject: str, kind: str = "MI") -> list[tuple[str, str]]:
    """[(run_tag, path), ...] for kind in {'MI','ME'}; e.g. [('MI', ...), ('MI2', ...)]."""
    d = os.path.join(BASE_DIR, date, subject)
    out = []
    for tag in (kind, kind + "2"):
        hits = glob.glob(os.path.join(d, f"*_{tag}.xdf"))
        if hits:
            out.append((tag, sorted(hits)[0]))
    return out


# ====================== XDF -> cached continuous =====================
def _decode_xdf(path: str) -> dict:
    """Decode one XDF into arrays. Returns raw (unfiltered) EEG in microvolts."""
    import pyxdf

    streams, _ = pyxdf.load_xdf(path)

    eeg = [s for s in streams if s["info"]["type"][0].lower() == "eeg"]
    eeg = eeg[0] if eeg else max(streams, key=lambda s: int(s["info"]["channel_count"][0]))
    ch_names = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    fs = float(eeg["info"]["nominal_srate"][0])

    data_uv = np.asarray(eeg["time_series"], dtype=np.float64).T   # (n_ch, n_samp)
    eeg_ts = np.asarray(eeg["time_stamps"], dtype=np.float64)

    # drop the sample-counter channel (monotonic 0..65535 ramp)
    keep = [i for i, c in enumerate(ch_names) if c != COUNTER_CHANNEL]
    data_uv = data_uv[keep]
    ch_names = [ch_names[i] for i in keep]

    mk = next(s for s in streams if s["info"]["name"][0] == "MarkerStream")
    codes = np.array([int(float(m[0])) for m in mk["time_series"]], dtype=int)
    mts = np.asarray(mk["time_stamps"], dtype=np.float64)

    # marker timestamp -> nearest EEG sample index.  Using the sample index
    # (not wall time) keeps markers aligned with the nominal-fs time base even
    # though the effective rate drifts (255.3-256.0 Hz across files).
    samp = np.searchsorted(eeg_ts, mts)
    samp = np.clip(samp, 0, len(eeg_ts) - 1)
    left = np.clip(samp - 1, 0, len(eeg_ts) - 1)
    pick_left = np.abs(eeg_ts[left] - mts) < np.abs(eeg_ts[samp] - mts)
    samp = np.where(pick_left, left, samp)

    return dict(data_uv=data_uv, ch_names=ch_names, fs=fs,
                codes=codes, samples=samp.astype(np.int64),
                n_samples=data_uv.shape[1],
                eff_fs=len(eeg_ts) / (eeg_ts[-1] - eeg_ts[0]))


def _cache_path(date: str, subject: str, tag: str) -> str:
    return os.path.join(CACHE_DIR, f"{date}_{subject}_{tag}.npz")


def build_cache(kinds=("MI",), force: bool = False, verbose: bool = True) -> None:
    """Decode every run once: notch-filter, de-mean, store as .npz (float32 V)."""
    import mne

    for (date, subject) in list_subjects():
        for kind in kinds:
            for tag, path in run_paths(date, subject, kind):
                out = _cache_path(date, subject, tag)
                if os.path.exists(out) and not force:
                    if verbose:
                        print(f"[cache] hit  {os.path.basename(out)}")
                    continue

                d = _decode_xdf(path)
                raw_uv = d["data_uv"]

                # clipping mask from the *raw* signal (band-pass hides it)
                clip = np.abs(raw_uv) > CLIP_UV

                x = raw_uv * 1e-6                        # uV -> V for MNE
                x = x - x.mean(axis=1, keepdims=True)    # kill DC (up to 17 mV)

                info = mne.create_info(d["ch_names"], d["fs"], "eeg", verbose=False)
                raw = mne.io.RawArray(x, info, verbose=False)
                raw.notch_filter([LINE_FREQ, 2 * LINE_FREQ], picks="eeg", verbose=False)

                np.savez_compressed(
                    out,
                    data=raw.get_data().astype(np.float32),
                    clip=np.packbits(clip, axis=1),
                    n_samples=d["n_samples"],
                    ch_names=np.array(d["ch_names"]),
                    fs=d["fs"], eff_fs=d["eff_fs"],
                    codes=d["codes"], samples=d["samples"],
                )
                if verbose:
                    print(f"[cache] made {os.path.basename(out)}  "
                          f"{raw_uv.shape}  eff_fs={d['eff_fs']:.2f}")


def load_run(date: str, subject: str, tag: str) -> dict:
    """Load one cached run. data is (8, n) float32 in Volts, notch-filtered."""
    p = _cache_path(date, subject, tag)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} — run build_cache() first")
    z = np.load(p, allow_pickle=False)
    n = int(z["n_samples"])
    return dict(
        data=z["data"],
        clip=np.unpackbits(z["clip"], axis=1)[:, :n].astype(bool),
        ch_names=[str(c) for c in z["ch_names"]],
        fs=float(z["fs"]), eff_fs=float(z["eff_fs"]),
        codes=z["codes"], samples=z["samples"],
    )


# ========================== Filtering + epoching =====================
def _to_raw(data: np.ndarray, ch_names: list[str], fs: float):
    import mne
    info = mne.create_info(ch_names, fs, "eeg", verbose=False)
    raw = mne.io.RawArray(data.astype(np.float64), info, verbose=False)
    raw.set_montage("standard_1020", on_missing="warn", verbose=False)
    return raw


# One entry holds a whole continuous run (~32 MB), so keep this just big
# enough to serve every run of one (band, ref) — sweeps loop band/ref
# outermost, so 16 is plenty and keeps a 7-worker pool inside RAM.
@lru_cache(maxsize=16)
def _filtered(date: str, subject: str, tag: str,
              l_freq: float, h_freq: float, ref: str) -> tuple:
    """Band-pass a cached run (memoised). ref in {'none','car'}."""
    r = load_run(date, subject, tag)
    raw = _to_raw(r["data"], r["ch_names"], r["fs"])
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg",
               method="fir", fir_design="firwin",
               l_trans_bandwidth="auto", h_trans_bandwidth="auto",
               verbose=False)
    if ref == "car":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw.get_data(), r["clip"], r["codes"], r["samples"], r["fs"], tuple(r["ch_names"])


def _epoch_run(date: str, subject: str, tag: str, *,
               l_freq: float, h_freq: float, tmin: float, tmax: float,
               ref: str, kind: str = "MI") -> dict:
    """Cut every task marker of one run into an epoch (MI: 2/5/8, ME: 1/3/5)."""
    code_to_y, class_names = maps_for(kind)
    X_c, clip, codes, samples, fs, ch_names = _filtered(
        date, subject, tag, l_freq, h_freq, ref)

    a = int(round(tmin * fs))
    b = int(round(tmax * fs))
    n_times = b - a
    n_tot = X_c.shape[1]

    order = np.argsort(samples, kind="stable")
    codes, samples = codes[order], samples[order]
    task_pos = np.where(np.isin(codes, list(code_to_y)))[0]

    X, y, clip_frac, trial_idx = [], [], [], []
    for k, pos in enumerate(task_pos):
        s0 = samples[pos] + a
        s1 = s0 + n_times
        if s0 < 0 or s1 > n_tot:
            continue
        X.append(X_c[:, s0:s1])
        y.append(code_to_y[codes[pos]])
        clip_frac.append(clip[:, s0:s1].mean(axis=1))
        trial_idx.append(k)          # position in the run's task-marker order

    if not X:
        return dict(X=np.empty((0, len(ch_names), n_times)), y=np.empty(0, int),
                    clip_frac=np.empty((0, len(ch_names))), trial_idx=np.empty(0, int),
                    n_task=len(task_pos), fs=fs, ch_names=list(ch_names))

    return dict(X=np.stack(X).astype(np.float64), y=np.asarray(y, int),
                clip_frac=np.stack(clip_frac), trial_idx=np.asarray(trial_idx, int),
                n_task=len(task_pos), fs=fs, ch_names=list(ch_names))


def get_epochs(date: str, subject: str, *,
               l_freq: float = 8.0, h_freq: float = 30.0,
               tmin: float = 0.5, tmax: float = 4.5,
               ref: str = "car",
               kind: str = "MI",
               use_keep_rules: bool = True,
               reject: bool = True,
               clip_tol: float = 0.005,
               var_mad: float = 5.0) -> dict:
    """
    Full per-subject epoch set: runs concatenated, keep-rules applied,
    artifact epochs optionally dropped.

    Parameters
    ----------
    ref        : 'car' (common average, recommended for this 8-ch motor cap)
                 or 'none'.
    reject     : drop epochs that are clipped or variance-outliers.
    clip_tol   : max fraction of clipped samples tolerated per channel.
    var_mad    : reject if any channel's log-variance is > var_mad robust
                 z-scores from that channel's median across the subject.

    Returns dict with X (n, 8, t), y (n,), plus bookkeeping fields.
    """
    parts = []
    for tag, _ in run_paths(date, subject, kind):
        ep = _epoch_run(date, subject, tag, l_freq=l_freq, h_freq=h_freq,
                        tmin=tmin, tmax=tmax, ref=ref, kind=kind)
        if ep["X"].shape[0] == 0:
            continue
        if use_keep_rules and not kind.upper().startswith("ME"):
            spec = KEEP_RULES.get(date, {}).get(tag, "*")
            keep = set(_expand_ranges(spec, ep["n_task"]))
            sel = np.array([i for i, t in enumerate(ep["trial_idx"]) if t in keep], int)
            if sel.size == 0:
                continue
            for k in ("X", "y", "clip_frac", "trial_idx"):
                ep[k] = ep[k][sel]
        ep["run"] = np.full(len(ep["y"]), tag)
        parts.append(ep)

    if not parts:
        raise RuntimeError(f"no epochs for {date}/{subject}")

    X = np.concatenate([p["X"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    clip_frac = np.concatenate([p["clip_frac"] for p in parts])
    run = np.concatenate([p["run"] for p in parts])
    fs, ch_names = parts[0]["fs"], parts[0]["ch_names"]

    good = np.ones(len(y), bool)
    reasons = dict(clip=0, var=0)
    if reject and len(y):
        bad_clip = (clip_frac > clip_tol).any(axis=1)
        reasons["clip"] = int(bad_clip.sum())

        logv = np.log(X.var(axis=2) + 1e-30)                 # (n, ch)
        med = np.median(logv, axis=0)
        mad = np.median(np.abs(logv - med), axis=0) * 1.4826 + 1e-12
        bad_var = (np.abs(logv - med) / mad > var_mad).any(axis=1)
        reasons["var"] = int(bad_var.sum() - (bad_var & bad_clip).sum())

        good = ~(bad_clip | bad_var)

    return dict(
        X=X[good], y=y[good], run=run[good],
        clip_frac=clip_frac[good], fs=fs, ch_names=ch_names,
        n_before=len(y), n_dropped=int((~good).sum()), drop_reasons=reasons,
        subject=f"{date}/{subject}",
        class_names=maps_for(kind)[1],
    )


def pair(ep: dict, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    """Binary subset, e.g. pair(ep, 'Stand', 'Walk') -> X, y in {0,1}."""
    ia, ib = CLASS_NAMES.index(a), CLASS_NAMES.index(b)
    m = np.isin(ep["y"], [ia, ib])
    return ep["X"][m], (ep["y"][m] == ib).astype(int)


DEFAULT_BANDS = ((4, 8), (8, 12), (12, 16), (16, 20),
                 (20, 24), (24, 28), (28, 32), (32, 40))


def get_bank(date: str, subject: str, *,
             bands=DEFAULT_BANDS,
             tmin: float = 0.5, tmax: float = 4.5,
             ref: str = "car", kind: str = "MI",
             use_keep_rules: bool = True,
             reject: bool = True,
             broad: tuple[float, float] = (4.0, 40.0),
             **rej_kw) -> dict:
    """
    Filter-bank version of get_epochs.

    Epoch rejection is decided once on the `broad` band and the *same* trial
    mask is applied to every sub-band, so all bands stay row-aligned.

    Returns dict with Xb : list of (n, 8, t) arrays, one per band, plus
    X (the broadband version), y, and the usual bookkeeping.
    """
    base = get_epochs(date, subject, l_freq=broad[0], h_freq=broad[1],
                      tmin=tmin, tmax=tmax, ref=ref, kind=kind,
                      use_keep_rules=use_keep_rules, reject=False)
    n = len(base["y"])

    good = np.ones(n, bool)
    reasons = dict(clip=0, var=0)
    if reject and n:
        bad_clip = (base["clip_frac"] > rej_kw.get("clip_tol", 0.005)).any(axis=1)
        logv = np.log(base["X"].var(axis=2) + 1e-30)
        med = np.median(logv, axis=0)
        mad = np.median(np.abs(logv - med), axis=0) * 1.4826 + 1e-12
        bad_var = (np.abs(logv - med) / mad > rej_kw.get("var_mad", 5.0)).any(axis=1)
        reasons = dict(clip=int(bad_clip.sum()),
                       var=int(bad_var.sum() - (bad_var & bad_clip).sum()))
        good = ~(bad_clip | bad_var)

    Xb = []
    for lo, hi in bands:
        e = get_epochs(date, subject, l_freq=lo, h_freq=hi, tmin=tmin, tmax=tmax,
                       ref=ref, kind=kind, use_keep_rules=use_keep_rules,
                       reject=False)
        assert len(e["y"]) == n and np.array_equal(e["y"], base["y"])
        Xb.append(e["X"][good])

    return dict(Xb=Xb, bands=list(bands), X=base["X"][good], y=base["y"][good],
                run=base["run"][good], fs=base["fs"], ch_names=base["ch_names"],
                n_before=n, n_dropped=int((~good).sum()), drop_reasons=reasons,
                subject=f"{date}/{subject}", class_names=CLASS_NAMES)


def get_task_arrays(date: str, subject: str, *,
                    lo: float, hi: float, tmin: float, tmax: float,
                    ref: str = "none", bands=DEFAULT_BANDS, **kw):
    """
    Plain-band and filter-bank views of the SAME trials.

    get_epochs() and get_bank() each decide artifact rejection on their own
    band, so calling them separately can leave the two views with different
    trial counts.  Requesting the plain band as an extra member of the bank
    forces one rejection decision for both.

    Returns (X_plain (n,8,t), X_bank (n,B,8,t), y (n,)).
    """
    bk = get_bank(date, subject, bands=tuple(bands) + ((lo, hi),),
                  tmin=tmin, tmax=tmax, ref=ref, **kw)
    return bk["Xb"][-1], np.stack(bk["Xb"][:-1], axis=1), bk["y"]


if __name__ == "__main__":
    build_cache()
    print()
    for date, subj in list_subjects():
        try:
            ep = get_epochs(date, subj)
        except Exception as e:                                   # noqa: BLE001
            print(f"{date}/{subj}: {e}")
            continue
        cnt = {n: int((ep["y"] == i).sum()) for i, n in enumerate(CLASS_NAMES)}
        print(f"{date}/{subj:<4} kept {len(ep['y']):3d}/{ep['n_before']:3d} "
              f"(drop {ep['n_dropped']:3d} {ep['drop_reasons']})  {cnt}")
