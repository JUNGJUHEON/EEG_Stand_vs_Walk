"""Fixed Filterbank rerun from authorized, already-preprocessed trial arrays.

Raw extraction is not performed here. See METHODS.md for the required input
contract, retained-trial rules, and the bounds of signal-level reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from model import feature_arrays, make_model

PAIRS = {"R_vs_S": (0, 1), "R_vs_W": (0, 2), "S_vs_W": (1, 2)}


def split_indices(y, pair):
    ids = np.concatenate([np.flatnonzero(y == c) for c in pair])
    binary = (y[ids] == pair[1]).astype(int)
    development, test = train_test_split(ids, test_size=.2, stratify=binary, random_state=42)
    train, val = train_test_split(development, test_size=.25,
                                  stratify=y[development], random_state=42)
    for a, b in ((train, val), (train, test), (val, test)):
        if np.intersect1d(a, b).size:
            raise ValueError("Overlapping original trials")
    if tuple(map(len, (train, val, test))) != (60, 20, 20):
        raise ValueError("Unexpected split sizes")
    return train, val, test


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensors(features, device):
    return tuple(torch.from_numpy(a).to(device) for a in features)


def train_model(train_features, y_train, val_features, y_val, device, *, max_epochs=100):
    """Fit on training data; choose checkpoint on validation, never test."""
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5)
    batch_size, patience, min_delta = 32, 5, .0001
    criterion = nn.CrossEntropyLoss()
    train_x, val_x = tensors(train_features, device), tensors(val_features, device)
    train_y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=device)
    best_loss, best_state, best_epoch, no_improvement = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, max_epochs+1):
        model.train()
        order = torch.randperm(len(y_train)).numpy()
        running, correct = 0., 0
        for start in range(0, len(order), batch_size):
            ids = torch.as_tensor(order[start:start+batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(x[ids] for x in train_x))
            loss = criterion(logits, train_y[ids])
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(ids)
            correct += int((logits.argmax(1) == train_y[ids]).sum().item())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(*val_x)
            val_loss = float(criterion(val_logits, val_y).item())
            val_acc = float((val_logits.argmax(1) == val_y).float().mean().item())
        if not np.isfinite([running, val_loss]).all():
            raise FloatingPointError("Non-finite training/validation loss")
        history.append(dict(epoch=epoch, train_loss=running/len(y_train),
                            train_accuracy=correct/len(y_train), val_loss=val_loss,
                            val_accuracy=val_acc, lr=optimizer.param_groups[0]["lr"]))
        if val_loss < best_loss-min_delta:
            best_loss, best_epoch, no_improvement = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_state, history, best_epoch


@torch.no_grad()
def predict(model, features, batch_size=32):
    """Frozen prediction deliberately takes no truth labels."""
    model.eval()
    probabilities = []
    for start in range(0, len(features[0]), batch_size):
        batch = tuple(x[start:start+batch_size] for x in features)
        probabilities.append(model(*batch).softmax(1).cpu().numpy())
    return np.concatenate(probabilities)


def validate_input(x, y):
    if x.shape != (150, 8, 1281) or y.shape != (150,):
        raise ValueError("Expected X=(150,8,1281) and y=(150,)")
    if not np.isfinite(x).all() or not np.array_equal(np.unique(y), [0, 1, 2]):
        raise ValueError("Nonfinite input or unexpected class IDs")
    if not np.array_equal(np.bincount(y), [50, 50, 50]):
        raise ValueError("Expected fifty retained trials per class")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Authorized preprocessed S01.npz ... S10.npz directory")
    parser.add_argument("--out", type=Path, default=Path("cache/filterbank_rerun"))
    parser.add_argument("--subjects", nargs="+", default=[f"S{i:02d}" for i in range(1, 11)])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if any(re.fullmatch(r"S\d{2}", s) is None for s in args.subjects):
        parser.error("Use pseudonymous subject IDs, such as S01")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new local run directory")
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True)
    records = []
    for subject in args.subjects:
        input_path = args.data / f"{subject}.npz"
        with np.load(input_path, allow_pickle=False) as cache:
            x, y = cache["X"], cache["y"]
        validate_input(x, y)
        for contrast, pair in PAIRS.items():
            train, val, test = split_indices(y, pair)
            binary = (y == pair[1]).astype(int)
            for seed in args.seeds:
                seed_all(seed)
                train_features = feature_arrays(x[train])
                val_features = feature_arrays(x[val])
                model, state, history, epoch = train_model(
                    train_features, binary[train], val_features, binary[val], device)
                dest = args.out / "fits_LOCAL" / f"{subject}_{contrast}_{seed}"
                dest.mkdir(parents=True)
                # Persist validation-selected weights before any test prediction.
                torch.save(dict(state_dict=state, best_epoch=epoch, history=history,
                                seed=seed, subject=subject, contrast=contrast,
                                train_indices=train.tolist(), val_indices=val.tolist(),
                                test_indices=test.tolist()), dest / "best.pt")
                probabilities = predict(model, tensors(feature_arrays(x[test]), device))
                predictions = probabilities.argmax(1)
                truth = binary[test]
                np.savez_compressed(dest / "predictions.npz", indices=test, truth=truth,
                                    prediction=predictions, probabilities=probabilities)
                records.append(dict(subject=subject, contrast=contrast, seed=seed,
                                    accuracy=float(np.mean(predictions == truth)),
                                    auc=float(roc_auc_score(truth, probabilities[:, 1]))))
                print(f"Completed {subject} {contrast} seed{seed}", flush=True)
    with (args.out / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    manifest = dict(split_seed=42, training_seeds=args.seeds, device=device,
                    numpy_version=np.__version__, torch_version=torch.__version__,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
                    input_contract="See published METHODS.md; raw data not distributed")
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
