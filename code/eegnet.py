# -*- coding: utf-8 -*-
"""
EEGNet-8,2 (Lawhern et al. 2018) in PyTorch, adapted to 256 Hz, plus a small
training harness with early stopping and light augmentation.

Kept deliberately close to the paper so the number is comparable with the
lab's existing EEGNet scripts — the difference in results should come from the
corrected labels / window / reference, not from a different architecture.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _MaxNormConv2d(nn.Conv2d):
    """Conv2d with the per-filter max-norm constraint EEGNet specifies."""

    def __init__(self, *a, max_norm=1.0, **kw):
        super().__init__(*a, **kw)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            n = self.weight.norm(2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            self.weight.mul_((n.clamp(max=self.max_norm) / n))
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_ch=8, n_times=1024, n_classes=2,
                 F1=8, D=2, F2=16, kern_len=128, drop=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depth = _MaxNormConv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(drop)

        self.sep_d = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                               groups=F1 * D, bias=False)
        self.sep_p = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(drop)

        with torch.no_grad():
            n_feat = self._features(torch.zeros(1, 1, n_ch, n_times)).shape[1]
        self.fc = nn.Linear(n_feat, n_classes)

    def _features(self, x):
        x = self.bn1(self.conv1(x))
        x = self.drop1(self.pool1(F.elu(self.bn2(self.depth(x)))))
        x = self.sep_p(self.sep_d(x))
        x = self.drop2(self.pool2(F.elu(self.bn3(x))))
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self._features(x))


# ============================ training loop ==========================
def _standardize(Xtr, *others):
    """Per-channel z-score using TRAIN statistics only."""
    m = Xtr.mean(axis=(0, 2), keepdims=True)
    s = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return [(A - m) / s for A in (Xtr, *others)]


def _augment(xb, yb, max_shift=32, noise=0.05):
    """Random circular time shift + gaussian noise (cheap MI-safe augmentation)."""
    n = xb.shape[0]
    sh = torch.randint(-max_shift, max_shift + 1, (n,), device=xb.device)
    idx = (torch.arange(xb.shape[-1], device=xb.device)[None, :] + sh[:, None]) % xb.shape[-1]
    xb = torch.gather(xb, -1, idx[:, None, None, :].expand_as(xb))
    return xb + noise * torch.randn_like(xb) * xb.std(), yb


def fit_predict(Xtr, ytr, Xte, *, n_classes=2, epochs=300, patience=40,
                bs=32, lr=1e-3, wd=1e-3, val_frac=0.2, seed=0, verbose=False):
    """
    Train EEGNet on (Xtr, ytr) with an internal validation split for early
    stopping, then return predicted labels and probabilities for Xte.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    dev = device()

    sss = StratifiedShuffleSplit(1, test_size=val_frac, random_state=seed)
    tr, va = next(sss.split(Xtr, ytr))
    Xa, Xv, Xt = _standardize(Xtr[tr], Xtr[va], Xte)

    to = lambda A: torch.as_tensor(A, dtype=torch.float32, device=dev).unsqueeze(1)
    Xa_t, Xv_t, Xt_t = to(Xa), to(Xv), to(Xt)
    ya_t = torch.as_tensor(ytr[tr], dtype=torch.long, device=dev)
    yv_t = torch.as_tensor(ytr[va], dtype=torch.long, device=dev)

    model = EEGNet(n_ch=Xtr.shape[1], n_times=Xtr.shape[2], n_classes=n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, best_state, bad = np.inf, None, 0
    n = len(ya_t)
    for ep in range(epochs):
        model.train()
        perm = torch.as_tensor(rng.permutation(n), device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb, yb = _augment(Xa_t[idx], ya_t[idx])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv_t), yv_t).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f"    ep{ep:3d} val_loss={vl:.4f} best={best:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xt_t)
        prob = torch.softmax(logits, 1).cpu().numpy()
    return prob.argmax(1), prob
