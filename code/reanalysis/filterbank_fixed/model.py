"""User-provided Filterbank v4 definition, adapted to a two-class output.

The class and Butterworth helper retain the original computation. This module
contains definitions only; it never loads recordings or starts a search.
"""

import numpy as np
from scipy.signal import butter, filtfilt, stft
import torch
from torch import nn
from torch.nn import functional as F

BANDS = [(1, 60), (1, 4), (4, 8), (8, 13), (14, 30)]


def butter_bandpass_filter(X, low, high, fs, order=4):
    nyq=0.5*fs; b,a=butter(order,[low/nyq,high/nyq],btype='band')
    return filtfilt(b,a,X,axis=-1)


class FilterbankTripleShallowCNN(nn.Module):
    def __init__(self,n_bands,n_channels,n_classes,dropout=0.5):
        super().__init__()
        self.b1_t        = nn.Conv2d(1,  40,(1,25),bias=False)
        self.b1_s        = nn.Conv2d(40, 40,(n_channels,1),bias=False)
        self.b1_bn       = nn.BatchNorm2d(40)
        self.b2_tc       = nn.Conv1d(n_channels, 40,32,bias=False)
        self.b2_dw       = nn.Conv1d(40,       40,25,groups=40,bias=False)
        self.b2_bn       = nn.BatchNorm1d(40)
        self.b2_pool     = nn.AvgPool1d(75,32)
        self.b3_c1       = nn.Conv2d(n_channels,30,(27,1),bias=False)
        self.b3_p1       = nn.MaxPool2d((1,2))
        self.b3_c2       = nn.Conv2d(30,      40,(5,1),bias=False)
        self.b3_p2       = nn.MaxPool2d((1,3))
        self.b3_bn       = nn.BatchNorm2d(40)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_drop     = nn.Dropout(dropout)
        self.fc          = nn.Linear(40*3,n_classes)
    def forward(self,x_fb,x_stft):
        bsz,nb,C,T = x_fb.shape
        xf = x_fb.view(bsz*nb,C,T)
        # branch1
        b1 = xf.unsqueeze(1)
        b1 = self.b1_t(b1); b1 = self.b1_s(b1)
        b1 = F.elu(self.b1_bn(b1))
        b1 = self.global_pool(b1).view(bsz*nb,-1)
        b1 = b1.view(bsz,nb,-1).mean(1)
        # branch2
        b2 = self.b2_tc(xf); b2 = self.b2_dw(b2)
        b2 = F.elu(self.b2_bn(b2))
        b2 = self.b2_pool(b2).unsqueeze(-1)
        b2 = self.global_pool(b2).view(bsz*nb,-1)
        b2 = b2.view(bsz,nb,-1).mean(1)
        # branch3
        b3 = self.b3_c1(x_stft); b3 = F.elu(self.b3_p1(b3))
        b3 = self.b3_c2(b3);     b3 = F.elu(self.b3_p2(b3))
        b3 = self.b3_bn(b3)
        b3 = self.global_pool(b3).view(bsz,-1)
        # concat & classify
        out = torch.cat([b1,b2,b3],dim=1)
        return self.fc(self.fc_drop(out))


def feature_arrays(x):
    bank = np.stack([butter_bandpass_filter(x, low, high, 256.)
                     for low, high in BANDS], axis=1)
    _, _, z = stft(bank[:, 0], fs=256., nperseg=128, noverlap=64,
                   boundary=None, nfft=128, axis=-1)
    return tuple(np.ascontiguousarray(a, dtype=np.float32)
                 for a in (bank, np.abs(z)))


def make_model():
    return FilterbankTripleShallowCNN(n_bands=5, n_channels=8,
                                      n_classes=2, dropout=.3)
