"""E-B — Một mô hình PHI TUYẾN nhẹ trên cùng mặt phẳng tài nguyên, giao thức P1 (GT-A).

Mục đích: kiểm tra kết luận bão hoà của Mục 4.1 có phải artifact của LDA không.

Giữ nguyên mọi thứ của P1: de_movingAve · cả 3 session · z-score theo subject · LOSO 15 ·
cùng 7 tập kênh lồng nhau · cùng bộ quantizer PTQ của `matrix2.py`.

Mô hình: MLP D -> H -> 3, ReLU, dropout 0.25. Adam lr 1e-3, batch 512, 20 epoch CỐ ĐỊNH,
**báo cáo epoch cuối, KHÔNG có bước chọn checkpoint** — phải nhất quán với luận điểm của chính
bài (Mục 4.8): mọi tiêu chí chọn checkpoint không dùng nhãn target đều không tốt hơn không chọn.

Footprint = (D*H + H*3) * b/8 byte trọng số + (H+3)*4 byte bias ở 32 bit.

Resume theo từng (H, số kênh). Chạy lại bao nhiêu lần cũng được.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

CACHE = "/mnt/user-data/uploads/1.EEG_Datasets/_rivf2026_cache"
OUT = "/home/claude/kgcc/results/eb_mlp.json"
BITS = [2, 3, 4, 6, 8, None]
MODES = ["row_mse", "tensor_max"]
EPOCHS, BATCH = 20, 512

M14 = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
M32 = ['FP1','FP2','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4','FT8','T7','C3','CZ','C4','T8',
       'TP7','CP3','CPZ','CP4','TP8','P7','P3','PZ','P4','P8','PO7','PO3','PO4','PO8','OZ']


def _q(W, s, b):
    lv = 2 ** (b - 1)
    return np.round(W / s).clip(-lv, lv - 1) * s


def quant(W, b, mode):
    if b is None:
        return W
    lv = 2 ** (b - 1) - 1
    if mode == "tensor_max":
        s = np.full((W.shape[0], 1), np.abs(W).max() / lv + 1e-12)
    elif mode == "row_mse":
        mx = np.abs(W).max(1, keepdims=True); best = None; bs = None
        for al in np.linspace(0.25, 1.0, 16):
            s = al * mx / lv + 1e-12
            e = ((_q(W, s, b) - W) ** 2).sum(1, keepdims=True)
            if best is None:
                best, bs = e, s
            else:
                m = e < best; bs = np.where(m, s, bs); best = np.where(m, e, best)
        s = bs
    else:
        raise ValueError(mode)
    return _q(W, s, b)


class MLP(nn.Module):
    def __init__(self, d, h, c=3):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.fc2 = nn.Linear(h, c)
        self.act = nn.ReLU(); self.drop = nn.Dropout(0.25)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


def run_cell(A, y, subj, h, seed=0):
    """LOSO 15 fold; tra ve dict {mode|bits: [15 accuracy]}."""
    torch.set_num_threads(2)
    out = {f"{m}|{b}": [] for m in MODES for b in BITS}
    d = A.shape[1]
    for u in np.unique(subj):
        torch.manual_seed(seed); np.random.seed(seed)
        te = subj == u; tr = ~te
        Xtr = torch.from_numpy(A[tr]); ytr = torch.from_numpy((y[tr] + 1).astype(np.int64))
        Xte = torch.from_numpy(A[te]); yte = (y[te] + 1).astype(np.int64)
        net = MLP(d, h); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss(); n = len(Xtr)
        for ep in range(EPOCHS):                       # so epoch CO DINH, khong early stopping
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n - BATCH + 1, BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                lossf(net(Xtr[idx]), ytr[idx]).backward()
                opt.step()
        W1 = net.fc1.weight.detach().numpy().astype(np.float64)
        W2 = net.fc2.weight.detach().numpy().astype(np.float64)
        net.eval()
        for md in MODES:
            for b in BITS:
                with torch.no_grad():
                    net.fc1.weight.copy_(torch.from_numpy(quant(W1, b, md).astype(np.float32)))
                    net.fc2.weight.copy_(torch.from_numpy(quant(W2, b, md).astype(np.float32)))
                    pred = net(Xte).argmax(1).numpy()
                out[f"{md}|{b}"].append(float((pred == yte).mean()))
        with torch.no_grad():                          # tra lai trong so float
            net.fc1.weight.copy_(torch.from_numpy(W1.astype(np.float32)))
            net.fc2.weight.copy_(torch.from_numpy(W2.astype(np.float32)))
    return out


def main(budget=520, hiddens=(32, 8)):
    t0 = time.time()
    CH = json.load(open(f"{CACHE}/info.json"))["channels"]
    S = {1: ['T7'], 2: ['T7', 'T8'], 4: ['T7', 'T8', 'P7', 'P8'],
         8: ['T7', 'T8', 'P7', 'P8', 'F3', 'F4', 'O1', 'O2'], 14: M14, 32: M32, 62: CH}
    m = np.load(f"{CACHE}/meta.npz")
    y, subj = m["label"].astype(np.int8), m["subject"]
    X = np.load(f"{CACHE}/X_de_movingAve.npy", mmap_mode="r")
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}

    todo = [(h, nc) for h in hiddens for nc in sorted(S) if f"h{h}|{nc}" not in R]
    for h, nc in todo:
        if time.time() - t0 > budget:
            print(f"het gio, con {len(todo) - todo.index((h, nc))} o", flush=True)
            return 1
        cols = [CH.index(c) * 5 + b for c in S[nc] for b in range(5)]
        A = np.asarray(X[:, cols], dtype=np.float32)
        for u in np.unique(subj):
            k = subj == u
            a = A[k]; A[k] = (a - a.mean(0)) / (a.std(0) + 1e-8)
        cell = run_cell(A, y, subj, h)
        R[f"h{h}|{nc}"] = {"hidden": h, "channels": nc, "dims": len(cols), "cells": cell}
        json.dump(R, open(OUT, "w"))
        print(f"H={h:>3} {nc:>3} kenh xong ({time.time()-t0:.0f}s) "
              f"float={100*np.mean(cell['row_mse|None']):.2f} "
              f"4bit={100*np.mean(cell['row_mse|4']):.2f} "
              f"2bit={100*np.mean(cell['row_mse|2']):.2f}", flush=True)
    print("XONG TAT CA")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 520))
