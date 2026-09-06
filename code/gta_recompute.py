"""Chạy lại ma trận GT-A (giao thức CHÍNH của bài) để có per-fold accuracy → CI + Wilcoxon.

GT-A = de_movingAve · CẢ 3 SESSION · z-score theo từng subject · LDA · LOSO 15 · PTQ.
Đây đúng là giao thức của `paper3_ondevice/matrix2.py`; file kết quả gốc `~/work/matrix2_res.json`
đã mất khi VM bị dọn, nên phải dựng lại (và lần này lưu per-fold để tính CI).

Resume theo từng mức số kênh — chạy lại bao nhiêu lần cũng được.
"""
import json, os, sys, time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import f1_score

CACHE = "/mnt/user-data/uploads/1.EEG_Datasets/_rivf2026_cache"
OUT = "/home/claude/kgcc/results/gta_matrix.json"
BITS = [2, 3, 4, 6, 8, None]
MODES = ["row_max", "tensor_max", "row_p999", "row_p99", "row_mse"]

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
    if mode == 'row_max':
        s = np.abs(W).max(1, keepdims=True) / lv + 1e-12
    elif mode == 'tensor_max':
        s = np.full((W.shape[0], 1), np.abs(W).max() / lv + 1e-12)
    elif mode == 'row_p999':
        s = np.percentile(np.abs(W), 99.9, axis=1, keepdims=True) / lv + 1e-12
    elif mode == 'row_p99':
        s = np.percentile(np.abs(W), 99.0, axis=1, keepdims=True) / lv + 1e-12
    elif mode == 'row_mse':
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


def main(budget=520):
    t0 = time.time()
    CH = json.load(open(f"{CACHE}/info.json"))["channels"]
    S = {1: ['T7'], 2: ['T7', 'T8'], 4: ['T7', 'T8', 'P7', 'P8'],
         8: ['T7', 'T8', 'P7', 'P8', 'F3', 'F4', 'O1', 'O2'], 14: M14, 32: M32, 62: CH}
    m = np.load(f"{CACHE}/meta.npz")
    y, subj = m["label"].astype(np.int8), m["subject"]
    X = np.load(f"{CACHE}/X_de_movingAve.npy", mmap_mode="r")
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}

    for nc in sorted(S):
        if str(nc) in R:
            continue
        if time.time() - t0 > budget:
            print(f"het gio, con lai: {[k for k in sorted(S) if str(k) not in R]}", flush=True)
            return 1
        cols = [CH.index(c) * 5 + b for c in S[nc] for b in range(5)]
        A = np.asarray(X[:, cols], dtype=np.float32)
        for u in np.unique(subj):                        # z-score theo tung subject
            k = subj == u
            a = A[k]; A[k] = (a - a.mean(0)) / (a.std(0) + 1e-8)
        cell = {f"{md}|{b}": {"acc": [], "f1": []} for md in MODES for b in BITS}
        for u in np.unique(subj):
            te = subj == u; tr = ~te
            L = LDA().fit(A[tr], y[tr]); Z = A[te]
            for md in MODES:
                for b in BITS:
                    W = quant(L.coef_.copy(), b, md)
                    p = L.classes_[np.argmax(Z @ W.T + L.intercept_, 1)]
                    cell[f"{md}|{b}"]["acc"].append(float((p == y[te]).mean()))
                    cell[f"{md}|{b}"]["f1"].append(float(f1_score(y[te], p, average="macro")))
        R[str(nc)] = {"dims": len(cols), "channels": S[nc], "cells": cell}
        json.dump(R, open(OUT, "w"))
        print(f"{nc:>3} kenh xong ({time.time()-t0:.0f}s) "
              f"float={100*np.mean(cell['row_mse|None']['acc']):.2f}", flush=True)

    print("\n=== GT-A day du (row_mse) ===")
    print(f"{'ch':>4}{'dims':>6}" + "".join(f"{(b if b else 'flt'):>8}" for b in BITS))
    for nc in sorted(S):
        c = R[str(nc)]["cells"]
        print(f"{nc:>4}{R[str(nc)]['dims']:>6}"
              + "".join(f"{100*np.mean(c[f'row_mse|{b}']['acc']):8.2f}" for b in BITS))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 520))
