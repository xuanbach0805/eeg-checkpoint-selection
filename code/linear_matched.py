"""Protocol-matched linear baseline for the KGCC comparison.

The existing channel x bit-width matrix (matrix2.py) used de_movingAve, all 3 sessions,
per-subject z-score.  KGCC/PR-PL uses de_LDS, session 1 only, per-session min-max
(torcheeg after_hook_normalize).  A head-to-head needs IDENTICAL inputs, so this script
reads the very same grid tensor that is fed to KGCC and flattens it back to 310-d.

Quantizers are copied verbatim from matrix2.py so the numbers stay comparable with the
existing tables.
"""
import json, os, time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import f1_score

DATA = "/home/claude/kgcc/data"
OUT = "/home/claude/kgcc/results/linear_matched.json"
BITS = [2, 3, 4, 6, 8, None]
MODES = ["row_max", "tensor_max", "row_p999", "row_p99", "row_mse"]

SEED_CHANNEL_LIST = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5',
    'FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4',
    'P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
SEED_LOCATION_LIST = [
    ['-','-','-','FP1','FPZ','FP2','-','-','-'],
    ['-','-','-','AF3','-','AF4','-','-','-'],
    ['F7','F5','F3','F1','FZ','F2','F4','F6','F8'],
    ['FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8'],
    ['T7','C5','C3','C1','CZ','C2','C4','C6','T8'],
    ['TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8'],
    ['P7','P5','P3','P1','PZ','P2','P4','P6','P8'],
    ['-','PO7','PO5','PO3','POZ','PO4','PO6','PO8','-'],
    ['-','-','CB1','O1','OZ','O2','CB2','-','-']]

M14 = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
M32 = ['FP1','FP2','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4','FT8','T7','C3','CZ','C4',
       'T8','TP7','CP3','CPZ','CP4','TP8','P7','P3','PZ','P4','P8','PO7','PO3','PO4','PO8','OZ']
SUBSETS = {1: ['T7'], 2: ['T7','T8'], 4: ['T7','T8','P7','P8'],
           8: ['T7','T8','P7','P8','F3','F4','O1','O2'], 14: M14, 32: M32,
           62: SEED_CHANNEL_LIST}


def _q(Wm, s, b):
    lv = 2 ** (b - 1)
    return np.round(Wm / s).clip(-lv, lv - 1) * s


def quant(Wm, b, mode):
    if b is None:
        return Wm
    lv = 2 ** (b - 1) - 1
    if mode == 'row_max':
        s = np.abs(Wm).max(1, keepdims=True) / lv + 1e-12
    elif mode == 'tensor_max':
        s = np.full((Wm.shape[0], 1), np.abs(Wm).max() / lv + 1e-12)
    elif mode == 'row_p999':
        s = np.percentile(np.abs(Wm), 99.9, axis=1, keepdims=True) / lv + 1e-12
    elif mode == 'row_p99':
        s = np.percentile(np.abs(Wm), 99.0, axis=1, keepdims=True) / lv + 1e-12
    elif mode == 'row_mse':
        mx = np.abs(Wm).max(1, keepdims=True); best = None; bs = None
        for al in np.linspace(0.25, 1.0, 16):
            s = al * mx / lv + 1e-12
            e = ((_q(Wm, s, b) - Wm) ** 2).sum(1, keepdims=True)
            if best is None:
                best, bs = e, s
            else:
                m = e < best; bs = np.where(m, s, bs); best = np.where(m, e, best)
        s = bs
    else:
        raise ValueError(mode)
    return _q(Wm, s, b)


def main():
    grid = np.load(f"{DATA}/grid_s1.npy", mmap_mode="r")
    meta = np.load(f"{DATA}/meta_s1.npz")
    subj, y = meta["subject"], meta["label"].astype(np.int8)

    pos = {ch: (r, c) for r, row in enumerate(SEED_LOCATION_LIST)
           for c, ch in enumerate(row) if ch != '-'}
    t0 = time.time()
    res = {}
    for nc in sorted(SUBSETS):
        names = SUBSETS[nc]
        cols = [(pos[c][0], pos[c][1]) for c in names]
        A = np.stack([np.asarray(grid[:, b, r, c]) for (r, c) in cols for b in range(5)], axis=1)
        A = A.astype(np.float32)
        cell = {f'{md}|{b}': {'acc': [], 'f1': []} for md in MODES for b in BITS}
        for u in np.unique(subj):
            te = subj == u; tr = ~te
            L = LDA().fit(A[tr], y[tr]); Z = A[te]
            for md in MODES:
                for b in BITS:
                    W = quant(L.coef_.copy(), b, md)
                    p = L.classes_[np.argmax(Z @ W.T + L.intercept_, 1)]
                    cell[f'{md}|{b}']['acc'].append(float((p == y[te]).mean()))
                    cell[f'{md}|{b}']['f1'].append(float(f1_score(y[te], p, average='macro')))
        res[str(nc)] = {'dims': A.shape[1], 'channels': names, 'cells': cell}
        print(f"{nc} channels done ({time.time()-t0:.0f}s) "
              f"float={100*np.mean(cell['row_mse|None']['acc']):.2f}", flush=True)
        json.dump(res, open(OUT, 'w'))

    maj = float(max(np.bincount(y + 1)) / len(y))
    print(f"\nglobal majority = {100*maj:.2f}  chance = 33.33")
    print(f"{'ch':>4} {'dims':>5} " + " ".join(f"{b if b else 'flt':>7}" for b in BITS))
    for nc in sorted(SUBSETS):
        c = res[str(nc)]['cells']
        row = " ".join(f"{100*np.mean(c[f'row_mse|{b}']['acc']):7.2f}" for b in BITS)
        print(f"{nc:>4} {res[str(nc)]['dims']:>5} {row}")


if __name__ == "__main__":
    main()
