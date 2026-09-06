"""E-A2 — Lượng tử hoá hậu huấn luyện (PTQ) trọng số KGCC/PR-PL, đặt CNN lên
cùng mặt phẳng tài nguyên với bộ phân loại tuyến tính.

Dùng ĐÚNG hai quantizer của `paper3_ondevice/matrix2.py` (copy nguyên văn hàm `quant`),
áp per-output-row: conv weight (O,I,k,k) -> reshape (O,-1); fc weight (O,I); U (32,32);
stored_mat (32,3) -> chuyển vị thành (3,32) để hàng = lớp đầu ra.
Bias giữ float (chỉ 288/112,416 tham số = 0.26%) nhưng VẪN tính vào footprint ở 32 bit.
"""
import glob, json, os
import numpy as np
import torch
import torch.nn.functional as F
from model_PR_PL import Domain_adaption_model

UP = "/mnt/user-data/uploads/10.phD/1.EEG/2026/Journal2026_pilot/BSPC2026/results/kgcc"
DATA = "/home/claude/kgcc/data"
OUT = "/home/claude/kgcc/results/ea2_ptq_kgcc.json"
BITS = [2, 3, 4, 6, 8, None]
MODES = ["row_mse", "tensor_max"]
torch.set_num_threads(2)


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


WKEYS = ["fea_extrator_f.conv1.weight", "fea_extrator_f.conv2.weight",
         "fea_extrator_f.conv3.weight", "fea_extrator_f.fc1.weight",
         "fea_extrator_f.fc2.weight"]
BKEYS = ["fea_extrator_f.conv1.bias", "fea_extrator_f.conv2.bias",
         "fea_extrator_f.conv3.bias", "fea_extrator_f.fc1.bias",
         "fea_extrator_f.fc2.bias"]


def footprint_bytes(b):
    """Trọng số ở b bit, bias ở 32 bit. b=None -> tất cả 32 bit."""
    nw = 1440 + 18432 + 73728 + 16384 + 1024 + 1024 + 96      # conv1-3, fc1-2, U, stored_mat
    nb = 32 + 64 + 128 + 32 + 32                              # bias
    return nw * (32 if b is None else b) / 8 + nb * 4


def predict(sd, stored_mat, cluster_label, Xt, bs=1024):
    m = Domain_adaption_model(32, 32, 32, 32, 3, 32, 90, 0.9, 0.5, "cpu")
    m.load_state_dict(sd); m.stored_mat = stored_mat; m.cluster_label = cluster_label
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            feat = m.fea_extrator_f(Xt[i:i + bs])
            logit = torch.matmul(torch.matmul(m.U, feat.T).T, m.stored_mat)
            out.append(torch.softmax(logit, 1).argmax(1).numpy())
    cl = np.concatenate(out); mapped = np.zeros_like(cl)
    for i in range(len(cluster_label)):
        mapped[cl == i] = cluster_label[i]
    return mapped


def main():
    grid = np.load(f"{DATA}/grid_s1.npy", mmap_mode="r")
    meta = np.load(f"{DATA}/meta_s1.npz")
    subj, lab = meta["subject"], meta["label"]
    subjects = np.unique(subj)

    res = {f"{md}|{b}": [] for md in MODES for b in BITS}
    for fold in range(15):
        p = f"{UP}/fold{fold:02d}_model.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd0, sm, cl = ck["model"], ck["stored_mat"], ck["cluster_label"]
        ts = int(subjects[fold]); tgt = subj == ts
        Xt = torch.from_numpy(np.asarray(grid[tgt]))
        y = lab[tgt].astype(np.int64) + 1

        for md in MODES:
            for b in BITS:
                sd = {k: v.clone() for k, v in sd0.items()}
                for k in WKEYS:
                    W = sd[k].detach().numpy(); sh = W.shape
                    Wq = quant(W.reshape(sh[0], -1).astype(np.float64), b, md)
                    sd[k] = torch.from_numpy(Wq.reshape(sh).astype(np.float32))
                sd["U"] = torch.from_numpy(
                    quant(sd["U"].detach().numpy().astype(np.float64), b, md).astype(np.float32))
                smq = torch.from_numpy(
                    quant(sm.detach().numpy().T.astype(np.float64), b, md).T.astype(np.float32))
                pred = predict(sd, smq, cl, Xt)
                res[f"{md}|{b}"].append(float((pred == y).mean()))
        print(f"fold {fold} xong  (row_mse: "
              + " ".join(f"{b if b else 'flt'}={100*res[f'row_mse|{b}'][-1]:.1f}" for b in BITS)
              + ")", flush=True)
        json.dump(res, open(OUT, "w"))

    print("\n=== E-A2: KGCC/PR-PL sau PTQ, 15 fold LOSO ===")
    print(f"{'bit':>5} {'byte':>10} {'row_mse':>9} {'tensor_max':>11}")
    for b in BITS:
        r1 = 100 * np.mean(res[f"row_mse|{b}"]); r2 = 100 * np.mean(res[f"tensor_max|{b}"])
        print(f"{str(b) if b else 'float':>5} {footprint_bytes(b):>10,.0f} {r1:>9.2f} {r2:>11.2f}")


if __name__ == "__main__":
    main()
