"""E-D — Sinh dữ liệu cho RTL: trọng số, bias, kích thích và vector vàng cho testbench.

Đường số học của phần cứng (số nguyên hoàn toàn):
    acc[c] = SUM_d ( x_int[d] * w_int[c][d] ) + b_int[c]
    class  = argmax_c acc[c]
với
    x_int = clip(round(x / s_x))   , s_x = max|x_train| / (2^(XBITS-1) - 1)
    w_int = clip(round(w / s_w))   , s_w = max|W_train| / (2^(WBITS-1) - 1)   <- MỘT hệ số (tensor_max)
    b_int = round(intercept / (s_x * s_w))

Vì s_w là MỘT hệ số chung cho cả ba lớp, argmax trên `acc` số nguyên tương đương argmax trên
điểm số thực. Đây là lý do RTL dùng `tensor_max` chứ không phải `row_mse` (row_mse cho mỗi lớp
một hệ số khác nhau nên không so sánh trực tiếp được).

Script tự KIỂM CHỨNG điều đó: so argmax số nguyên với argmax thực trên toàn bộ tập test.
"""
import json, os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

CACHE = "/mnt/user-data/uploads/1.EEG_Datasets/_rivf2026_cache"
OUTDIR = "/home/claude/kgcc/ed/rtl_data"
FOLD_SUBJECT = 1            # fold 0 của LOSO: giữ lại subject 1 làm test
N_STIM = 256                # số cửa sổ đưa vào testbench

M32 = ['FP1','FP2','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4','FT8','T7','C3','CZ','C4','T8',
       'TP7','CP3','CPZ','CP4','TP8','P7','P3','PZ','P4','P8','PO7','PO3','PO4','PO8','OZ']
SUBSETS = {1: ['T7'], 4: ['T7','T8','P7','P8'],
           8: ['T7','T8','P7','P8','F3','F4','O1','O2'], 32: M32, 62: None}
# (tên, số kênh, WBITS, XBITS) — khớp các điểm trên đường Pareto của bài
CONFIGS = [("C1_1ch_w2",   1, 2, 10),
           ("C2_4ch_w4",   4, 4, 10),
           ("C3_8ch_w6",   8, 6, 10),
           ("C4_32ch_w6", 32, 6, 10),
           ("C5_62ch_w8", 62, 8, 10)]


def twos(v, bits):
    return format(int(v) & ((1 << bits) - 1), f"0{(bits + 3) // 4}x")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    CH = json.load(open(f"{CACHE}/info.json"))["channels"]
    m = np.load(f"{CACHE}/meta.npz")
    y, subj = m["label"].astype(np.int8), m["subject"]
    X = np.load(f"{CACHE}/X_de_movingAve.npy", mmap_mode="r")
    manifest = []

    for name, nc, wb, xb in CONFIGS:
        chans = CH if SUBSETS[nc] is None else SUBSETS[nc]
        cols = [CH.index(c) * 5 + b for c in chans for b in range(5)]
        A = np.asarray(X[:, cols], dtype=np.float64)
        for u in np.unique(subj):                      # z-score theo subject (giao thức P1)
            k = subj == u
            a = A[k]; A[k] = (a - a.mean(0)) / (a.std(0) + 1e-8)
        te = subj == FOLD_SUBJECT; tr = ~te
        L = LDA().fit(A[tr], y[tr])
        W, bvec = L.coef_.astype(np.float64), L.intercept_.astype(np.float64)
        D, C = W.shape[1], W.shape[0]

        s_x = np.abs(A[tr]).max() / (2 ** (xb - 1) - 1)
        s_w = np.abs(W).max() / (2 ** (wb - 1) - 1)
        w_int = np.round(W / s_w).clip(-2 ** (wb - 1), 2 ** (wb - 1) - 1).astype(np.int64)
        b_int = np.round(bvec / (s_x * s_w)).astype(np.int64)

        Xte = A[te]
        x_int = np.round(Xte / s_x).clip(-2 ** (xb - 1), 2 ** (xb - 1) - 1).astype(np.int64)
        acc = x_int @ w_int.T + b_int                   # đúng phép toán của RTL
        pred_int = L.classes_[np.argmax(acc, 1)]
        # Tham chiếu ĐÚNG: cùng mô hình đã lượng tử hoá, nhưng tính bằng số thực.
        # Nếu hai cái này khớp 100% thì đường fixed-point của RTL là tái lập chính xác,
        # và mọi sai khác so với mô hình gốc đã được quy về bước lượng tử hoá (đã đo riêng).
        deq = (x_int * s_x) @ (w_int * s_w).T + b_int * (s_x * s_w)   # bias ĐÃ làm tròn
        pred_deq = L.classes_[np.argmax(deq, 1)]
        pred_ref = L.classes_[np.argmax(Xte @ W.T + bvec, 1)]   # mô hình gốc, chưa lượng tử
        yte = y[te]
        acc_int = float((pred_int == yte).mean())
        acc_flt = float((pred_ref == yte).mean())
        agree = float((pred_int == pred_deq).mean())            # <- phép kiểm của fixed-point
        # ảnh hưởng riêng của việc làm tròn bias, tách khỏi phép kiểm trên
        deq_exact = (x_int * s_x) @ (w_int * s_w).T + bvec
        bias_effect = float((np.argmax(deq, 1) != np.argmax(deq_exact, 1)).mean())
        # Hai phep kiem THUC SU co the lam hong RTL:
        #  (a) tran thanh ghi tich luy  (b) hoa argmax (RTL pha hoa bang chi so nho nhat)
        amax = int(np.abs(acc).max())
        srt = np.sort(acc, 1)
        n_tie = int((srt[:, -1] == srt[:, -2]).sum())

        # bề rộng thanh ghi tích luỹ: đủ cho trường hợp xấu nhất TUYỆT ĐỐI + bias
        need = int(np.ceil(np.log2(D * (2 ** (xb - 1)) * (2 ** (wb - 1)) + abs(b_int).max() + 1))) + 1
        accw = max(need, 16)
        headroom = (2 ** (accw - 1) - 1) / max(amax, 1)

        d = f"{OUTDIR}/{name}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/weights.hex", "w") as f:        # thứ tự: lớp 0 hết D, rồi lớp 1, lớp 2
            for c in range(C):
                for k in range(D):
                    f.write(twos(w_int[c, k], wb) + "\n")
        for c in range(C):                              # ROM riêng cho từng lớp (3 cổng đọc/chu kỳ)
            with open(f"{d}/w_c{c}.hex", "w") as f:
                for k in range(D):
                    f.write(twos(w_int[c, k], wb) + "\n")
        with open(f"{d}/bias.hex", "w") as f:
            for c in range(C):
                f.write(twos(b_int[c], accw) + "\n")
        n = min(N_STIM, len(x_int))
        with open(f"{d}/stimulus.hex", "w") as f:
            for i in range(n):
                for k in range(D):
                    f.write(twos(x_int[i, k], xb) + "\n")
        with open(f"{d}/golden.txt", "w") as f:         # 0=negative 1=neutral 2=positive
            for i in range(n):
                f.write(f"{int(np.argmax(acc[i]))}\n")
        with open(f"{d}/params.vh", "w") as f:
            f.write(f"// sinh tu dong boi gen_rtl_data.py — {name}\n"
                    f"`define DIM    {D}\n`define NCLASS {C}\n`define WBITS  {wb}\n"
                    f"`define XBITS  {xb}\n`define ACCW   {accw}\n`define NSTIM  {n}\n"
                    f"`define ADDRW  {max(1, int(np.ceil(np.log2(D))) if D > 1 else 1)}\n")

        row = dict(name=name, channels=nc, dims=D, wbits=wb, xbits=xb, accw=accw,
                   weight_bytes=D * C * wb / 8, n_stim=n,
                   acc_integer=acc_int, acc_float=acc_flt, agreement=agree, bias_round_effect=bias_effect,
                   acc_absmax=amax, acc_limit=2 ** (accw - 1) - 1, headroom=headroom, n_tie=n_tie,
                   s_x=s_x, s_w=s_w, bias_int=b_int.tolist())
        manifest.append(row)
        print(f"{name:<12} D={D:>3} w={wb}b x={xb}b ACCW={accw:>2}  "
              f"int={100*acc_int:.2f}% (goc {100*acc_flt:.2f}%)  |acc|max={amax:>9} / "
              f"gioi han {2**(accw-1)-1:>10} (du {headroom:5.1f}x)  hoa={n_tie}", flush=True)

    json.dump(manifest, open(f"{OUTDIR}/manifest.json", "w"), indent=1)
    ovf = [r["name"] for r in manifest if r["acc_absmax"] > r["acc_limit"]]
    print()
    if ovf:
        print("LOI: TRAN thanh ghi tich luy o " + str(ovf))
    else:
        print(f"OK: khong cau hinh nao tran thanh ghi tich luy "
              f"(du it nhat {min(r['headroom'] for r in manifest):.1f}x).")
    print(f"OK: tong so cua so bi hoa argmax = {sum(r['n_tie'] for r in manifest)} "
          f"(RTL pha hoa bang chi so nho nhat, giong numpy.argmax).")
    print(f"Khop voi mo hinh da luong tu tinh bang float: "
          f"{100*min(r['agreement'] for r in manifest):.2f}%–100.00%. Phan le nam o PHIA THAM CHIEU "
          f"(tich luy float64 lam tron khac thu tu), khong phai o duong so nguyen — "
          f"duong so nguyen la chinh xac theo dinh nghia.")


if __name__ == "__main__":
    main()
