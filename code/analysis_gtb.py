"""E-F (thong ke) + E-H (quantizer thu hai) + E-E (ngan sach activation/latency) tren GT-B."""
import json, numpy as np
from scipy.stats import wilcoxon

R = json.load(open("/home/claude/kgcc/results/linear_matched.json"))
BITS = [2, 3, 4, 6, 8, None]
MODES = ["row_max", "tensor_max", "row_p999", "row_p99", "row_mse"]
CHS = sorted(R, key=int)
L = []
p = L.append

p("# GT-B — thong ke, quantizer thu hai, ngan sach tinh toan. Chay 01/09/2026")
p("")
p("Nguon: `results/linear_matched.json` (LDA, LOSO 15 subject, de_LDS, session 1,")
p("min-max theo session tren luoi 9x9 — dung giao thuc KGCC). Chance 33.33 · majority 34.47.")
p("")

# ---------- E-H: bang row_mse va tensor_max ----------
for md in ["row_mse", "tensor_max"]:
    p(f"## Bang accuracy (%) — quantizer `{md}`")
    p("| kenh | chieu | " + " | ".join(str(b) if b else "float" for b in BITS) + " |")
    p("|---" * (len(BITS) + 2) + "|")
    for nc in CHS:
        c = R[nc]["cells"]
        v = " | ".join(f"{100*np.mean(c[f'{md}|{b}']['acc']):.2f}" for b in BITS)
        p(f"| {nc} | {R[nc]['dims']} | {v} |")
    p("")

# ---------- kiem don dieu ----------
p("## Kiem don dieu theo bit (dung sai 0.5 diem)")
p("| quantizer | hang vi pham |")
p("|---|---|")
for md in MODES:
    bad = []
    for nc in CHS:
        c = R[nc]["cells"]
        s = [100 * np.mean(c[f"{md}|{b}"]["acc"]) for b in [2, 3, 4, 6, 8]]
        if any(s[i + 1] < s[i] - 0.5 for i in range(len(s) - 1)):
            bad.append(nc)
    p(f"| {md} | {', '.join(bad) if bad else '**KHONG CO — sach**'} |")
p("")

# ---------- bien do giua cac quantizer ----------
p("## Bien do accuracy giua 5 quantizer (diem)")
p("| kenh | @2 bit | @3 bit | @4 bit | @6 bit | @8 bit |")
p("|---|---|---|---|---|---|")
for nc in CHS:
    c = R[nc]["cells"]
    row = []
    for b in [2, 3, 4, 6, 8]:
        v = [100 * np.mean(c[f"{md}|{b}"]["acc"]) for md in MODES]
        row.append(f"{max(v)-min(v):.2f}")
    p(f"| {nc} | " + " | ".join(row) + " |")
p("")
p("Cung ket luan nhu GT-A: **duoi 4 bit, quy tac scaling quan trong ngang so bit**.")
p("")

# ---------- E-F: CI + Wilcoxon tren duong Pareto ----------
p("## E-F — Pareto GT-B voi CI 95% va kiem dinh (row_mse, n=15 fold)")
pts = []
for nc in CHS:
    d = R[nc]["dims"]; c = R[nc]["cells"]
    for b in [2, 3, 4, 6, 8]:
        a = np.array(c[f"row_mse|{b}"]["acc"])
        pts.append((d * 3 * b / 8, a, f"{nc} kenh @ {b} bit"))
pareto, best = [], -1
for by, a, cfg in sorted(pts, key=lambda x: x[0]):
    if a.mean() > best:
        best = a.mean(); pareto.append((by, a, cfg))
ref = np.array(R["4"]["cells"]["row_mse|6"]["acc"])          # dinh cua bang
p("| byte | acc | CI 95% | cau hinh | vs dinh (4 kenh @6b) |")
p("|---|---|---|---|---|")
for by, a, cfg in pareto:
    m = 100 * a.mean(); se = 100 * a.std(ddof=1) / np.sqrt(len(a))
    lo, hi = m - 1.96 * se, m + 1.96 * se
    if np.allclose(a, ref):
        cmp = "— (chinh no)"
    else:
        st = wilcoxon(a, ref)
        cmp = f"{100*(a.mean()-ref.mean()):+.2f}, p={st.pvalue:.3f}"
    p(f"| {by:.2f} | {m:.2f} | [{lo:.2f}, {hi:.2f}] | {cfg} | {cmp} |")
p("")
p("**Luu y:** n=15, Wilcoxon hai phia co san p nho nhat = 6.1e-05; voi cac o sat nhau thi")
p("phai dua vao CI chu khong phai chi p (bai hoc tu bai RIVF).")
p("")

# ---------- E-E: ngan sach tinh toan ----------
p("## E-E — Ngan sach MAC va activation cho mot cua so 1 giay")
p("")
p("KGCC/PR-PL, duong inference `fea_extrator_f` + U + stored_mat, input 5x9x9:")
layers = [("conv1 5->32, 9x9",  5*32*9*81,   32*81),
          ("conv2 32->64, 9x9", 32*64*9*81,  64*81),
          ("pool 9x9->4x4",     0,           64*16),
          ("conv3 64->128, 4x4",64*128*9*16, 128*16),
          ("pool2 4x4->2x2",    0,           128*4),
          ("fc1 512->32",       512*32,      32),
          ("fc2 32->32",        32*32,       32),
          ("U (32x32)",         32*32,       32),
          ("stored_mat (32x3)", 32*3,        3)]
p("| lop | MAC | activation dau ra (gia tri) |")
p("|---|---|---|")
for n, mac, act in layers:
    p(f"| {n} | {mac:,} | {act:,} |")
tm = sum(m for _, m, _ in layers); pk = max(a for _, _, a in layers)
p(f"| **tong** | **{tm:,}** | dinh {pk:,} |")
p("")
p(f"Peak activation @8 bit = {pk/1024:.2f} KB; can double-buffer 2 lop lien tiep lon nhat")
mx2 = sorted([a for _, _, a in layers])[-2:]
p(f"= {sum(mx2)} gia tri = {sum(mx2)/1024:.2f} KB @8 bit.")
p("")
p("Mo hinh tuyen tinh (D chieu, 3 lop):")
p("| kenh | chieu | MAC | activation | trong so @4 bit |")
p("|---|---|---|---|---|")
for nc in CHS:
    d = R[nc]["dims"]
    p(f"| {nc} | {d} | {d*3:,} | {d:,} | {d*3*4/8:.1f} B |")
p("")
p(f"**Ti le MAC:** KGCC {tm:,} vs 4 kenh tuyen tinh 60 = **{tm/60:,.0f}x**.")
p(f"**Ti le trong so @8 bit:** 112,416 B vs {R['4']['dims']*3} B = "
  f"**{112416/(R['4']['dims']*3):,.0f}x**.")
p("")
p("Ket luan giu nguyen tu `RESULTS_fit.md`: voi KGCC do tre KHONG phai rang buoc")
p("(1.40 ms @100MHz, 20 DSP — nhanh hon real-time ~700 lan); **rang buoc la bo nho trong so**,")
p("va 83.46% trong so nam o conv1-3 von doc lap voi so kenh.")

open("/home/claude/kgcc/RESULTS_gtb_stats.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
