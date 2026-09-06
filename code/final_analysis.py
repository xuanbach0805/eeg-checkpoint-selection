"""Tổng hợp cuối: KGCC/PR-PL (15 fold thật) vs bộ phân loại tuyến tính, cùng giao thức GT-B."""
import json
import numpy as np
from scipy.stats import wilcoxon

K = json.load(open("/home/claude/kgcc/results/ea2_ptq_kgcc.json"))
L = json.load(open("/home/claude/kgcc/results/linear_matched.json"))
FOLD = [json.load(open(f"/home/claude/kgcc/results/kgcc/fold{f:02d}.json")) for f in range(15)] \
    if False else None
BITS = [2, 3, 4, 6, 8, None]
N = 15

# tu 15 file fold JSON (da doc qua device_bash, dan vao day de khong phai stage lai)
ACC_LAST = np.array([0.7445492044784915, 0.7430760164997053, 0.7265763111373011,
                     0.7309958750736594, 0.6202121390689452, 0.8149675898644667,
                     0.6823806717737183, 0.7377725397760754, 0.6175604007071303,
                     0.706246317030053, 0.835592221567472, 0.7919858573954036,
                     0.7383618149675899, 0.921921037124337, 0.5091337654684738])
ACC_BEST = np.array([0.870654095462581, 0.7754861520329994, 0.8603417796110784,
                     0.85592221567472, 0.7271655863288156, 0.8238067177371833,
                     0.8361814967589865, 0.8677077195050088, 0.7365939893930465,
                     0.796405421331762, 0.8992339422510313, 0.9065998821449617,
                     0.8485562757807896, 0.9619917501473187, 0.7955215085444903])
ACC_M10 = np.array([0.7437831467295226, 0.7314967589864466, 0.7265763111373011,
                    0.7211844431349441, 0.6084855627578079, 0.7804655274012965,
                    0.6926340601060696, 0.7492928697701826, 0.6175604007071304,
                    0.7141426045963465, 0.8393341190335887, 0.7580141426045964,
                    0.7532999410724808, 0.9241897466116675, 0.5389216263995286])
BEST_EPOCH = [49, 6, 15, 64, 27, 4, 4, 41, 19, 12, 4, 51, 29, 43, 2]


def kgcc_fp(b):
    nw = 1440 + 18432 + 73728 + 16384 + 1024 + 1024 + 96
    nb = 32 + 64 + 128 + 32 + 32
    return nw * (32 if b is None else b) / 8 + nb * 4


def lin(nc, b, mode="row_mse"):
    return np.array(L[str(nc)]["cells"][f"{mode}|{b}"]["acc"])


def lin_fp(nc, b):
    d = L[str(nc)]["dims"]
    return d * 3 * (32 if b is None else b) / 8


def ci(a):
    m = 100 * a.mean(); se = 100 * a.std(ddof=1) / np.sqrt(len(a))
    return m, m - 1.96 * se, m + 1.96 * se


out = []
p = out.append
p("# E-A + E-A2 — KẾT QUẢ CUỐI, 15/15 fold. 02/09/2026")
p("")
p("KGCC/PR-PL chạy thật trên RTX 2050, 90 epoch/fold, hyperparams mặc định của tác giả.")
p("Giao thức GT-B (của chính họ): de_LDS, session 1, lưới 9×9, min-max theo session, LOSO 15.")
p("Chance 33.33 · majority toàn cục 34.47. Nguồn: `results/kgcc/fold00..14.json`,")
p("`results/ea2_ptq_kgcc.json`, `results/linear_matched.json`.")
p("")

p("## 1. KGCC/PR-PL — ba cách báo cáo cùng một lần chạy")
p("")
p("| chỉ số | mean | SD | CI 95% |")
p("|---|---|---|---|")
for name, a in (("acc_last (trung thực)", ACC_LAST),
                ("acc_best (giao thức tác giả)", ACC_BEST),
                ("mean_last10", ACC_M10)):
    m, lo, hi = ci(a)
    p(f"| {name} | **{m:.2f}** | {100*a.std(ddof=1):.2f} | [{lo:.2f}, {hi:.2f}] |")
st = wilcoxon(ACC_BEST, ACC_LAST)
p("")
p(f"**Chênh acc_best − acc_last = {100*(ACC_BEST.mean()-ACC_LAST.mean()):.2f} điểm** "
  f"(Wilcoxon ghép cặp p = {st.pvalue:.2e}, tăng ở {int((ACC_BEST>ACC_LAST).sum())}/15 subject).")
p(f"Epoch \"tốt nhất\" rải {min(BEST_EPOCH)}→{max(BEST_EPOCH)} "
  f"(trung vị {int(np.median(BEST_EPOCH))}) — không có quy luật, tức không thể đoán bằng bất kỳ")
p("tiêu chí nào không dùng nhãn của subject target.")
p("")

p("## 2. E-A2 — KGCC sau PTQ (cùng quantizer với mô hình tuyến tính)")
p("")
p("| bit | byte trọng số | row_mse | tensor_max |")
p("|---|---|---|---|")
for b in BITS:
    r1 = 100 * np.mean(K[f"row_mse|{b}"]); r2 = 100 * np.mean(K[f"tensor_max|{b}"])
    p(f"| {b if b else 'float'} | {kgcc_fp(b):,.0f} | {r1:.2f} | {r2:.2f} |")
p("")
p("CNN sống được tới **3 bit** với `row_mse` (71.35%) rồi **sụp ở 2 bit** (49.17%).")
p("Với `tensor_max` — quantizer rẻ nhất trên phần cứng, một hệ số cho cả tensor — nó chỉ sống")
p("tới **4 bit** (71.05%) và sụp ở 3 bit (52.96%). Mô hình tuyến tính chịu được 2 bit.")
p(f"=> **Footprint nhỏ nhất còn dùng được của CNN ≈ {kgcc_fp(3):,.0f} byte** (3 bit, row_mse).")
p("")

p("## 3. So sánh đầu-đối-đầu, 15 fold, cùng giao thức")
p("")
p("| mô hình | byte | acc % | CI 95% |")
p("|---|---|---|---|")
rows = [("KGCC — acc_best *(giao thức tác giả)*", kgcc_fp(None), ACC_BEST),
        ("KGCC — acc_last *(trung thực)*", kgcc_fp(None), ACC_LAST),
        ("KGCC — PTQ 4 bit", kgcc_fp(4), np.array(K["row_mse|4"])),
        ("KGCC — PTQ 3 bit *(nhỏ nhất còn chạy)*", kgcc_fp(3), np.array(K["row_mse|3"])),
        ("KGCC — PTQ 2 bit *(đã sụp)*", kgcc_fp(2), np.array(K["row_mse|2"])),
        ("LDA 4 kênh @6 bit", lin_fp(4, 6), lin(4, 6)),
        ("LDA 2 kênh @4 bit", lin_fp(2, 4), lin(2, 4)),
        ("LDA 1 kênh @2 bit", lin_fp(1, 2), lin(1, 2))]
for name, by, a in rows:
    m, lo, hi = ci(a)
    p(f"| {name} | {by:,.2f} | {m:.2f} | [{lo:.2f}, {hi:.2f}] |")
p("")

p("### Wilcoxon ghép cặp (n = 15, sàn p hai phía = 6.1e-05)")
p("")
p("| so sánh | chênh | p | KGCC thắng |")
p("|---|---|---|---|")
for kname, ka in (("acc_last", ACC_LAST), ("acc_best", ACC_BEST),
                  ("PTQ 3 bit", np.array(K["row_mse|3"]))):
    for lname, nc, b in (("LDA 4 kênh @6 bit", 4, 6), ("LDA 2 kênh @4 bit", 2, 4),
                         ("LDA 1 kênh @2 bit", 1, 2)):
        a = lin(nc, b); d = ka - a; s = wilcoxon(ka, a)
        p(f"| KGCC {kname} vs {lname} | {100*d.mean():+.2f} | {s.pvalue:.4f} | "
          f"{int((d>0).sum())}/15 |")
p("")

r3 = np.array(K["row_mse|3"]); l46 = lin(4, 6)
p("## 4. KẾT LUẬN")
p("")
p(f"- Tỉ lệ bộ nhớ giữa cấu hình CNN nhỏ nhất còn chạy ({kgcc_fp(3):,.0f} B) và LDA 4 kênh @6 bit "
  f"({lin_fp(4,6):.0f} B) = **{kgcc_fp(3)/lin_fp(4,6):,.0f}×**.")
s = wilcoxon(r3, l46)
p(f"- Cái giá đó mua được **{100*(r3.mean()-l46.mean()):+.2f} điểm** "
  f"(p = {s.pvalue:.4f}, thắng {int((r3>l46).sum())}/15 subject).")
s = wilcoxon(ACC_LAST, l46)
p(f"- Ở full precision (449,664 B = **{kgcc_fp(None)/lin_fp(4,6):,.0f}×**), đo trung thực: "
  f"**{100*(ACC_LAST.mean()-l46.mean()):+.2f} điểm** (p = {s.pvalue:.4f}).")
s = wilcoxon(ACC_BEST, l46)
p(f"- Nhưng theo giao thức của tác giả (chọn epoch trên tập test): "
  f"**{100*(ACC_BEST.mean()-l46.mean()):+.2f} điểm** (p = {s.pvalue:.4f}).")
p("")
p("> Nói cách khác: **hơn một nửa khoảng cách biểu kiến giữa mô hình SOTA và một bộ phân loại")
p("> tuyến tính 45 byte là do bước chọn checkpoint bằng chính subject test, không phải do kiến trúc.**")

txt = "\n".join(out) + "\n"
open("/home/claude/kgcc/RESULTS_final_kgcc_vs_linear.md", "w").write(txt)
print(txt)
