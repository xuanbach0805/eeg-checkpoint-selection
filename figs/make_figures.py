"""Sinh 4 hinh cho ban nop BSPC. Chay: python3 make_figures.py

Xuat ca .pdf (vector, de nop) va .png (300 dpi, de xem nhanh) vao ./figs/
Khong co truc y doi (dual axis) o bat ky hinh nao — dai luong khac thang do thi tach panel.
Bang mau: slot 1 blue #2a78d6, slot 2 orange #eb6834 (da qua validate all-pairs),
ramp sequential mot sac blue 100->700 cho heatmap. Moi chuoi deu co marker rieng
de hinh con doc duoc khi in trang den.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

UP = "/mnt/user-data/uploads/10.phD/1.EEG/2026/Journal2026_pilot/BSPC2026/results"
LOC = "results"
OUT = "figs"
os.makedirs(OUT, exist_ok=True)

MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM          # Elsevier: 1 cot 90 mm, 2 cot 190 mm

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "grid.color": "#e2e1dd", "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  {OUT}/{name}.pdf + .png")


def load(sub):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{UP}/{sub}/fold*.json"))]


# ============================== Figure 1 ==============================
# (a) 15 quy dao 90 epoch (14 source) voi S_last va S_oracle
# (b) 4 tieu chi chon checkpoint, tung fold (12 source)
def fig1():
    A, E = load("kgcc"), load("kgcc_ea3")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.5 * MM * 25.4 * 0.4 + 1.9),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # --- panel a ---
    H = np.array([d["history"] for d in A]) * 100
    ep = np.arange(1, H.shape[1] + 1)
    for h in H:
        ax1.plot(ep, h, color=INK3, lw=0.45, alpha=0.75)
    ax1.plot(ep, H.mean(0), color=INK, lw=1.6, zorder=5, label="mean over 15 participants")

    bo = [d["acc_best_epoch"] for d in A]
    ax1.scatter(bo, [H[i, bo[i] - 1] for i in range(15)], s=17, facecolor=ORANGE,
                edgecolor="white", linewidth=0.5, zorder=7, marker="o",
                label=r"$S_{\mathrm{oracle}}$ (chosen on the test participant)")
    ax1.scatter([ep[-1]] * 15, H[:, -1], s=17, facecolor=BLUE, edgecolor="white",
                linewidth=0.5, zorder=7, marker="s", label=r"$S_{\mathrm{last}}$ (final epoch)")

    # nhan trung binh dat NGOAI vung du lieu (x > 90) de khong de len quy dao
    mo, ml = H[np.arange(15), np.array(bo) - 1].mean(), H[:, -1].mean()
    ax1.annotate(f"{mo:.2f}%", xy=(96, mo), color=ORANGE, fontsize=7.5, va="center",
                 ha="left", weight="bold")
    ax1.annotate(f"{ml:.2f}%", xy=(96, ml), color=BLUE, fontsize=7.5, va="center",
                 ha="left", weight="bold")
    ax1.annotate("", xy=(93, ml), xytext=(93, mo),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax1.text(94, ml - 8.5, f"{mo - ml:.2f} points\n15/15 folds",
             ha="left", va="top", fontsize=7, color=INK, linespacing=1.3)

    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("Accuracy on the held-out participant (%)")
    ax1.set_title("(a) Accuracy trajectories, 14 source participants", loc="left")
    ax1.set_xlim(0, 128); ax1.set_ylim(30, 100)
    ax1.set_xticks([0, 20, 40, 60, 80])
    ax1.yaxis.grid(True); ax1.set_axisbelow(True)
    lg = ax1.legend(loc="lower left", frameon=True, handletextpad=0.4, borderpad=0.3,
                    framealpha=0.92, edgecolor="none", facecolor="white")
    lg.set_zorder(20)

    # --- panel b ---
    crit = [(r"$S_{\mathrm{srcval}}$", [d["acc_at_srcval"] for d in E]),
            (r"$S_{\mathrm{random}}$", [np.mean(d["history_target"]) for d in E]),
            (r"$S_{\mathrm{last}}$",   [d["acc_last"] for d in E]),
            (r"$S_{\mathrm{oracle}}$", [d["acc_best_oracle"] for d in E])]
    V = np.array([[100 * v for v in vals] for _, vals in crit])
    x = np.arange(len(crit))
    for j in range(15):                                   # noi tung fold
        ax2.plot(x, V[:, j], color=INK3, lw=0.4, alpha=0.7, zorder=2)
    cols = [INK2, INK2, INK2, ORANGE]
    mk = ["o", "^", "s", "D"]
    for i in range(len(crit)):
        ax2.scatter([x[i]] * 15, V[i], s=13, facecolor=cols[i], edgecolor="white",
                    linewidth=0.4, marker=mk[i], zorder=4)
        ax2.plot([x[i] - 0.26, x[i] + 0.26], [V[i].mean()] * 2, color=INK, lw=1.8, zorder=6)
        # nhan trung binh xep thanh mot hang o dinh panel -> khong bao gio de len mark
        ax2.text(x[i], 97.2, f"{V[i].mean():.2f}", ha="center", va="center", fontsize=7.5,
                 weight="bold", color=ORANGE if i == 3 else INK)
    ax2.set_xticks(x); ax2.set_xticklabels([c for c, _ in crit])
    ax2.set_xlim(-0.5, len(crit) - 0.5); ax2.set_ylim(30, 101)
    ax2.set_ylabel("Accuracy on the held-out participant (%)")
    ax2.set_title("(b) Checkpoint criteria, 12 source participants", loc="left")
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)
    ax2.text(0.5, 32, "admissible without target labels", ha="center", fontsize=6.5,
             color=INK2, style="italic")
    ax2.plot([-0.35, 2.35], [35.5, 35.5], color=INK2, lw=0.6)
    ax2.text(3.0, 32, "not admissible", ha="center", fontsize=6.5, color=ORANGE, style="italic")
    ax2.plot([2.65, 3.35], [35.5, 35.5], color=ORANGE, lw=0.6)

    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig1_checkpoint_selection")


# ============================== Figure 2 ==============================
def fig2():
    G = json.load(open(f"{LOC}/gta_matrix.json"))
    chans = ["1", "2", "4", "8", "14", "32", "62"]
    bits = ["2", "3", "4", "6", "8", "None"]
    M = np.array([[100 * np.mean(G[c]["cells"][f"row_mse|{b}"]["acc"]) for b in bits]
                  for c in chans])
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.92))
    im = ax.imshow(M, cmap=CMAP, aspect="auto", origin="lower", vmin=M.min(), vmax=M.max())
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(["2", "3", "4", "6", "8", "float"])
    ax.set_yticks(range(len(chans))); ax.set_yticklabels(chans)
    ax.set_xlabel("Weight word length (bits)")
    ax.set_ylabel("Electrodes")
    ax.set_title("Cross-subject accuracy (%), protocol P1", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(width=0.6, color=INK2)
    # danh dau diem van hanh, khong ghi so vao moi o (Bang 2 da co so)
    # chi danh dau bang vong tron; chu thich hinh giai thich — tranh chu mau tren o toi
    ax.scatter([3], [3], s=52, facecolor="none", edgecolor=ORANGE, linewidth=1.5, zorder=5)
    save(fig, "fig2_channel_wordlength_plane")


# ============================== Figure 3 ==============================
def fig3():
    L = json.load(open(f"{LOC}/linear_matched.json"))
    C = json.load(open(f"{LOC}/ea2_ptq_kgcc.json"))
    pts = []
    for ch, o in L.items():
        D = o["dims"]
        for b in (2, 3, 4, 6, 8):
            key = f"row_mse|{b}"
            if key in o["cells"]:
                pts.append((D * 3 * b / 8, 100 * np.mean(o["cells"][key]["acc"]), int(ch), b))
    pts.sort()
    front, best = [], -1e9
    for by, ac, ch, b in pts:
        if ac > best:
            front.append((by, ac, ch, b)); best = ac
    CB = {2: 29184, 3: 43200, 4: 57216, 6: 85248, 8: 113280}
    cp = [(CB[b], 100 * np.mean(C[f"row_mse|{b}"])) for b in (2, 3, 4, 6, 8) if f"row_mse|{b}" in C]

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.80))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=7, facecolor=INK3,
               edgecolor="none", alpha=0.55, zorder=2)
    ax.plot([p[0] for p in front], [p[1] for p in front], color=BLUE, lw=1.3,
            marker="o", ms=3.8, mec="white", mew=0.4, zorder=4,
            label="Linear (LDA), Pareto front")
    ax.plot([p[0] for p in cp], [p[1] for p in cp], color=ORANGE, lw=1.3,
            marker="s", ms=4, mec="white", mew=0.4, zorder=4, label="PR-PL (convolutional)")
    ax.axvline(43200, color=ORANGE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(3500, 58, "compression floor, 43.2 kB", rotation=90, ha="center",
            va="center", fontsize=6.5, color=ORANGE)
    ax.annotate("", xy=(41000, 58), xytext=(4800, 58),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
    # diem van hanh dung trong Bang 6 (4 kenh @ 6 bit)
    m45 = min(front, key=lambda t: abs(t[0] - 45))
    ax.scatter([m45[0]], [m45[1]], s=46, facecolor="none", edgecolor=INK,
               linewidth=0.9, zorder=6)
    ax.annotate(f"{m45[2]} ch, {m45[3]} bit\n{m45[0]:g} bytes", xy=(m45[0], m45[1]),
                xytext=(m45[0] * 0.9, m45[1] - 9.5), fontsize=6.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    ax.set_xscale("log")
    ax.set_xlabel("Weight footprint (bytes, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy versus weight footprint, protocol P2", loc="left")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.5, borderpad=0.2)
    save(fig, "fig3_accuracy_vs_footprint")


# ============================== Figure 4 ==============================
def fig4():
    S = json.load(open("ed/out/ed_synthesis.json"))
    c = S["configs"]
    D = np.array([r["DIM"] for r in c]); LE = np.array([r["logic_elements"] for r in c])
    FM = np.array([r["Fmax_MHz"] for r in c]); MU = np.array([r["mult_9bit"] for r in c])
    WB = [r["WBITS"] for r in c]; CH = [r["channels"] for r in c]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.15))

    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(D); ax.set_xticklabels([str(d) for d in D])
        ax.set_xlim(3.6, 430)
        ax.minorticks_off()
        ax.set_xlabel("Feature dimensionality $D$")
        ax.yaxis.grid(True); ax.set_axisbelow(True)

    # --- (a) logic, phan biet theo viec co dung bo nhan cung hay khong ---
    a1.plot(D, LE, color=INK3, lw=1.0, zorder=3)
    m0, m6 = MU == 0, MU > 0
    a1.scatter(D[m0], LE[m0], s=42, facecolor="white", edgecolor=BLUE, linewidth=1.3,
               marker="o", zorder=5, label="multipliers built in logic (0 used)")
    a1.scatter(D[m6], LE[m6], s=36, facecolor=BLUE, edgecolor="white", linewidth=0.6,
               marker="s", zorder=5, label="mapped to hard multipliers (6 used)")
    off = {0: (13, 0, "left", "center")}          # 1-ch nhan sang phai, khoi bi cat
    for i2 in range(len(c)):
        dx, dy, ha, va = off.get(i2, (0, -13, "center", "top"))
        a1.annotate(f"{CH[i2]} ch, {WB[i2]} b", xy=(D[i2], LE[i2]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=6.2, color=INK2)
    a1.annotate("", xy=(40, 285), xytext=(20, 330),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0,
                                connectionstyle="arc3,rad=-0.35"))
    a1.text(62, 620, "twice the dimensionality,\nfewer logic elements",
            fontsize=6.6, color=ORANGE, ha="left", va="center", style="italic",
            linespacing=1.3)
    a1.set_ylabel("Logic elements")
    a1.set_title("(a) Logic occupancy", loc="left")
    a1.set_ylim(90, 1080)
    a1.legend(loc="upper left", frameon=False, handletextpad=0.35, borderpad=0.15,
              fontsize=6.4)

    # --- (b) Fmax ---
    a2.axhspan(91.0, 93.5, color=BLUE, alpha=0.10, zorder=1)
    a2.plot(D, FM, color=BLUE, lw=1.3, marker="o", ms=4.5, mec="white", mew=0.5, zorder=4)
    for i2 in range(len(c)):
        a2.annotate(f"{FM[i2]:.1f}", xy=(D[i2], FM[i2]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=6.4, color=INK2)
    a2.text(400, 128, "critical path is the\naccumulator adder;\nACCW grows as $\\log D$",
            fontsize=6.6, color=INK2, ha="right", va="top", style="italic", linespacing=1.35)
    a2.set_ylabel(r"$F_{\max}$ (MHz)")
    a2.set_title("(b) Clock frequency", loc="left")
    a2.set_ylim(82, 165)

    fig.subplots_adjust(wspace=0.24)
    save(fig, "fig4_hardware_cost")


if __name__ == "__main__":
    print("Sinh hinh:")
    fig1(); fig2(); fig3(); fig4()
    print("Xong.")
