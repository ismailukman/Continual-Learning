"""
Generate journal-quality vector figures (PDF) for the continual-learning report.

Figures are reproduced directly from the recorded experiment results so the
report is self-contained. Output is vector PDF with a consistent serif style
matching the LaTeX document.

Run from the repo root:
    py report/make_figures.py
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ---------------------------------------------------------------------------
# Journal style: serif fonts (Computer Modern-ish), tight, colorblind-safe.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # embed TrueType, editable text
})

# Okabe-Ito colorblind-safe palette
C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY   = "#999999"
METHOD_COLORS = {
    "Baseline":   C_GRAY,
    "EWC":        C_BLUE,
    "Replay":     C_GREEN,
    "EWC+Replay": C_RED,
}

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Recorded results
# ---------------------------------------------------------------------------
# Experiment 1: class-incremental (2 tasks). R[i,j] = acc on test task j after
# training through task i (%). Source: report/legacy + results_20260122_115841.csv
CI = {
    "Baseline":   np.array([[100.0,   0.0], [  0.0, 100.0]]),
    "EWC":        np.array([[100.0,   0.0], [  0.0, 100.0]]),
    "Replay":     np.array([[100.0,   0.0], [ 13.25, 95.50]]),
    "EWC+Replay": np.array([[100.0,   0.0], [ 70.50, 31.00]]),
}
CI_SUMMARY = {  # method -> (avg_acc, forgetting)
    "Baseline":   (50.00, 100.00),
    "EWC":        (50.00, 100.00),
    "Replay":     (54.38,  86.75),
    "EWC+Replay": (50.75,  29.50),
}
# Experiment 2: window-based domain-incremental (3 tasks).
# Source: logs/matrices_v2_windows_20260122_123836.npz
WIN = {
    "Baseline": np.array([[54.375, 50.0, 50.0],
                          [53.25, 52.125, 52.25],
                          [55.5, 50.0, 55.25]]),
    "EWC": np.array([[51.625, 50.0, 50.0],
                     [51.0, 52.25, 50.0],
                     [55.25, 50.0, 48.125]]),
    "Replay": np.array([[65.5, 50.0, 50.0],
                        [62.625, 50.0, 59.125],
                        [62.0, 50.0, 61.125]]),
    "EWC+Replay": np.array([[57.75, 50.0, 50.0],
                            [59.125, 54.125, 56.125],
                            [55.75, 50.125, 55.625]]),
}
WIN_SUMMARY = {
    "Baseline":   (53.58,  0.50),
    "EWC":        (51.13, -0.69),
    "Replay":     (57.71,  1.75),
    "EWC+Replay": (53.83,  3.00),
}


# ---------------------------------------------------------------------------
def heatmap_grid(matrices, n_tasks, fname, panel_letters=True):
    methods = list(matrices.keys())
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.8), constrained_layout=True)
    cmap = matplotlib.colormaps["cividis"]
    im = None
    for k, (ax, m) in enumerate(zip(axes.ravel(), methods)):
        R = matrices[m]
        im = ax.imshow(R, cmap=cmap, vmin=40, vmax=100, aspect="equal")
        tag = f"({chr(97+k)}) " if panel_letters else ""
        ax.set_title(tag + m, fontweight="bold")
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))
        ax.set_xticklabels([f"$\\mathcal{{T}}_{j+1}$" for j in range(n_tasks)])
        ax.set_yticklabels([f"after $\\mathcal{{T}}_{i+1}$" for i in range(n_tasks)])
        ax.set_xlabel("evaluated on")
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                v = R[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < 72 else "black", fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("accuracy (\\%)")
    cb.outline.set_visible(False)
    save(fig, fname)


def summary_bars(summary, fname, ymax=110):
    methods = list(summary.keys())
    acc = [summary[m][0] for m in methods]
    forget = [summary[m][1] for m in methods]
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)
    b1 = ax.bar(x - w / 2, acc, w, label="Avg.\\ accuracy", color=C_BLUE,
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, forget, w, label="Forgetting", color=C_RED,
                edgecolor="black", linewidth=0.5, hatch="//")
    ax.axhline(50, ls=(0, (4, 3)), c=C_GRAY, lw=1)
    ax.text(len(methods) - 0.5, 51.5, "chance", color=C_GRAY, fontsize=8,
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha="right")
    ax.set_ylabel("percent (\\%)")
    ax.set_ylim(min(0, min(forget) - 5), ymax)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + (1.5 if h >= 0 else -3.5),
                f"{h:.1f}", ha="center", fontsize=7.5,
                va="bottom" if h >= 0 else "top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, fname)


def stability_plasticity(fname):
    methods = list(CI.keys())
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    # ideal-frontier shading
    ax.fill_between([70, 100], 70, 100, color=to_rgba(C_GREEN, 0.08), zorder=0)
    ax.text(85, 95, "ideal region", color=C_GREEN, fontsize=8, ha="center")
    ax.plot([0, 100], [100, 0], ls=":", c=C_GRAY, lw=1, zorder=1)
    for m in methods:
        s, p = CI[m][1, 0], CI[m][1, 1]
        ax.scatter(s, p, s=120, color=METHOD_COLORS[m], edgecolor="black",
                   linewidth=0.7, zorder=3)
        dx, dy = (6, 5)
        if m == "Baseline":
            dy = -14
        ax.annotate(m, (s, p), textcoords="offset points", xytext=(dx, dy),
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("Stability — final $R_{T,1}$ on Task 1 (\\%)")
    ax.set_ylabel("Plasticity — final $R_{T,T}$ on Task $T$ (\\%)")
    ax.set_xlim(-5, 108)
    ax.set_ylim(-5, 108)
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, fname)


# ---------------------------------------------------------------------------
# Improved experiment: pretrained ResNet-18, 2-task class-incremental.
# Source: logs/experiment_improved_20260603_023608.log (12/13 configs;
# combined(lambda=1000) hung and was excluded). Representative configs shown.
# ---------------------------------------------------------------------------
IMP = {
    "Baseline":          np.array([[100.0, 0.0], [0.0, 100.0]]),
    "Replay":            np.array([[100.0, 0.0], [100.0, 0.0]]),
    "LwF":               np.array([[100.0, 0.0], [100.0, 0.0]]),
    "EWC+Replay ($\\lambda{=}500$)": np.array([[100.0, 0.0], [91.25, 4.0]]),
}


# ---------------------------------------------------------------------------
# Patient-level Attention-MIL, 3-window domain-incremental. Two runs:
#   run A: 286 train / 88 test patients   (logs/results_mil_20260603_185420.csv)
#   run B (tuned, more data): 1106 train / 340 test (results_mil_20260604_235836)
# Test patient-level AUC per method. Story: methods separate in the small run,
# then regress toward chance (~50) with 4x test data -> the ~0.55 ceiling.
# ---------------------------------------------------------------------------
MIL_METHODS = ["baseline", "EWC", "Replay", "LwF", "EWC+Replay"]
MIL_AUC_A = [44.3, 55.4, 62.9, 50.5, 58.7]   # run A (88 test patients)
MIL_AUC_B = [55.5, 52.8, 53.7, 52.9, 54.3]   # run B (tuned, 340 test patients)


def mil_compare(fname):
    x = np.arange(len(MIL_METHODS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    ax.bar(x - w / 2, MIL_AUC_A, w, label="Run A (88 test patients)",
           color=C_ORANGE, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, MIL_AUC_B, w, label="Run B (tuned, 340 test patients)",
           color=C_BLUE, edgecolor="black", linewidth=0.5)
    ax.axhline(50, ls=(0, (4, 3)), c=C_GRAY, lw=1)
    ax.text(len(MIL_METHODS) - 0.5, 50.6, "chance (AUC 50)", color=C_GRAY,
            fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(MIL_METHODS, rotation=12, ha="right")
    ax.set_ylabel("test patient-level AUC")
    ax.set_ylim(40, 70)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, fname)


if __name__ == "__main__":
    heatmap_grid(CI, 2, "ci_heatmaps")
    summary_bars(CI_SUMMARY, "ci_summary", ymax=118)
    stability_plasticity("ci_stability_plasticity")
    heatmap_grid(WIN, 3, "win_heatmaps")
    summary_bars(WIN_SUMMARY, "win_summary", ymax=70)
    heatmap_grid(IMP, 2, "imp_heatmaps")
    mil_compare("mil_compare")
    print("Vector figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
