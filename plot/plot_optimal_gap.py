"""
Conference-quality plot: Optimal Gap
  (Left)  λ diffusion weight 0.1 / 0.2 / 0.5
  (Right) Slot count 8 / 16 / 32
Horizontal bars, size 50.  Suitable for 2-column paper layout.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  Style: Computer Modern serif (LaTeX look) or clean sans-serif fallback
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "text.usetex": False,
})

# ═══════════════════════════════════════════════════════════════════════════
#  Conference palette – darker, more saturated, high contrast
# ═══════════════════════════════════════════════════════════════════════════
BAR_A = "#7FA884"   # light sage green
BAR_B = "#6F9172"   # sampled green (main color)
BAR_C = "#4F6F53"   # dark forest green

TEXT_INK   = "#000000"
MUTED_INK  = "#444444"
AXIS_INK   = "#666666"
BASELINE_C = "#999999"

# ── Mock data (size 50) ─────────────────────────────────────────────────────
lambdas    = [0.1, 0.2, 0.5]
gaps_lamb  = [4.42, 4.54, 4.52]

slots      = [8, 16, 32]
gaps_slot  = [4.57, 4.51, 4.42]

# ═══════════════════════════════════════════════════════════════════════════
#  Figure — compact, 2-column friendly
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2),
                                gridspec_kw={"wspace": 0.45})

bar_h = 0.45
colors_lamb = [BAR_A, BAR_B, BAR_C]
colors_slot = [BAR_A, BAR_B, BAR_C]


def draw_bars(ax, y_labels, values, colors, title):
    """Draw a clean horizontal bar chart."""
    y_pos = np.arange(len(y_labels))

    # ── Bars ────────────────────────────────────────────────────────────
    ax.barh(y_pos, values, height=bar_h, color=colors,
            zorder=3, edgecolor="none",
            linewidth=0)

    # ── Value labels at bar tips ─────────────────────────────────────────
    for yi, vi in zip(y_pos, values):
        ax.text(vi + 0.06, yi, f"{vi:.2f}",
                va="center", ha="left", fontsize=8,
                color=MUTED_INK)

    # ── Y ticks ──────────────────────────────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)

    # ── Title (subfigure label, flush left) ──────────────────────────────
    ax.set_title(title, fontweight="bold", loc="left", pad=8)

    # ── Label below bars ─────────────────────────────────────────────────
    ax.set_xlabel("Optimal gap (%)")

    # ── Spines: only left and bottom, thin ───────────────────────────────
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["left"].set_color(AXIS_INK)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(AXIS_INK)

    # ── Ticks ────────────────────────────────────────────────────────────
    ax.tick_params(axis="y", colors=TEXT_INK, length=0)
    ax.tick_params(axis="x", colors=AXIS_INK, length=3, width=0.5)

    # ── No grid ─────────────────────────────────────────────────────────
    ax.grid(False)

    # ── Margins ─────────────────────────────────────────────────────────
    ax.set_ylim(-0.6, len(y_labels) - 0.4)

    # ── Zoom x-axis to [4.4, 4.6] ────────────────────────────────────────
    ax.set_xlim(4.4, 4.6)

    # ── X ticks ──────────────────────────────────────────────────────────
    ax.set_xticks([4.40, 4.45, 4.50, 4.55, 4.60])


# ── Left panel ──────────────────────────────────────────────────────────────
draw_bars(ax1,
          y_labels=[r"$\lambda = 0.1$", r"$\lambda = 0.2$", r"$\lambda = 0.5$"],
          values=gaps_lamb,
          colors=colors_lamb,
          title="(a) Effect of diffusion weight λ")

# ── Right panel ─────────────────────────────────────────────────────────────
draw_bars(ax2,
          y_labels=["Slots = 8", "Slots = 16", "Slots = 32"],
          values=gaps_slot,
          colors=colors_slot,
          title="(b) Effect of slot count")

# ═══════════════════════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════════════════════
plt.savefig("plot/optimal_gap_lambda_slots.pdf",
            bbox_inches="tight", pad_inches=0.02)
plt.savefig("plot/optimal_gap_lambda_slots.png",
            dpi=400, bbox_inches="tight", pad_inches=0.02)
print("✅ Saved: plot/optimal_gap_lambda_slots.pdf + .png")
