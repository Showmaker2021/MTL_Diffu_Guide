# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # Load data
# # df = pd.read_csv("results/test/20260410_155829/attn_stats.csv")  # change path if needed

# # x = df["distance"].values
# # y = df["attention"].values

# # # Create figure
# # plt.figure(figsize=(6, 5))

# # hb = plt.hexbin(
# #     x, y,
# #     gridsize=60,
# #     bins='log',      # log density
# #     cmap='Blues',
# #     mincnt=1
# # )

# # # Labels
# # plt.xlabel("Distance")
# # plt.ylabel("Attention")
# # plt.title("(a) SOLID")

# # # Colorbar
# # cb = plt.colorbar(hb)
# # cb.set_label("log10(N)")

# # plt.tight_layout()

# # # 🔥 Save figure (IMPORTANT)
# # plt.savefig("solid_attention_plot.png", dpi=300)  # high-quality for paper

# # # Show (optional)
# # plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# df = pd.read_csv("attn_stats.csv")

# # plt.figure(figsize=(6,5))

# # sns.kdeplot(
# #     x=df["distance"],
# #     y=df["attention"],
# #     fill=True,
# #     cmap="Blues",
# #     levels=50
# # )

# # plt.xlabel("Distance")
# # plt.ylabel("Attention")
# # plt.title("(a) SOLID - KDE")

# # plt.tight_layout()
# # plt.savefig("kde_plot.png", dpi=300)
# # plt.close()

# # bin distances
# # bins = np.linspace(df["distance"].min(), df["distance"].max(), 30)
# # digitized = np.digitize(df["distance"], bins)

# # # average attention per bin
# # bin_means = [
# #     df["attention"][digitized == i].mean()
# #     for i in range(1, len(bins))
# # ]

# # bin_centers = (bins[:-1] + bins[1:]) / 2

# # plt.figure(figsize=(6,5))

# # plt.plot(bin_centers, bin_means, marker='o')

# # plt.xlabel("Distance")
# # plt.ylabel("Avg Attention")
# # plt.title("(a) SOLID - Distance vs Attention")

# # plt.grid(alpha=0.3)

# # plt.tight_layout()
# # plt.savefig("line_plot.png", dpi=300)
# # plt.close()


# x = df["distance"].values

# # Create histogram
# bins = 100
# hist, edges = np.histogram(x, bins=bins)

# # Log scale (like CaDA)
# hist = np.log10(hist + 1)

# # Normalize for color
# hist = hist / hist.max()

# # Convert to 2D (for imshow)
# heatmap = hist[np.newaxis, :]  # shape (1, bins)

# # Plot
# plt.figure(figsize=(6, 1.2))

# plt.imshow(
#     heatmap,
#     aspect='auto',
#     cmap='viridis',
#     extent=[edges[0], edges[-1], 0, 1]
# )

# # Remove y-axis
# plt.yticks([])

# # X label
# # plt.xlabel("Distance")
# # plt.title("(a) SOLID")

# # Colorbar
# cbar = plt.colorbar()
# cbar.set_label("Density")

# plt.tight_layout()

# # Save
# plt.savefig("density_bar.png", dpi=300)
# plt.close()


import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Colors
# -----------------------------
COLOR_A = "#6F8F72"   # muted green
COLOR_B = "#E6D8B5"   # soft beige

# -----------------------------
# Decoder data (split into 2)
# -----------------------------
labels_decoder = [
    "SOLID",
    "Embedding",
    "No Cap",
    "No Residual",
    "No Struct."
]

seen = [4.42, 4.64, 4.55, 4.69, 4.66]
unseen = [4.74, 4.91, 4.95, 4.83, 4.97]

# Split
labels_mech = labels_decoder[:3]
seen_mech = seen[:3]
unseen_mech = unseen[:3]

labels_struct = [labels_decoder[0]] + labels_decoder[3:]
seen_struct = [seen[0]] + seen[3:]
unseen_struct = [unseen[0]] + unseen[3:]

# -----------------------------
# Diffusion
# -----------------------------
models = ["SOLID", "w/o Diff"]
with_diff = [4.42, 4.74]
without_diff = [4.62, 5.03]

# -----------------------------
# Slots
# -----------------------------
# slots = ["Slot-4", "Slot-8", "Slot-16", "Slot-32"]
# size50_slots = [4.60, 4.57, 4.44, 4.42]
# size100_slots = [4.78, 4.94, 4.80, 4.74]


slots = ["Slot-8", "Slot-16", "Slot-32", "Slot-64"]
size100_slots = [4.94, 4.86, 4.74, 5.02]
size50_slots = [4.60, 4.57, 4.44, np.nan]

# -----------------------------
# Plot setup (4 panels)
# -----------------------------
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
width = 0.35

# -----------------------------
# (a) Decoder - mechanism
# -----------------------------
ax = axes[0]
x = np.arange(len(labels_mech))

ax.bar(x - width/2, seen_mech, width, color=COLOR_B)
ax.bar(x + width/2, unseen_mech, width, color=COLOR_A)
ax.set_ylim(4, 5)

ax.set_xticks(x)
ax.set_xticklabels(labels_mech, rotation=15)
ax.set_title("(a) Gating Mechanism")

ax.legend(["Size 50", "Size 100"], fontsize=10, frameon=False, loc="upper left")

# -----------------------------
# (b) Decoder - structural
# -----------------------------
ax = axes[1]
x = np.arange(len(labels_struct))

ax.bar(x - width/2, seen_struct, width, color=COLOR_B)
ax.bar(x + width/2, unseen_struct, width, color=COLOR_A)
ax.set_ylim(4, 5)

ax.set_xticks(x)
ax.set_xticklabels(labels_struct, rotation=15)
ax.set_title("(b) Structural Components")

ax.legend(["Size 50", "Size 100"], fontsize=10, frameon=False, loc="upper left")

# -----------------------------
# (c) Diffusion
# -----------------------------
ax = axes[2]
x = np.arange(len(models))

ax.bar(x - width/2, with_diff, width, color=COLOR_B)
ax.bar(x + width/2, without_diff, width, color=COLOR_A)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_title("(c) Diffusion Objective")
ax.set_ylim(4, 5.5)

ax.legend(["Size 50", "Size 100"], fontsize=10, frameon=False, loc="upper left")

# -----------------------------
# (d) Slots
# -----------------------------
ax = axes[3]
x = np.arange(len(slots))

ax.bar(x - width/2, size50_slots, width, color=COLOR_B)
ax.bar(x + width/2, size100_slots, width, color=COLOR_A)

ax.set_xticks(x)
ax.set_xticklabels(slots)
ax.set_title("(d) Slot Sensitivity")
ax.set_ylim(4, 5.2)

ax.legend(["Size 50", "Size 100"], fontsize=10, frameon=False, loc="upper left")

# -----------------------------
# Styling
# -----------------------------
for ax in axes:
    ax.set_ylabel("Gap (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("ablation_4panel.png", dpi=300, bbox_inches="tight")
plt.show()