import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- 1. Define tasks ----
tasks = ["CVRP", "OVRP", "VRPB", "OVRPB", "VRPBL"]

# ---- 2. Input Δ consistency (replace with your real data) ----
delta = np.array([
    [4.2, 4.8, 5.1, 4.5, 4.7],
    [4.0, 4.3, 4.9, 4.6, 4.4],
    [3.8, 4.2, 4.5, 4.3, 4.1],
    [4.5, 4.7, 5.2, 4.9, 4.8],
    [4.1, 4.4, 4.8, 4.6, 4.3],
])

# ---- 3. Custom warm colormap (yellow → green) ----
cmap = LinearSegmentedColormap.from_list(
    "warm_green",
    ["#F5E6C8", "#C8D5B9", "#6DAA6F"]
)

# ---- 4. Create figure ----
fig, ax = plt.subplots(figsize=(5.2, 4.2))

# ---- 5. Heatmap ----
im = ax.imshow(delta, cmap=cmap, vmin=np.min(delta), vmax=np.max(delta))

# ---- 6. Axis ticks ----
ax.set_xticks(np.arange(len(tasks)))
ax.set_yticks(np.arange(len(tasks)))
ax.set_xticklabels(tasks, fontsize=10)
ax.set_yticklabels(tasks, fontsize=10)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# ---- 7. Annotate cells ----
# ---- Mask diagonal ----
delta_masked = delta.copy().astype(object)

for i in range(len(tasks)):
    delta_masked[i, i] = None  # or np.nan

# ---- Heatmap (use original but mask visually) ----
im = ax.imshow(delta, cmap=cmap, vmin=np.min(delta), vmax=np.max(delta))

# ---- Annotate cells ----
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        if i == j:
            ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="black")
        else:
            val = delta[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)

# ---- 8. Labels ----
ax.set_xlabel("Target Task", fontsize=11)
ax.set_ylabel("Source Task", fontsize=11)

# ---- 9. Colorbar ----
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label("Δ Consistency (%)", fontsize=10)

# ---- 10. Clean layout ----
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

# ---- 11. Save ----
plt.savefig("delta_heatmap_neurips_style.pdf", bbox_inches="tight")
plt.show()