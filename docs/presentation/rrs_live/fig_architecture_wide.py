"""Left-to-right version of the model architecture figure, sized for a 16:9
slide (12.3 in wide, readable when shown full width). Grayscale, Arial.

    python fig_architecture_wide.py      ->  fig_architecture_wide.png / .pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
INK, LIGHT, DARK, MUTED = "#222222", "#F0F0F0", "#D2D2D2", "#666666"

# (title, detail, output shape, fill)
BLOCKS = [
    ("Input", "8 s at 128 Hz\nacc. X, Y, Z\n+ optical\nz-scored", "1024 × 4", LIGHT),
    ("Conv block 1", "32 filters\nkernel 9\nstride 2, pool 2", "256 × 32", LIGHT),
    ("Conv block 2", "64 filters\nkernel 9\nstride 2, pool 2", "64 × 64", LIGHT),
    ("Conv block 3", "64 filters\nkernel 5", "64 × 64", LIGHT),
    ("Channel\nattention", "reweights the\n64 features", "64 × 64", DARK),
    ("Attention\npooling", "weights the\n64 time steps", "64", DARK),
    ("Dense", "64 units\ndropout 0.3", "64", LIGHT),
    ("Eating\nhead", "probability\nof eating", "1", LIGHT),
]
GROUPS = [("Signal", 0, 0), ("Feature extractor", 1, 3), ("Attention", 4, 5), ("Output", 6, 7)]

W, H = 12.3, 3.75
BOX_W, BOX_H, GAP = 1.22, 1.75, 0.31
X0 = (W - (len(BLOCKS) * BOX_W + (len(BLOCKS) - 1) * GAP)) / 2
Y = 1.15

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def x_of(i):
    return X0 + i * (BOX_W + GAP)


for i, (title, detail, shape, fill) in enumerate(BLOCKS):
    x = x_of(i)
    ax.add_patch(FancyBboxPatch((x, Y), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=fill, edgecolor=INK, linewidth=1.1))
    ax.text(x + BOX_W / 2, Y + BOX_H - 0.12, title, ha="center", va="top", fontsize=12.5,
            fontweight="bold", color=INK, linespacing=1.05)
    ax.text(x + BOX_W / 2, Y + (BOX_H - 0.6) / 2, detail, ha="center", va="center", fontsize=10.5,
            color=INK, linespacing=1.15)
    ax.text(x + BOX_W / 2, Y - 0.12, shape, ha="center", va="top", fontsize=10.5,
            style="italic", color=MUTED)
    if i < len(BLOCKS) - 1:
        ax.add_patch(FancyArrowPatch((x + BOX_W + 0.03, Y + BOX_H / 2),
                                     (x + BOX_W + GAP - 0.03, Y + BOX_H / 2),
                                     arrowstyle="-|>", mutation_scale=13, color=INK, lw=1.1))

for name, first, last in GROUPS:
    left, right = x_of(first), x_of(last) + BOX_W
    ax.plot([left, right], [Y + BOX_H + 0.2] * 2, color=MUTED, lw=1)
    ax.text((left + right) / 2, Y + BOX_H + 0.28, name, ha="center", va="bottom",
            fontsize=12, color=MUTED)

# Auxiliary chew-count head, training only: dashed box under the output group.
left, right = x_of(6), x_of(7) + BOX_W
ax.add_patch(FancyBboxPatch((left, 0.12), right - left, 0.5,
                            boxstyle="round,pad=0,rounding_size=0.06", facecolor="white",
                            edgecolor=MUTED, linewidth=1, linestyle="--"))
ax.text((left + right) / 2, 0.37, "Chew-count head\n(training only, loss weight 0.2)",
        ha="center", va="center", fontsize=10, color=MUTED, style="italic", linespacing=1.1)
dense_x = x_of(6) + BOX_W / 2
ax.add_patch(FancyArrowPatch((dense_x - 0.45, Y - 0.02), (dense_x - 0.45, 0.64),
                             arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1,
                             linestyle="--"))

out = Path(__file__).with_name("fig_architecture_wide")
for suffix in (".png", ".pdf"):
    fig.savefig(out.with_suffix(suffix), dpi=300)
print("wrote", out.with_suffix(".png"))
