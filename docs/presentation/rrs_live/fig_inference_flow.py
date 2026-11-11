"""Left-to-right diagram of the inference step (database in, model, database out),
sized for a 16:9 slide. Grayscale, Arial.

    python fig_inference_flow.py      ->  fig_inference_flow.png / .pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
INK, LIGHT, DARK, MUTED = "#222222", "#F0F0F0", "#D2D2D2", "#666666"

BOXES = [
    ("Study database", "raw sensor data\n(4 signals, 128 Hz)\n+ annotations", LIGHT),
    ("Inference script", "cuts each recording\ninto 8 s windows,\nruns the trained model", DARK),
    ("Results", "eating or not\nprobability of eating\nchewing rate", LIGHT),
    ("Study database", "results saved next to\nthe annotations\n(checked, insert only)", LIGHT),
]
W, H = 12.3, 2.9
BOX_W, BOX_H, GAP = 2.55, 2.0, 0.6
X0 = (W - (len(BOXES) * BOX_W + (len(BOXES) - 1) * GAP)) / 2
Y = 0.45

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
for i, (title, detail, fill) in enumerate(BOXES):
    x = X0 + i * (BOX_W + GAP)
    ax.add_patch(FancyBboxPatch((x, Y), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=fill, edgecolor=INK, linewidth=1.2))
    ax.text(x + BOX_W / 2, Y + BOX_H - 0.18, title, ha="center", va="top", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(x + BOX_W / 2, Y + 0.78, detail, ha="center", va="center", fontsize=12.5,
            color=INK, linespacing=1.25)
    if i < len(BOXES) - 1:
        ax.add_patch(FancyArrowPatch((x + BOX_W + 0.06, Y + BOX_H / 2),
                                     (x + BOX_W + GAP - 0.06, Y + BOX_H / 2),
                                     arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.4))
ax.text(W / 2, 0.12, "One command processes all 61 recordings", ha="center", va="center",
        fontsize=12, style="italic", color=MUTED)

out = Path(__file__).with_name("fig_inference_flow")
for suffix in (".png", ".pdf"):
    fig.savefig(out.with_suffix(suffix), dpi=300)
print("wrote", out.with_suffix(".png"))
