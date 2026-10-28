r"""
plot_meal_types.py - per-session F1 by meal type, and lunch vs the same person's other meal.

Reads the per-session CSV that session_type_analysis.py writes
(results_fixed\<run>_sessions.csv; the newest one is used if none is given)
and draws one figure, grayscale, Arial 11:

    left    every session as a dot, grouped by meal type, with a line at the
            median of each group
    right   participants with a lunch (--type) AND another session: a line
            from their other meal(s) (mean F1) to their lunch. A line going
            down means that person was detected worse at lunch.

Writes figures\meal_types.png and .pdf.

    python plot_meal_types.py
    python plot_meal_types.py results_fixed\<run>_sessions.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

INK, MID, LIGHT = "#222222", "#777777", "#bbbbbb"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("sessions", nargs="?", help="<run>_sessions.csv (default: newest in results_fixed)")
    parser.add_argument("--type", default="ETS_lunch")
    parser.add_argument("--out", default=str(Path("figures") / "meal_types"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})

    path = Path(args.sessions) if args.sessions else max(
        Path("results_fixed").glob("*_sessions.csv"), key=lambda p: p.stat().st_mtime, default=None)
    if path is None or not path.exists():
        raise SystemExit("no <run>_sessions.csv found; run session_type_analysis.py first")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["bouts"]) > 0]
    print(f"{path}: {len(rows)} sessions")

    studies = sorted({r["study"] for r in rows})
    bottom = min(0.4, np.floor(min(float(r["f1"]) for r in rows) * 10) / 10)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4),
                                         gridspec_kw={"width_ratios": [1.6, 1]})
    rng = np.random.default_rng(0)
    for i, study in enumerate(studies):
        f1 = np.array([float(r["f1"]) for r in rows if r["study"] == study])
        x = i + rng.uniform(-0.18, 0.18, len(f1))
        highlight = study == args.type
        left.plot(x, f1, "o", markersize=6, markeredgecolor=INK,
                  markerfacecolor=INK if highlight else "white", alpha=0.9)
        left.plot([i - 0.3, i + 0.3], [np.median(f1)] * 2, color=INK, linewidth=2.5)
        left.text(i, 1.005, f"median {np.median(f1):.3f}", ha="center", va="bottom", fontsize=9,
                  color=INK if highlight else MID, transform=left.get_xaxis_transform())
    left.set_xticks(range(len(studies)))
    left.set_xticklabels([f"{s.replace('ETS_', '').capitalize()}\n({sum(r['study'] == s for r in rows)})"
                          for s in studies])
    left.set_ylabel("F1 per session")
    left.set_ylim(bottom, 1.0)

    inside = [r for r in rows if r["study"] == args.type]
    pairs = []
    for person in sorted({r["participant"] for r in inside}):
        other = [float(r["f1"]) for r in rows if r["participant"] == person and r["study"] != args.type]
        if other:
            mine = np.mean([float(r["f1"]) for r in inside if r["participant"] == person])
            pairs.append((person, float(np.mean(other)), float(mine)))
    name = args.type.replace("ETS_", "").capitalize()
    for person, other, mine in pairs:
        down = mine < other
        right.plot([0, 1], [other, mine], "-o", color=INK if down else LIGHT,
                   markersize=5, linewidth=1.5 if down else 1.2)
    worse = sum(m < o for _, o, m in pairs)
    right.set_xticks([0, 1])
    right.set_xticklabels(["Other meal", name])
    right.set_xlim(-0.3, 1.3)
    right.set_ylim(bottom, 1.0)
    right.set_ylabel("F1 per session")
    right.set_title(f"Same participants: {worse} of {len(pairs)} lower at {name.lower()}",
                    loc="left", fontsize=11)
    left.set_title(f"Every session ({name.lower()} filled)", loc="left", fontsize=11, pad=16)
    figure.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=200)
    print(f"{len(pairs)} participants with {name.lower()} and another meal, {worse} lower at "
          f"{name.lower()}\nwrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
