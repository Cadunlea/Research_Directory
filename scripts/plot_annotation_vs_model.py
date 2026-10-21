r"""
plot_annotation_vs_model.py - an annotation file against the model's output.

Draws the 10 Hz annotation exactly as exported from the annotation tool
(chew bouts, bites, chews) above the model's eating probability for the same
stretch of time, so you can see where the model agrees, where it misses, and
whether a miss is a pause, a short bout or a weak signal.

The model's output can come from either:

  --run      a finished leave-one-out run (results_fixed\<run>.json). These
             are held-out predictions: the model never saw this participant.
             This is the fair view of performance.
  --windows  a detect_eating.py output (detections\<...>_windows.csv), from
             the final model, which was trained on (almost) everyone.

USAGE
-----
    python plot_annotation_vs_model.py --annotation AIM147560_annotation_v01.csv --run results_fixed\time_se_attnpool_chew0.2_w8_128hz_loso_seed9_stopauc_lr0.0003.json --participant AIM147560
    python plot_annotation_vs_model.py --annotation AIM147560_annotation_v01.csv --windows detections\AIM147560_ETS_lunch_windows.csv

TIME ZONES. The annotation file's clock may be local time while the model
output is UTC. Unless --tz-offset is given, the whole-hour shift that best
overlaps the two is used, and printed; check it is the one you expect
(US Central is -5 in summer, -6 in winter).

Writes figures\annotation_vs_model_<name>.png and .pdf (grayscale, Arial 11).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
INK, MID, LIGHT = "#222222", "#888888", "#d9d9d9"


def epoch(text: str, fmt: str) -> float:
    return datetime.strptime(text.strip(), fmt).replace(tzinfo=timezone.utc).timestamp()


def read_annotation(path: Path):
    with open(path, encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    t = np.array([epoch(r["bin_timecode"], "%Y-%m-%d %H:%M:%S.%f") for r in rows])
    get = lambda name: np.array([int(float(r[name] or 0)) for r in rows])  # noqa: E731
    return t, get("Chew Bout"), get("Bite"), get("Chew Count")


def read_run(path: Path, participant: str, study: str | None):
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    data = np.load(path.with_name(f"{record['tag']}_predictions.npz"), allow_pickle=False)
    seed = record["repeats"][0]["seed"]
    p, thr = data[f"probability_seed{seed}"], data[f"threshold_seed{seed}"]
    keep = data["evaluated"] & (data["participants"] == participant) & np.isfinite(p)
    if study:
        keep &= data["studies"] == study
    if not keep.any():
        raise SystemExit(f"{participant} has no held-out windows in {path.name}")
    order = np.flatnonzero(keep)[np.argsort(data["starts"][keep])]
    return (data["starts"][order], p[order], float(np.median(thr[order])),
            float(data["window_seconds"]), f"held out ({record['tag']})")


def read_windows(path: Path):
    with open(path, encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["probability"] != ""]
    starts = np.array([epoch(r["start_utc"], "%Y-%m-%d %H:%M:%S") for r in rows])
    window = epoch(rows[0]["end_utc"], "%Y-%m-%d %H:%M:%S") - starts[0]
    p = np.array([float(r["probability"]) for r in rows])
    eating = np.array([int(r["eating"]) for r in rows])
    # The windows file records decisions, not the threshold; recover it.
    threshold = float(np.min(p[eating == 1])) if eating.any() else np.nan
    return starts, p, threshold, window, f"final model ({path.name})"


def spans(flags: np.ndarray):
    d = np.diff(np.r_[0, flags.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--annotation", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run")
    source.add_argument("--windows")
    parser.add_argument("--participant", help="with --run")
    parser.add_argument("--study", help="with --run, if the participant has two sessions")
    parser.add_argument("--tz-offset", type=float, default=None,
                        help="hours to ADD to the annotation clock to get UTC")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    t_ann, bout, bite, chew = read_annotation(Path(args.annotation))
    if args.run:
        if not args.participant:
            raise SystemExit("--run needs --participant")
        starts, p, threshold, window, label = read_run(Path(args.run), args.participant, args.study)
    else:
        starts, p, threshold, window, label = read_windows(Path(args.windows))

    if args.tz_offset is None:
        def overlap(hours):
            lo = max(t_ann[0] + hours * 3600, starts[0])
            hi = min(t_ann[-1] + hours * 3600, starts[-1] + window)
            return hi - lo
        shift = max(range(-14, 15), key=overlap)
        if overlap(shift) <= 0:
            raise SystemExit("the annotation and the model output do not overlap at "
                             "any whole-hour shift; check the participant and session")
        print(f"aligned with a {shift:+d} h shift of the annotation clock "
              f"(override with --tz-offset)")
    else:
        shift = args.tz_offset
    t_ann = t_ann + shift * 3600
    origin = t_ann[0]
    minutes = lambda t: (t - origin) / 60.0                          # noqa: E731

    inside = (starts + window > t_ann[0]) & (starts < t_ann[-1])
    starts, p = starts[inside], p[inside]

    figure, (top, strip) = plt.subplots(
        2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for a, b in spans(bout > 0):
        top.axvspan(minutes(t_ann[a]), minutes(t_ann[b - 1] + 0.1), color=LIGHT, linewidth=0)
    top.step(minutes(np.r_[starts, starts[-1] + window]), np.r_[p, p[-1]],
             where="post", color=INK, linewidth=1.3)
    if np.isfinite(threshold):
        top.axhline(threshold, color=MID, linestyle="--", linewidth=1)
    top.set_ylim(0, 1)
    top.set_ylabel("p(eating)")
    top.set_title(f"{Path(args.annotation).stem}: shaded = annotated chew bouts; "
                  f"line = model, {label}; dashed = threshold", loc="left", fontsize=10)

    rows = [("Model: eating", (p >= threshold) if np.isfinite(threshold) else p >= 0.5, starts, window),
            ("Chew bout", bout > 0, t_ann, 0.1), ("Bite", bite > 0, t_ann, 0.1),
            ("Chew", chew > 0, t_ann, 0.1)]
    for level, (name, flags, times, width) in enumerate(rows):
        y = len(rows) - 1 - level
        for a, b in spans(flags):
            start = minutes(times[a])
            end = minutes(times[b - 1] + width)
            strip.broken_barh([(start, max(end - start, 0.012))], (y - 0.35, 0.7),
                              color=INK if name != "Chew bout" else MID)
    strip.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    strip.set_ylim(-0.6, len(rows) - 0.4)
    strip.set_xlabel("Minutes from the start of the annotation file")
    strip.tick_params(axis="y", length=0)

    eat_frac = float(np.mean((bout > 0) | (bite > 0)))
    short = [b - a for a, b in spans(bout > 0)]
    print(f"annotation: {len(t_ann) / 600:.1f} min, eating {eat_frac:.0%}, "
          f"{len(short)} chew bouts (median {np.median(short) / 10:.1f} s, "
          f"{np.mean(np.array(short) < 80):.0%} shorter than one 8 s window), "
          f"{int(np.sum(np.diff(np.r_[0, bite > 0].astype(int)) == 1))} bites")
    figure.tight_layout()
    out = Path(args.out) if args.out else Path("figures") / f"annotation_vs_model_{Path(args.annotation).stem}"
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=300)
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
