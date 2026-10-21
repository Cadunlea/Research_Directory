r"""
plot_sensor_vs_annotations.py - raw AIM-2 signals, the annotation, and the
model, for one annotated session, on one time axis.

Based on plot_sensor_vs_annotations_central_time_corrected.py, with these
corrections:

  1. data_timestamp is the packet START. The original subtracted the packet
     duration, treating it as the end, which drew the sensor 8 s early
     against the annotation (measured on this database: +0.1 s lag read as
     start, -7.9 s read as end, on all 20 sessions checked).
  2. Time is handled as POSIX seconds (UTC) throughout, and drawn in this
     computer's local time, which follows daylight saving. The original
     added a fixed +5 h, which is right only in summer.
  3. Only the annotated stretch (plus --padding minutes) is loaded, and only
     the packets inside it. The original loaded every packet of the
     participant and spanned all of them, which for a participant with two
     sessions on different days builds an array covering days.
  4. "Eating" is chew bout OR bite, the definition the model is trained on.
     (The original also counted any chew-count mark.) Chew counts keep their
     own panel.
  5. Missing sensor samples are left as gaps, not drawn as -1.
  6. Optional bottom panel: the model's eating probability for this
     participant, either held out (--run, a leave-one-out result: the fair
     view) or from the final model (--windows, a detect_eating.py output).

The annotation clock (bin_timecode) is local wall-clock time. The whole-hour
offset to UTC is found by matching the annotation to the sensor packets that
exist, and printed; set --utc-offset to override (US Central: -5 summer,
-6 winter).

USAGE
-----
    python plot_sensor_vs_annotations.py --config "%CFG%" --annotation "C:\...\AIM147560_annotation_v01.csv" --participant AIM147560
    python plot_sensor_vs_annotations.py --config "%CFG%" --annotation "C:\...\AIM147560_annotation_v01.csv" --participant AIM147560 --run results_fixed\time_se_attnpool_chew0.2_w8_128hz_loso_seed9_stopauc_lr0.0003.json
    ... --windows detections\AIM147560_ETS_lunch_windows.csv    (final-model output instead)
    ... --show                                                  (also open a window)

Writes figures\sensor_vs_annotations_<participant>_<file>.png and .pdf
(grayscale, Arial 11).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
import ets_data

INK, MID, LIGHT = "#222222", "#777777", "#d9d9d9"


# --------------------------------------------------------------------------- #
# annotation
# --------------------------------------------------------------------------- #
def read_annotation(path: Path):
    """Naive wall-clock seconds (as if UTC) and the three annotation tracks."""
    with open(path, encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty")
    stamps = []
    for r in rows:
        text = r["bin_timecode"].strip().replace("'", "")
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                stamps.append(datetime.strptime(text, fmt)
                              .replace(tzinfo=timezone.utc).timestamp())
                break
            except ValueError:
                continue
        else:
            raise SystemExit(f"cannot read bin_timecode {text!r}; expected "
                             f"'YYYY-MM-DD HH:MM:SS.f'")
    track = lambda name: np.array([float(r.get(name) or 0) for r in rows])  # noqa: E731
    return (np.asarray(stamps), track("Chew Bout"), track("Bite"),
            track("Chew Count"))


def packet_times(conn, participant: str) -> np.ndarray:
    rows = ets_data.query(
        conn, f"SELECT data_timestamp FROM {ets_data.RAW_DB}.numeric_data "
              f"WHERE data_identifier=%s AND participantID=%s",
        (ets_data.SENSOR_IDENTIFIER, participant))
    return np.sort(np.array([float(r[0]) for r in rows]))


def utc_offset_hours(annotation_naive: np.ndarray, packets: np.ndarray) -> int:
    """Whole hours h such that naive local time - h hours is UTC, chosen so
    the annotated stretch is covered by as many sensor packets as possible."""
    def covered(h):
        lo, hi = annotation_naive[0] - h * 3600, annotation_naive[-1] - h * 3600
        return int(np.sum((packets >= lo - ets_data.PACKET_SECONDS) & (packets <= hi)))
    best = max(range(-14, 15), key=lambda h: (covered(h), -abs(h + 5.5)))
    if covered(best) == 0:
        raise SystemExit("no sensor packets fall inside the annotated stretch at "
                         "any whole-hour offset; check the participant and file")
    return best


# --------------------------------------------------------------------------- #
# sensor
# --------------------------------------------------------------------------- #
def load_sensor(conn, participant: str, start: float, end: float) -> np.ndarray:
    """(n, 4) at 128 Hz over [start, end); NaN where nothing was received.
    data_timestamp is the packet START."""
    n = int(np.ceil((end - start) * ets_data.SENSOR_FS))
    block = np.full((n, 4), np.nan)
    rows = ets_data.query(
        conn, f"SELECT data_timestamp, data_ndarray_pickle "
              f"FROM {ets_data.RAW_DB}.numeric_data "
              f"WHERE data_identifier=%s AND participantID=%s "
              f"AND data_timestamp BETWEEN %s AND %s ORDER BY data_timestamp",
        (ets_data.SENSOR_IDENTIFIER, participant,
         int(start) - int(ets_data.PACKET_SECONDS) - 1, int(end) + 1))
    for timestamp, blob in rows:
        try:
            array = np.asarray(pickle.loads(blob), dtype=float)
        except Exception:                                       # noqa: BLE001
            continue
        if array.ndim != 2 or 4 not in array.shape:
            continue
        samples = array if array.shape[1] == 4 and array.shape[0] != 4 else array.T
        samples = samples[:, :4]
        offset = int(round((float(timestamp) - start) * ets_data.SENSOR_FS))
        lo, hi = max(offset, 0), min(offset + len(samples), n)
        if lo < hi:
            block[lo:hi] = samples[lo - offset:hi - offset]
    block[block == ets_data.MISSING] = np.nan
    return block


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def model_from_run(path: Path, participant: str):
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    data = np.load(path.with_name(f"{record['tag']}_predictions.npz"), allow_pickle=False)
    seed = record["repeats"][0]["seed"]
    p, thr = data[f"probability_seed{seed}"], data[f"threshold_seed{seed}"]
    keep = data["evaluated"] & (data["participants"] == participant) & np.isfinite(p)
    order = np.flatnonzero(keep)[np.argsort(data["starts"][keep])]
    if not len(order):
        raise SystemExit(f"{participant} has no held-out windows in {path.name}")
    return (data["starts"][order], p[order], thr[order],
            float(data["window_seconds"]), "held out (leave-one-out)")


def model_from_windows(path: Path):
    with open(path, encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["probability"]]
    t = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(  # noqa: E731
        tzinfo=timezone.utc).timestamp()
    starts = np.array([t(r["start_utc"]) for r in rows])
    p = np.array([float(r["probability"]) for r in rows])
    eating = np.array([int(r["eating"]) for r in rows])
    threshold = float(p[eating == 1].min()) if eating.any() else np.nan
    return (starts, p, np.full(len(p), threshold),
            t(rows[0]["end_utc"]) - starts[0], "final model")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ets_data.add_database_arguments(parser)
    parser.add_argument("--annotation", required=True, help="annotation CSV (bin_timecode, Chew Count, Chew Bout, Bite)")
    parser.add_argument("--participant", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", help="leave-one-out result JSON (held-out model panel)")
    source.add_argument("--windows", help="detect_eating.py windows CSV (final-model panel)")
    parser.add_argument("--padding", type=float, default=2.0, help="minutes either side (default 2)")
    parser.add_argument("--utc-offset", type=int, default=None,
                        help="hours of the annotation clock relative to UTC (Central: -5 summer, -6 winter)")
    parser.add_argument("--show", action="store_true", help="also open the figure in a window")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not args.config and os.environ.get("CFG"):
        args.config = os.environ["CFG"]

    matplotlib.use("TkAgg" if args.show else "Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})

    naive, bout, bite, chew = read_annotation(Path(args.annotation))
    conn = ets_data.connect(args)
    try:
        packets = packet_times(conn, args.participant)
        if not len(packets):
            raise SystemExit(f"no XYZO packets for {args.participant}")
        offset = args.utc_offset if args.utc_offset is not None else utc_offset_hours(naive, packets)
        ann_t = naive - offset * 3600
        start = ann_t[0] - args.padding * 60
        end = ann_t[-1] + 0.1 + args.padding * 60
        sensor = load_sensor(conn, args.participant, start, end)
    finally:
        conn.close()
    print(f"annotation clock = UTC{offset:+d} h"
          f"{'' if args.utc_offset is not None else ' (found from the sensor packets)'}; "
          f"{(ann_t[-1] - ann_t[0]) / 60:.1f} min annotated, "
          f"{np.mean(~np.isnan(sensor[:, 3])):.0%} of the plotted time has sensor data")

    model = None
    if args.run:
        model = model_from_run(Path(args.run), args.participant)
    elif args.windows:
        model = model_from_windows(Path(args.windows))

    local = lambda t: [datetime.fromtimestamp(x) for x in np.atleast_1d(t)]  # noqa: E731
    sensor_t = start + np.arange(len(sensor)) / ets_data.SENSOR_FS
    step = max(1, len(sensor) // 60000)          # keep the line plots light
    eating = (bout > 0) | (bite > 0)

    panels = 8 + (model is not None)
    heights = [1.2] * 4 + [0.6] * 4 + ([1.4] if model else [])
    figure, axes = plt.subplots(panels, 1, figsize=(13, 1.25 * panels + 1.5),
                                sharex=True, gridspec_kw={"height_ratios": heights})
    for i, name in enumerate(("Acc X", "Acc Y", "Acc Z", "Optical")):
        axes[i].plot(local(sensor_t[::step]), sensor[::step, i], color=INK, linewidth=0.6)
        axes[i].set_ylabel(name)

    def track(axis, values, name):
        axis.fill_between(local(ann_t), 0, values, step="post", color=MID, linewidth=0)
        axis.set_ylim(0, 1.15)
        axis.set_yticks([])
        axis.set_ylabel(name, rotation=0, ha="right", va="center")
    track(axes[4], eating.astype(float), "Eating\n(bout or bite)")
    track(axes[5], (bout > 0).astype(float), "Chew bout")
    track(axes[6], (bite > 0).astype(float), "Bite")
    track(axes[7], (chew > 0).astype(float), "Chew")

    for axis in axes[:4]:
        for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, eating.astype(int), 0]) == k)
                          for k in (1, -1)]):
            axis.axvspan(*local([ann_t[a], ann_t[b - 1] + 0.1]), color=LIGHT, linewidth=0, zorder=0)

    if model:
        starts, p, thr, window, label = model
        inside = (starts + window > start) & (starts < end)
        starts, p, thr = starts[inside], p[inside], thr[inside]
        axis = axes[-1]
        for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, eating.astype(int), 0]) == k)
                          for k in (1, -1)]):
            axis.axvspan(*local([ann_t[a], ann_t[b - 1] + 0.1]), color=LIGHT, linewidth=0)
        if len(starts):
            axis.step(local(np.r_[starts, starts[-1] + window]), np.r_[p, p[-1]],
                      where="post", color=INK, linewidth=1.2)
            if np.isfinite(thr).any():
                axis.axhline(float(np.nanmedian(thr)), color=MID, linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_ylabel(f"Model\np(eating)\n{label}", fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("Clock time (this computer's time zone)")
    axes[0].set_title(f"{args.participant}  -  {Path(args.annotation).name}.  Shaded: annotated "
                      f"eating (chew bout or bite). Gaps: no sensor data."
                      + ("  Dashed: model threshold." if model else ""),
                      loc="left", fontsize=10)
    figure.tight_layout()

    out = Path(args.out) if args.out else (
        Path("figures") / f"sensor_vs_annotations_{args.participant}_{Path(args.annotation).stem}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=200)
    print(f"wrote {out.with_suffix('.png')}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
