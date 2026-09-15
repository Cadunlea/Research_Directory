r"""
plot_drift_evidence.py - one figure that settles the 8-second question.

WHAT IT IS FOR
--------------
check_annotation_drift.py produces the numbers. This produces the picture to
put on a slide, because the argument is visual: the correlation curve has the
SAME SHAPE under both readings of data_timestamp and is merely displaced by one
8 s packet. That is what distinguishes a units bug from real clock drift, and
it is far more convincing seen than described.

    Panel A   correlation against candidate lag, every session, both
              conventions. Two bundles of curves, identical in shape, 8 s
              apart. Real drift would scatter and flatten them.
    Panel B   the lag each session settled on. 20 points at 0, 20 at -8.
    Panel C   a worked example: the annotated eating periods with the sensor's
              chewing energy drawn under each convention, so the alignment can
              be checked by eye against the shaded blocks.

USAGE
-----
    python plot_drift_evidence.py --config path\to\jitai_config.ini
    python plot_drift_evidence.py --config ... --example AIM121260
    python plot_drift_evidence.py --config ... --output-dir slides

Writes drift_evidence.png (and .pdf, for a paper) plus drift_evidence_data.csv
with the per-session numbers behind it.

Recomputes the curves from the database rather than reading the CSV, because
the CSV stores only each session's chosen lag, not the curve it came from.
Takes a couple of minutes. Uses only packages pinned in
requirements_ETS_full.txt.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, transforms

import check_annotation_drift as drift


# --------------------------------------------------------------------------- #
# Palette. Two categories only - the correct reading and the old one - so two
# categorical hues plus recessive ink for everything that is not data.
# Validated (blue/orange, light surface): CVD dE 24.7 protan / 32.7 tritan,
# normal-vision dE 33.6, contrast >= 3:1, all checks pass.
# --------------------------------------------------------------------------- #
CORRECT = "#2a78d6"          # data_timestamp = packet START
LEGACY = "#eb6834"           # data_timestamp = packet END (what the old code did)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"
EATING_BAND = "#d8d7d2"

CONVENTION_LABEL = {
    "start": "data_timestamp = packet start  (correct)",
    "end": "data_timestamp = packet end  (previous code)",
}
CONVENTION_COLOR = {"start": CORRECT, "end": LEGACY}


def style() -> None:
    plt.rcParams.update({
        # Times New Roman to match the lab's other figures, with a fallback so
        # this still runs on a machine that does not have it.
        "font.family": ["Times New Roman", "DejaVu Serif", "serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": "#e4e3de",
        "grid.linewidth": 0.7,
    })


def clean(axis, grid_axis: str = "both") -> None:
    """Recessive frame: no top/right spines, a light grid behind the data."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.grid(True, axis=grid_axis, linestyle="--", alpha=0.6)
    axis.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def measure_all(conn, sessions, max_lag: float) -> List[dict]:
    results = []
    for convention in ("start", "end"):
        print(f"  measuring [{convention}] ...")
        for session in sessions:
            result = drift.measure_session(conn, session, convention, max_lag)
            if result is not None:
                results.append(result)
    return results


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def panel_curves(axis, results: List[dict]) -> None:
    """Every session's correlation-vs-lag curve, both conventions.

    The point of the panel is the SHAPE. Individual curves are drawn thin and
    translucent so twenty of them read as a bundle rather than a tangle, with
    the median curve solid on top.
    """
    for convention in ("start", "end"):
        subset = [r for r in results if r["convention"] == convention]
        if not subset:
            continue
        colour = CONVENTION_COLOR[convention]
        lags = subset[0]["_lags"]
        for result in subset:
            axis.plot(result["_lags"], result["_correlations"],
                      color=colour, linewidth=0.8, alpha=0.25)
        stack = np.vstack([r["_correlations"] for r in subset])
        median = np.nanmedian(stack, axis=0)
        axis.plot(lags, median, color=colour, linewidth=2.0,
                  label=CONVENTION_LABEL[convention])

        peak = lags[int(np.nanargmax(median))]
        axis.plot([peak], [np.nanmax(median)], marker="o", markersize=8,
                  color=colour, markeredgecolor=SURFACE, markeredgewidth=2,
                  zorder=5)
        # Nudge the label away from the vertical zero line, which otherwise
        # runs straight through the text.
        nudge = 14 if peak >= 0 else -14
        axis.annotate(f"{peak:+.1f} s", xy=(peak, np.nanmax(median)),
                      xytext=(nudge, 9), textcoords="offset points",
                      ha="left" if peak >= 0 else "right", color=colour,
                      fontsize=10, fontweight="bold")

    axis.axvline(0, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=0)
    axis.set_xlabel("candidate lag applied to the annotation (s)")
    axis.set_ylabel("correlation with annotated eating")
    axis.set_title("A.  Same curve, displaced by one 8 s packet")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38))
    clean(axis)


def panel_lags(axis, results: List[dict]) -> None:
    """Where each session's curve peaked. A dot per session, not a histogram:
    at n=20 the individual points are the evidence, and binning hides that
    every single session agrees."""
    rng = np.random.default_rng(0)
    for index, convention in enumerate(("start", "end")):
        subset = [r for r in results if r["convention"] == convention]
        if not subset:
            continue
        lags = np.array([r["best_lag_seconds"] for r in subset])
        jitter = rng.uniform(-0.16, 0.16, len(lags))
        axis.scatter(lags, np.full(len(lags), index) + jitter,
                     s=42, color=CONVENTION_COLOR[convention],
                     edgecolor=SURFACE, linewidth=1.2, zorder=3)

        # Left-align the summary in AXES coordinates, not beside the cluster.
        # Anchored to the points it would run off the edge, since the two
        # clusters sit at opposite ends of the range.
        median = float(np.median(lags))
        blended = transforms.blended_transform_factory(axis.transAxes,
                                                       axis.transData)
        axis.text(0.02, index + 0.40,
                  f"median {median:+.2f} s   (n={len(lags)}, "
                  f"spread {lags.min():+.2f} to {lags.max():+.2f})",
                  transform=blended, ha="left",
                  color=CONVENTION_COLOR[convention], fontsize=9,
                  fontweight="bold")

    axis.axvline(0, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=0)
    axis.axvline(-8.0, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=0)
    axis.annotate("the shift the previous\nmodel hardcoded", xy=(-8.0, -0.42),
                  ha="center", va="top", color=INK_SECONDARY, fontsize=8)
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["packet\nstart", "packet\nend"], fontsize=9)
    axis.set_xlim(-11.5, 3.5)
    axis.set_ylim(-0.95, 1.7)
    axis.set_xlabel("measured lag (s)")
    axis.set_title("B.  Every session agrees")
    clean(axis, grid_axis="x")


def panel_example(axis, conn, session, minutes: float = 5.0) -> None:
    """One session's chewing energy against the annotated eating blocks.

    The check anyone can make by eye: does the sensor's chewing energy rise
    inside the shaded blocks? Under the correct convention it does; under the
    old one it is displaced by 8 s, visibly leading each block.
    """
    first, last = session.annotated_range()
    frequency = session.frequency
    video_start = session.day_start + first / frequency
    video_end = session.day_start + (last + 1) / frequency

    # Start the excerpt shortly before the first eating period rather than at
    # the top of the video. Most sessions open with a minute or two of nothing,
    # and an excerpt of empty signal shows the reader nothing either way.
    full_mask = session.eating_mask()[first:last + 1]
    eating_at = np.flatnonzero(full_mask == 1)
    lead_in = 20.0
    offset = (max(0.0, eating_at[0] / frequency - lead_in)
              if eating_at.size else 0.0)
    span_start = min(video_start + offset, max(video_start, video_end - minutes * 60))
    span_end = min(video_end, span_start + minutes * 60)

    begin = first + int(round((span_start - video_start) * frequency))
    mask = session.eating_mask()[begin:begin + int(round((span_end - span_start) * frequency))]
    times = np.arange(len(mask)) / frequency

    # Shaded blocks = annotated eating. Drawn first, behind everything.
    edges = np.diff(np.concatenate(([0], (mask == 1).astype(np.int8), [0])))
    for start_index, stop_index in zip(np.flatnonzero(edges == 1),
                                       np.flatnonzero(edges == -1)):
        axis.axvspan(times[start_index], times[min(stop_index, len(times) - 1)],
                     color=EATING_BAND, zorder=0, linewidth=0)

    for convention in ("start", "end"):
        optical = drift.load_optical(conn, session.participant,
                                     span_start, span_end, convention)
        activity, valid = drift.chewing_activity(optical)
        if activity.size == 0:
            continue
        activity = np.where(valid, activity, np.nan)
        finite = activity[np.isfinite(activity)]
        if finite.size:
            activity = (activity - finite.min()) / max(finite.ptp(), 1e-9)
        axis_times = np.arange(len(activity)) / drift.ANALYSIS_FS
        axis.plot(axis_times, activity, color=CONVENTION_COLOR[convention],
                  linewidth=1.4, alpha=0.9,
                  label=CONVENTION_LABEL[convention])

    axis.set_xlim(0, span_end - span_start)
    axis.set_ylim(-0.03, 1.15)
    axis.set_xlabel(f"seconds from {offset:.0f} s into the annotated video")
    axis.set_ylabel("chewing-band energy\n(normalised)")
    axis.set_title(f"C.  {session.participant} {session.study} - shaded blocks "
                   f"are annotated eating; blue rises with them, orange leads "
                   f"them by 8 s")
    axis.legend(loc="upper right", ncol=2)
    clean(axis, grid_axis="x")


# --------------------------------------------------------------------------- #
def write_csv(results: List[dict], path: Path) -> None:
    fields = ["participant", "study", "convention", "best_lag_seconds",
              "best_correlation", "correlation_at_zero_lag",
              "sensor_coverage", "eating_fraction", "annotated_seconds"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})


def summarise(results: List[dict]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for convention in ("start", "end"):
        subset = [r for r in results if r["convention"] == convention]
        if not subset:
            continue
        lags = np.array([r["best_lag_seconds"] for r in subset])
        peaks = np.array([r["best_correlation"] for r in subset])
        summary[convention] = {
            "n": len(subset),
            "median_lag": float(np.median(lags)),
            "min_lag": float(lags.min()),
            "max_lag": float(lags.max()),
            "median_peak_r": float(np.median(peaks)),
            "within_1s": int(np.sum(np.abs(lags) <= 1.0)),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--host", default=drift.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=drift.DEFAULT_PORT)
    parser.add_argument("--user", default=drift.DEFAULT_USER)
    parser.add_argument("--password", default=None)
    parser.add_argument("--config", default=None,
                        help="jitai_config.ini to read credentials from")
    parser.add_argument("--example", default=None,
                        help="participant for panel C (default: the clearest)")
    parser.add_argument("--max-lag", type=float, default=30.0)
    parser.add_argument("--minutes", type=float, default=5.0,
                        help="how much of the example session to draw")
    parser.add_argument("--output-dir", default="drift_check_outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style()

    user, password = drift.resolve_credentials(args)
    print(f"connecting to {args.host}:{args.port} as {user}")
    try:
        conn = drift.connect(args.host, args.port, user, password,
                             drift.STUDY_DB)
    except Exception as exc:                                    # noqa: BLE001
        print(f"\nCONNECTION FAILED: {exc}")
        print("  30071 is MariaDB; 30073 is the phpMyAdmin web page.")
        return 1

    try:
        sessions = drift.load_sessions(conn, None)
        print(f"{len(sessions)} session(s) with ground truth")
        if not sessions:
            return 1

        results = measure_all(conn, sessions, args.max_lag)
        if not results:
            print("nothing could be measured")
            return 1

        # Panel C shows one session. Default to the one with the clearest
        # correlation, which is the fairest thing to show as an illustration -
        # named in the caption either way so it is never mistaken for the
        # whole result. Panels A and B carry all 20.
        if args.example:
            example = next((s for s in sessions
                            if s.participant == args.example), None)
            if example is None:
                print(f"no session for {args.example}; using the clearest")
        else:
            example = None
        if example is None:
            best = max((r for r in results if r["convention"] == "start"),
                       key=lambda r: r["best_correlation"])
            example = next(s for s in sessions
                           if s.participant == best["participant"]
                           and s.study == best["study"])

        figure = plt.figure(figsize=(13, 9))
        grid = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0],
                                 hspace=0.58, wspace=0.20,
                                 left=0.07, right=0.97, top=0.865, bottom=0.09)
        panel_curves(figure.add_subplot(grid[0, 0]), results)
        panel_lags(figure.add_subplot(grid[0, 1]), results)
        panel_example(figure.add_subplot(grid[1, :]), conn, example,
                      args.minutes)

        summary = summarise(results)
        start = summary.get("start", {})
        end = summary.get("end", {})
        figure.suptitle(
            "The 8-second sensor/annotation shift is a timestamp convention, "
            "not clock drift",
            fontsize=15, fontweight="bold", color=INK, y=0.975)
        if start and end:
            figure.text(
                0.5, 0.935,
                f"Packet start: median lag {start['median_lag']:+.2f} s, "
                f"{start['within_1s']}/{start['n']} sessions within 1 s."
                f"      Packet end: {end['median_lag']:+.2f} s, "
                f"{end['within_1s']}/{end['n']} within 1 s.",
                ha="center", fontsize=10.5, color=INK_SECONDARY)
            figure.text(
                0.5, 0.908,
                f"Peak correlation is unchanged either way "
                f"({start['median_peak_r']:.3f} vs {end['median_peak_r']:.3f}) - "
                f"a displacement, not a degradation, which is what separates a "
                f"units error from genuine clock drift.",
                ha="center", fontsize=10.5, color=INK_SECONDARY)

        for extension in ("png", "pdf"):
            path = output_dir / f"drift_evidence.{extension}"
            figure.savefig(path, dpi=200, facecolor=SURFACE)
            print(f"wrote {path}")
        plt.close(figure)

        csv_path = output_dir / "drift_evidence_data.csv"
        write_csv(results, csv_path)
        print(f"wrote {csv_path}")

        print("\n  convention   n   median lag      range        median peak r"
              "   within +-1s")
        for convention in ("start", "end"):
            if convention not in summary:
                continue
            s = summary[convention]
            print(f"  {convention:<12}{s['n']:>3}  {s['median_lag']:>+9.2f} s  "
                  f"{s['min_lag']:>+6.2f}..{s['max_lag']:>+6.2f}  "
                  f"{s['median_peak_r']:>12.3f}   {s['within_1s']:>5}/{s['n']}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
