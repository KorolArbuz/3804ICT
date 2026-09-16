"""Aggregate completed independent benchmark sessions and plot session medians."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .benchmark_utils import (
    BENCHMARK_DIR,
    COLORS,
    DISPLAY_NAMES,
    FIGURES_DIR,
    IMPLEMENTATIONS,
    read_rows,
    summarize,
    write_rows,
)


def _session_rows(sessions_dir):
    sources = sorted(sessions_dir.glob("*/raw_prediction_runs.csv"))
    if not sources:
        raise FileNotFoundError(f"No completed sessions found under {sessions_dir}")
    output = []
    for source in sources:
        session_id = source.parent.name
        if not (source.parent / "environment.json").is_file():
            raise AssertionError(f"Session is missing environment metadata: {session_id}")
        raw = [row for row in read_rows(source) if row["warmup"] == "false"]
        by_implementation = {
            name: [row for row in raw if row["implementation"] == name]
            for name in IMPLEMENTATIONS
        }
        expected = {len(rows) for rows in by_implementation.values()}
        if len(expected) != 1 or not expected or next(iter(expected)) < 15:
            raise AssertionError(f"Incomplete session: {session_id}")
        by_trial = {
            (int(row["run_number_within_implementation"]), row["implementation"]): float(row["runtime_seconds"])
            for row in raw
        }
        for name in IMPLEMENTATIONS:
            values = [float(row["runtime_seconds"]) for row in by_implementation[name]]
            stats = summarize(values)
            ratios = []
            if name != "custom_python_v5_1":
                ratios = [
                    by_trial[(trial, "custom_python_v5_1")] / by_trial[(trial, name)]
                    for trial in sorted(
                        int(row["run_number_within_implementation"])
                        for row in by_implementation[name]
                    )
                ]
            output.append({
                "session_id": session_id,
                "implementation": name,
                "runs": stats["count"],
                "median": stats["median_seconds"],
                "IQR": stats["iqr_seconds"],
                "min": stats["min_seconds"],
                "max": stats["max_seconds"],
                "paired_speedup_vs_v5_1": 1.0 if name == "custom_python_v5_1" else summarize(ratios)["median_seconds"],
            })
    return output


def _plot(rows):
    session_ids = sorted({row["session_id"] for row in rows})
    x = list(range(len(session_ids)))
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    for name in IMPLEMENTATIONS:
        values = [
            next(float(row["median"]) for row in rows if row["session_id"] == session and row["implementation"] == name)
            for session in session_ids
        ]
        axis.plot(x, values, marker="o", linewidth=1.6, color=COLORS[name], label=DISPLAY_NAMES[name])
    axis.set_xticks(x, session_ids, rotation=20, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("Session median prediction time (seconds, log scale)")
    axis.set_title("Prediction runtime by independent benchmark session")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout(pad=1.5)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "runtime_by_session.png", dpi=220, facecolor="white")
    svg_path = FIGURES_DIR / "runtime_by_session.svg"
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, default=BENCHMARK_DIR / "sessions")
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "session_summary.csv")
    args = parser.parse_args(argv)
    rows = _session_rows(args.sessions_dir)
    write_rows(args.output, rows, tuple(rows[0]))
    _plot(rows)
    count = len({row["session_id"] for row in rows})
    print(f"Aggregated {count} real session(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
