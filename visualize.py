"""
Program.md v2 visualization script.

This script reads benchmark rows from benchmark_results.tsv and builds
pair-by-pair comparison plots for A..G across fairness tracks.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAIR_ORDER = ("A", "B", "C", "D", "E", "F", "G")
TRACK_ORDER = ("matched-F", "equal-D")
PAIR_LABELS = {
    "A": "A (t)",
    "B": "B (x)",
    "C": "C (x,y)",
    "D": "D (t,x)",
    "E": "E (t,x,y)",
    "F": "F (x,y,z)",
    "G": "G (t,x,y,z)",
}
TIME_ACTIVE_PAIRS = ("A", "D", "E", "G")

ROPE_METRICS = ("E_rel", "E_rope", "spectrum", "tail", "anisotropy")
TIME_METRICS = ("E_eta", "N_boost", "cone_sep")


@dataclass(frozen=True)
class ResultRow:
    idx: int
    commit: str
    benchmark_version: str
    pair: str
    track: str
    active_coords: str
    status: str
    description: str
    E_rel: float
    E_rope: float
    spectrum: float
    tail: float
    anisotropy: float
    E_eta: float
    N_boost: float
    cone_sep: float
    nonseparability: float

    @property
    def score(self) -> float:
        vals = [self.E_rel, self.E_rope, self.spectrum, self.tail, self.anisotropy]
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return float("inf")
        return float(sum(finite))


def parse_float(text: str) -> float:
    if text in {"NA", "", "nan", "NaN"}:
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def read_rows(path: Path) -> list[ResultRow]:
    out: list[ResultRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, r in enumerate(reader):
            out.append(
                ResultRow(
                    idx=idx,
                    commit=r["commit"],
                    benchmark_version=r["benchmark_version"],
                    pair=r["pair"],
                    track=r["track"],
                    active_coords=r["active_coords"],
                    status=r["status"],
                    description=r["description"],
                    E_rel=parse_float(r["E_rel"]),
                    E_rope=parse_float(r["E_rope"]),
                    spectrum=parse_float(r["spectrum"]),
                    tail=parse_float(r["tail"]),
                    anisotropy=parse_float(r["anisotropy"]),
                    E_eta=parse_float(r["E_eta"]),
                    N_boost=parse_float(r["N_boost"]),
                    cone_sep=parse_float(r["cone_sep"]),
                    nonseparability=parse_float(r["nonseparability"]),
                )
            )
    return out


def latest_benchmark_version(rows: list[ResultRow]) -> str:
    if not rows:
        raise ValueError("no rows found")
    return rows[-1].benchmark_version


def select_rows(
    rows: list[ResultRow],
    *,
    benchmark_version: str,
    statuses: set[str],
) -> list[ResultRow]:
    selected = [
        r
        for r in rows
        if r.benchmark_version == benchmark_version
        and r.pair in PAIR_ORDER
        and r.track in TRACK_ORDER
        and r.status in statuses
    ]
    if selected:
        return selected
    # Fallback to all statuses if requested statuses have no rows.
    return [
        r
        for r in rows
        if r.benchmark_version == benchmark_version
        and r.pair in PAIR_ORDER
        and r.track in TRACK_ORDER
    ]


def choose_best_by_pair_track(rows: list[ResultRow]) -> dict[tuple[str, str], ResultRow]:
    out: dict[tuple[str, str], ResultRow] = {}
    for r in rows:
        key = (r.pair, r.track)
        prev = out.get(key)
        if prev is None:
            out[key] = r
            continue
        # Prefer lower score; on ties prefer later row.
        if (r.score < prev.score) or (abs(r.score - prev.score) <= 1e-12 and r.idx > prev.idx):
            out[key] = r
    return out


def values_for_metric(
    best: dict[tuple[str, str], ResultRow],
    *,
    metric: str,
    track: str,
    pairs: tuple[str, ...] = PAIR_ORDER,
) -> np.ndarray:
    vals = []
    for pair in pairs:
        row = best.get((pair, track))
        if row is None:
            vals.append(np.nan)
        else:
            vals.append(float(getattr(row, metric)))
    return np.asarray(vals, dtype=np.float64)


def scores_for_track(
    best: dict[tuple[str, str], ResultRow],
    *,
    track: str,
    pairs: tuple[str, ...] = PAIR_ORDER,
) -> np.ndarray:
    vals = []
    for pair in pairs:
        row = best.get((pair, track))
        vals.append(np.nan if row is None else float(row.score))
    return np.asarray(vals, dtype=np.float64)


def _plot_grouped_bars(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    left_vals: np.ndarray,
    right_vals: np.ndarray,
    left_label: str = "matched-F",
    right_label: str = "equal-D",
) -> None:
    width = 0.38

    lmask = np.isfinite(left_vals)
    rmask = np.isfinite(right_vals)

    ax.bar(
        x[lmask] - width / 2,
        left_vals[lmask],
        width=width,
        label=left_label,
        color="#2a9d8f",
        alpha=0.9,
    )
    ax.bar(
        x[rmask] + width / 2,
        right_vals[rmask],
        width=width,
        label=right_label,
        color="#e76f51",
        alpha=0.9,
    )


def save_score_plot(
    *,
    best: dict[tuple[str, str], ResultRow],
    benchmark_version: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(PAIR_ORDER), dtype=np.int64)
    labels = [PAIR_LABELS[p] for p in PAIR_ORDER]

    matched = scores_for_track(best, track="matched-F")
    equal_d = scores_for_track(best, track="equal-D")

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    _plot_grouped_bars(ax, x=x, left_vals=matched, right_vals=equal_d)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Composite score (lower is better)")
    ax.set_title(f"Best Pair Comparison by Track | {benchmark_version}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_rope_metric_grid(
    *,
    best: dict[tuple[str, str], ResultRow],
    benchmark_version: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(PAIR_ORDER), dtype=np.int64)
    labels = [PAIR_LABELS[p] for p in PAIR_ORDER]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()

    for i, metric in enumerate(ROPE_METRICS):
        ax = axes[i]
        m = values_for_metric(best, metric=metric, track="matched-F")
        e = values_for_metric(best, metric=metric, track="equal-D")
        _plot_grouped_bars(ax, x=x, left_vals=m, right_vals=e)
        ax.set_title(metric)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].axis("off")
    fig.suptitle(f"RoPE-like Metrics by Pair | {benchmark_version}", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_time_metric_grid(
    *,
    best: dict[tuple[str, str], ResultRow],
    benchmark_version: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(TIME_ACTIVE_PAIRS), dtype=np.int64)
    labels = [PAIR_LABELS[p] for p in TIME_ACTIVE_PAIRS]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for i, metric in enumerate(TIME_METRICS):
        ax = axes[i]
        m = values_for_metric(best, metric=metric, track="matched-F", pairs=TIME_ACTIVE_PAIRS)
        e = values_for_metric(best, metric=metric, track="equal-D", pairs=TIME_ACTIVE_PAIRS)
        _plot_grouped_bars(ax, x=x, left_vals=m, right_vals=e)
        ax.set_title(metric)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right")

    fig.suptitle(f"Time-Active Structural Metrics | {benchmark_version}", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_markdown_summary(
    *,
    best: dict[tuple[str, str], ResultRow],
    benchmark_version: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Pair Comparison Summary ({benchmark_version})")
    lines.append("")
    lines.append(
        "| Pair | Track | Score | E_rel | E_rope | spectrum | tail | anisotropy | "
        "E_eta | N_boost | cone_sep | nonseparability | status | description |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")

    def f(v: float) -> str:
        if not np.isfinite(v):
            return "NA"
        return f"{v:.6f}"

    for pair in PAIR_ORDER:
        for track in TRACK_ORDER:
            row = best.get((pair, track))
            if row is None:
                lines.append(
                    f"| {pair} | {track} | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | missing | - |"
                )
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        pair,
                        track,
                        f(row.score),
                        f(row.E_rel),
                        f(row.E_rope),
                        f(row.spectrum),
                        f(row.tail),
                        f(row.anisotropy),
                        f(row.E_eta),
                        f(row.N_boost),
                        f(row.cone_sep),
                        f(row.nonseparability),
                        row.status,
                        row.description,
                    ]
                )
                + " |"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Program.md v2 pair comparisons from benchmark_results.tsv.")
    parser.add_argument(
        "--results-tsv",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results.tsv",
        help="path to benchmark TSV",
    )
    parser.add_argument(
        "--benchmark-version",
        type=str,
        default=None,
        help="benchmark version to visualize (default: latest in TSV)",
    )
    parser.add_argument(
        "--statuses",
        type=str,
        nargs="*",
        default=("keep", "ablation"),
        help="statuses eligible for selecting best rows (default: keep ablation)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "run_artifacts" / "plots_v2",
        help="directory for plots and summary",
    )
    args = parser.parse_args()

    rows = read_rows(args.results_tsv)
    if not rows:
        raise SystemExit(f"no rows found in {args.results_tsv}")

    version = args.benchmark_version or latest_benchmark_version(rows)
    selected = select_rows(rows, benchmark_version=version, statuses=set(args.statuses))
    if not selected:
        raise SystemExit(f"no rows matched benchmark version {version}")
    best = choose_best_by_pair_track(selected)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    score_plot = out_dir / f"pair_score_comparison_{version}.png"
    rope_plot = out_dir / f"pair_rope_metrics_{version}.png"
    time_plot = out_dir / f"pair_time_metrics_{version}.png"
    summary_md = out_dir / f"pair_summary_{version}.md"

    save_score_plot(best=best, benchmark_version=version, out_path=score_plot)
    save_rope_metric_grid(best=best, benchmark_version=version, out_path=rope_plot)
    save_time_metric_grid(best=best, benchmark_version=version, out_path=time_plot)
    save_markdown_summary(best=best, benchmark_version=version, out_path=summary_md)

    manifest_rows: dict[str, dict[str, object]] = {}
    for pair in PAIR_ORDER:
        for track in TRACK_ORDER:
            row = best.get((pair, track))
            key = f"{pair}:{track}"
            if row is None:
                manifest_rows[key] = {"missing": True}
                continue
            manifest_rows[key] = {
                "commit": row.commit,
                "status": row.status,
                "description": row.description,
                "score": row.score,
                "E_rel": row.E_rel,
                "E_rope": row.E_rope,
                "spectrum": row.spectrum,
                "tail": row.tail,
                "anisotropy": row.anisotropy,
                "E_eta": row.E_eta,
                "N_boost": row.N_boost,
                "cone_sep": row.cone_sep,
                "nonseparability": row.nonseparability,
            }

    manifest = {
        "benchmark_version": version,
        "results_tsv": str(args.results_tsv),
        "statuses_used": list(args.statuses),
        "out_dir": str(out_dir),
        "score_plot": str(score_plot),
        "rope_metrics_plot": str(rope_plot),
        "time_metrics_plot": str(time_plot),
        "summary_markdown": str(summary_md),
        "best_rows": manifest_rows,
    }
    manifest_path = out_dir / f"manifest_{version}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()

