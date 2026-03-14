"""
Program.md v2 experiment runner.

Usage examples:
    uv run python experiment.py
    uv run python experiment.py --pair C --track matched-F --F 96
    uv run python experiment.py --pair G --track equal-D --D 384 --description "time-active probe"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

import benchmark_v2

RUN_ARTIFACTS = Path(__file__).resolve().parent / "run_artifacts"
REPORT_PATH = RUN_ARTIFACTS / "v2_latest_report.json"
RESULTS_TSV = Path(__file__).resolve().parent / "benchmark_results.tsv"
NUMERICAL_BLOWUP_GATE = 1e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Program.md v2 MonSTER benchmark.")
    parser.add_argument(
        "--pair",
        type=str,
        default=benchmark_v2.DEFAULT_PAIR,
        choices=sorted(benchmark_v2.PAIRINGS.keys()),
        help="pairing ID (A..G)",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=benchmark_v2.DEFAULT_TRACK,
        choices=("matched-F", "equal-D"),
        help="fairness track",
    )
    parser.add_argument("--F", type=int, default=benchmark_v2.DEFAULT_F, help="frequency count for matched-F track")
    parser.add_argument("--D", type=int, default=benchmark_v2.DEFAULT_D, help="embedding width for equal-D track")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="*",
        default=list(benchmark_v2.DEFAULT_WINDOWS),
        help="context windows (default: 64 512 1024 2048)",
    )
    parser.add_argument(
        "--theta-base",
        type=float,
        default=prepare_theta_default(),
        help="base for frequency schedule",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument(
        "--max-pairs-per-window",
        type=int,
        default=200_000,
        help="maximum sampled pairs per window for coordinate-space shell metrics",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="phase-1 calibration run",
        help="short run description for TSV logging",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="auto",
        choices=("auto", "keep", "discard", "ablation", "crash"),
        help="status override; use auto for heuristic decision",
    )
    parser.add_argument(
        "--results-tsv",
        type=Path,
        default=RESULTS_TSV,
        help="path to benchmark TSV log",
    )
    parser.add_argument("--freq-scale", type=float, default=1.0, help="MonSTER frequency scale")
    parser.add_argument("--freq-exponent", type=float, default=1.0, help="MonSTER frequency rank exponent")
    parser.add_argument("--boost-scale", type=float, default=1.0, help="MonSTER boost scale")
    parser.add_argument("--rotation-scale", type=float, default=1.0, help="MonSTER rotation scale")
    parser.add_argument("--axis-mode", type=str, default=None, help="optional MonSTER axis mode override")
    parser.add_argument("--axis-blend", type=float, default=1.0, help="MonSTER axis blend (for hybrid mode)")
    parser.add_argument("--block-mode", type=str, default="lorentz", help="MonSTER block mode")
    return parser.parse_args()


def prepare_theta_default() -> float:
    # Deferred import to keep startup cheap.
    import prepare

    return float(prepare.REFERENCE_THETA_BASE)


def current_commit_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def rope_like_scalar(row: dict[str, object]) -> float:
    keys = ("E_rel", "E_rope", "spectrum", "tail", "anisotropy")
    vals = []
    for k in keys:
        v = float(row[k])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("inf")
    return float(sum(vals))


def choose_status_auto(row: dict[str, object], *, tsv_path: Path) -> str:
    if bool(row["collapse_fail"]):
        return "discard"
    if row_has_numerical_failure(row):
        return "discard"
    if not tsv_path.exists():
        return "keep"

    best_score = float("inf")
    best_n_boost = float("-inf")
    with tsv_path.open("r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 20:
                continue
            benchmark_version = parts[1]
            pair = parts[2]
            track = parts[3]
            status = parts[18]
            if benchmark_version != row["benchmark_version"]:
                continue
            if pair != row["pair"] or track != row["track"]:
                continue
            if status not in {"keep", "ablation"}:
                continue
            e_rel = parse_float(parts[8])
            e_rope = parse_float(parts[9])
            spectrum = parse_float(parts[10])
            tail = parse_float(parts[11])
            anis = parse_float(parts[12])
            n_boost = parse_float(parts[14])
            score = sum(v for v in (e_rel, e_rope, spectrum, tail, anis) if np.isfinite(v))
            best_score = min(best_score, score)
            if np.isfinite(n_boost):
                best_n_boost = max(best_n_boost, n_boost)

    my_score = rope_like_scalar(row)
    if best_score == float("inf"):
        return "keep"

    if bool(row["time_active"]):
        n_boost = float(row["N_boost"])
        boost_ok = (not np.isfinite(best_n_boost)) or (np.isfinite(n_boost) and n_boost >= 0.5 * best_n_boost)
        if my_score < best_score and boost_ok:
            return "keep"
        return "discard"

    return "keep" if my_score < best_score else "discard"


def row_has_numerical_failure(row: dict[str, object]) -> bool:
    core = ("E_rel", "E_rope", "spectrum", "tail", "anisotropy", "nonseparability")
    for k in core:
        v = float(row[k])
        if (not np.isfinite(v)) or abs(v) > NUMERICAL_BLOWUP_GATE:
            return True

    if bool(row["time_active"]):
        for k in ("E_eta", "N_boost"):
            v = float(row[k])
            if (not np.isfinite(v)) or abs(v) > NUMERICAL_BLOWUP_GATE:
                return True
        cone = float(row["cone_sep"])
        if np.isfinite(cone) and abs(cone) > NUMERICAL_BLOWUP_GATE:
            return True

    return False


def parse_float(text: str) -> float:
    if text in {"NA", "", "nan", "NaN"}:
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    spec = benchmark_v2.PAIRINGS[args.pair]
    windows = benchmark_v2.parse_windows(args.windows)

    t0 = time.time()
    row = benchmark_v2.evaluate_pairing(
        spec=spec,
        track=args.track,
        F=int(args.F),
        D=int(args.D),
        windows=windows,
        theta_base=float(args.theta_base),
        rng_seed=int(args.seed),
        max_pairs_per_window=int(args.max_pairs_per_window),
        monster_overrides=monster_overrides_from_args(args),
    )
    elapsed = time.time() - t0

    status = args.status
    if status == "auto":
        status = choose_status_auto(row, tsv_path=args.results_tsv)

    commit = current_commit_short()
    benchmark_v2.append_tsv_row(
        args.results_tsv,
        row,
        commit=commit,
        status=status,
        description=args.description,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        **row,
        "status": status,
        "description": args.description,
        "commit": commit,
        "elapsed_seconds": elapsed,
    }
    REPORT_PATH.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print("Meta:")
    print(f"  benchmark_version: {row['benchmark_version']}")
    print(f"  pair:              {row['pair']}")
    print(f"  track:             {row['track']}")
    print(f"  active_coords:     {row['active_coords']}")
    print(f"  F:                 {row['F']}")
    print(f"  D_rope:            {row['D_rope']}")
    print(f"  D_monster:         {row['D_monster']}")
    print()

    print("MonSTER config:")
    for k, v in row["monster_config"].items():
        print(f"  {k:16s}: {v}")
    print()

    print("Aggregate metrics:")
    print(f"  E_rel:             {row['E_rel']:.6f}")
    print(f"  E_rope:            {format_metric(row['E_rope'])}")
    print(f"  spectrum:          {row['spectrum']:.6f}")
    print(f"  tail:              {row['tail']:.6f}")
    print(f"  anisotropy:        {row['anisotropy']:.6f}")
    print(f"  E_eta:             {format_metric(row['E_eta'])}")
    print(f"  N_boost:           {format_metric(row['N_boost'])}")
    print(f"  cone_sep:          {format_metric(row['cone_sep'])}")
    print(f"  nonseparability:   {row['nonseparability']:.6f}")
    print(f"  collapse_fail:     {row['collapse_fail']}")
    print()

    print("Per-window metrics:")
    for wr in row["windows"]:
        print(
            f"  L={int(wr['length']):4d}  "
            f"E_rel={float(wr['E_rel']):.6f}  "
            f"E_rope={format_metric(wr['E_rope'])}  "
            f"spectrum={float(wr['spectrum']):.6f}  "
            f"tail={float(wr['tail']):.6f}  "
            f"anis={float(wr['anisotropy']):.6f}  "
            f"E_eta={format_metric(wr['E_eta'])}  "
            f"N_boost={format_metric(wr['N_boost'])}  "
            f"cone_sep={format_metric(wr['cone_sep'])}  "
            f"nonsep={float(wr['nonseparability']):.6f}"
        )

    print()
    print(f"status:              {status}")
    print(f"description:         {args.description}")
    print(f"results_tsv:         {args.results_tsv}")
    print(f"report_json:         {REPORT_PATH}")
    print(f"total_seconds:       {elapsed:.3f}")


def format_metric(x: object) -> str:
    v = float(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.6f}"


def monster_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {
        "freq_scale": float(args.freq_scale),
        "freq_exponent": float(args.freq_exponent),
        "boost_scale": float(args.boost_scale),
        "rotation_scale": float(args.rotation_scale),
        "axis_blend": float(args.axis_blend),
        "block_mode": str(args.block_mode),
    }
    if args.axis_mode is not None:
        out["axis_mode"] = str(args.axis_mode)
    return out


if __name__ == "__main__":
    main()
