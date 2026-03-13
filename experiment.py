"""
Kernel-matching experiment runner.

This is the agent-edited file in the minimal autoresearch harness.
It mirrors the role of Karpathy's train.py, but the objective is the
kernel mismatch between F-MonSTER and the fixed axial RoPE reference.

Usage:
    python experiment.py
    python experiment.py --plots
    python experiment.py --plots --plot-windows 64 2048
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import f_monster
import prepare

# ---------------------------------------------------------------------------
# Editable experiment settings
# ---------------------------------------------------------------------------

DIM = prepare.DEFAULT_DIM
REFERENCE_THETA_BASE = prepare.REFERENCE_THETA_BASE
WINDOWS = prepare.TARGET_WINDOWS

CANDIDATE_NAME = "dual_plane_cycle_boost11_rot10"

# Edit these directly during autoresearch.
CANDIDATE = f_monster.MonsterConfig(
    span=2.0 * 3.141592653589793,
    top_delta=2048.0,
    unit=1.0,
    theta_base=10_000.0,
    freq_scale=1.0,
    freq_exponent=1.0,
    boost_scale=1.1,
    rotation_scale=1.0,
    axis_mode="cycle",
    axis_blend=1.0,
    block_mode="dual_plane",
)

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

RUN_ARTIFACTS = Path(__file__).resolve().parent / "run_artifacts"
REPORT_PATH = RUN_ARTIFACTS / "latest_report.json"
CURVES_PATH = RUN_ARTIFACTS / "latest_curves.npz"
PLOT_DIR = RUN_ARTIFACTS / "plots"


def build_candidate_curve(bank: prepare.PositionBank, dim: int):
    return f_monster.relative_kernel(bank=bank, dim=dim, config=CANDIDATE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the kernel-matching F-MonSTER experiment.")
    parser.add_argument(
        "--plots",
        action="store_true",
        help="also generate comparison dashboards under run_artifacts/plots",
    )
    parser.add_argument(
        "--plot-windows",
        type=int,
        nargs="*",
        default=None,
        help="optional subset of windows for plot generation",
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=0,
        help="random seed for the fixed Q/K vectors used in similarity-matrix plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Make sure the fixed reference exists.
    prepare.get_reference_curves(dim=DIM, theta_base=REFERENCE_THETA_BASE, windows=WINDOWS)

    report = prepare.evaluate_candidate(
        build_candidate_curve,
        dim=DIM,
        theta_base=REFERENCE_THETA_BASE,
        windows=WINDOWS,
        candidate_name=CANDIDATE_NAME,
    )
    prepare.save_report(report, report_path=REPORT_PATH, curves_path=CURVES_PATH)

    print("Candidate config:")
    for key, value in asdict(CANDIDATE).items():
        print(f"  {key:16s}: {value}")
    print()

    prepare.print_report(report, report_path=REPORT_PATH, curves_path=CURVES_PATH)

    if args.plots:
        import visualize

        plot_windows = tuple(args.plot_windows) if args.plot_windows else tuple(WINDOWS)
        if plot_windows == tuple(WINDOWS):
            plot_report = report
        else:
            plot_report = prepare.evaluate_candidate(
                build_candidate_curve,
                dim=DIM,
                theta_base=REFERENCE_THETA_BASE,
                windows=plot_windows,
                candidate_name=CANDIDATE_NAME,
            )

        manifest = visualize.save_all_plots(
            report=plot_report,
            dim=DIM,
            theta_base=REFERENCE_THETA_BASE,
            config=CANDIDATE,
            candidate_name=CANDIDATE_NAME,
            windows=plot_windows,
            out_dir=PLOT_DIR,
            seed=args.plot_seed,
        )
        print(f"plot_manifest:          {manifest['manifest_path']}")
        print(f"plot_summary:           {manifest['summary_curve_grid']}")
        for length in plot_windows:
            path = manifest["dashboards"][str(length)]
            print(f"plot_window_{length}:       {path}")

    elapsed = time.time() - t0
    print(f"total_seconds:          {elapsed:.3f}")


if __name__ == "__main__":
    main()
