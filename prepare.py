"""
Fixed utilities for the kernel-matching autoresearch harness.

This plays the role of Karpathy's prepare.py:
- builds the fixed position banks for the target context windows,
- precomputes and caches the axial RoPE reference curves,
- defines the deterministic mismatch metric used by experiment.py.

Usage:
    python prepare.py            # build / refresh the cached RoPE reference curves
    python prepare.py --refresh  # force a refresh of the cache
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Fixed experiment constants (do not edit during autoresearch)
# ---------------------------------------------------------------------------

TARGET_WINDOWS = (64, 512, 1024, 2048)
DEFAULT_DIM = 384                  # divisible by 12 => works for axial RoPE and F-MonSTER
REFERENCE_THETA_BASE = 10_000.0
AXIAL_COORD_DIMS = 3
SPECTRUM_COMPARE_BINS = 32
ENVELOPE_COMPARE_BINS = 16
ALIAS_TAIL_FRACTION = 0.75
EFFECTIVE_CONTEXT_THRESHOLD = 0.10

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionBank:
    length: int
    axial_shape: tuple[int, int, int]
    sequence_positions: np.ndarray  # (L,)
    axial_positions: np.ndarray     # (L, 3)
    monster_positions: np.ndarray   # (L, 4) with columns (t, x, y, z)


# ---------------------------------------------------------------------------
# Fixed geometry helpers
# ---------------------------------------------------------------------------

def prime_factors(n: int) -> list[int]:
    if n <= 1:
        return []
    factors: list[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d = 3 if d == 2 else d + 2
    if n > 1:
        factors.append(n)
    return factors


def near_cubic_shape(length: int, axes: int = AXIAL_COORD_DIMS) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("length must be positive")
    dims = [1] * axes
    for factor in sorted(prime_factors(length), reverse=True):
        idx = min(range(axes), key=lambda i: (dims[i], i))
        dims[idx] *= factor
    return tuple(sorted(dims))


@lru_cache(maxsize=None)
def get_position_bank(length: int) -> PositionBank:
    axial_shape = near_cubic_shape(length, AXIAL_COORD_DIMS)
    sequence_positions = np.arange(length, dtype=np.float64)
    axial_positions = np.stack(np.unravel_index(np.arange(length), axial_shape), axis=-1).astype(np.float64)
    monster_positions = np.concatenate((sequence_positions[:, None], axial_positions), axis=1)
    return PositionBank(
        length=length,
        axial_shape=tuple(int(x) for x in axial_shape),
        sequence_positions=sequence_positions,
        axial_positions=axial_positions,
        monster_positions=monster_positions,
    )


def make_probe_vectors(dim: int, mode: str = "basis", *, num_random: int = 1024, seed: int = 0) -> np.ndarray:
    """
    Fixed vector factory for debugging or alternative experiments.

    The default metric uses exact block-transform autocorrelations instead of Monte Carlo,
    but these probes are convenient if an agent wants to inspect encoded vectors manually.
    """
    mode = mode.lower()
    if mode == "basis":
        return np.eye(dim, dtype=np.float64)
    if mode == "gaussian":
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((num_random, dim))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
        return vecs
    raise ValueError(f"unknown probe mode: {mode}")


def base_frequencies(num_freq: int, theta_base: float = REFERENCE_THETA_BASE) -> np.ndarray:
    if num_freq <= 0:
        raise ValueError("num_freq must be positive")
    ranks = np.arange(num_freq, dtype=np.float64) / float(num_freq)
    return theta_base ** (-ranks)


def warped_frequencies(
    num_freq: int,
    *,
    theta_base: float,
    exponent: float = 1.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Rope-style inverse frequencies with an optional rank warp.

    exponent=1.0 reproduces the standard RoPE spacing:
        inv_freq[i] = theta_base ** (-(i / num_freq))
    """
    if num_freq <= 0:
        raise ValueError("num_freq must be positive")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    ranks = np.arange(num_freq, dtype=np.float64) / float(num_freq)
    warped_ranks = ranks ** exponent
    return float(scale) * (theta_base ** (-warped_ranks))


# ---------------------------------------------------------------------------
# Exact kernel utilities
# ---------------------------------------------------------------------------

def relative_curve_from_block_transforms(transforms: np.ndarray) -> np.ndarray:
    """
    Compute the exact relative-position similarity curve from per-position block transforms.

    Args:
        transforms: array with shape (L, B, block_dim, block_dim)

    Returns:
        curve[delta] = average over blocks and valid position pairs of
                       trace(T_p^T T_{p+delta}) / block_dim

    This is equivalent to averaging the encoded dot product over the standard basis,
    but it is deterministic and much faster than explicit vector Monte Carlo.
    """
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 4:
        raise ValueError("transforms must have shape (L, B, block_dim, block_dim)")

    length, num_blocks, block_dim, block_dim_2 = transforms.shape
    if block_dim != block_dim_2:
        raise ValueError("block transforms must be square")
    if length <= 0 or num_blocks <= 0:
        raise ValueError("empty transform array")

    seqs = transforms.transpose(1, 2, 3, 0).reshape(num_blocks * block_dim * block_dim, length)
    nfft = 1 << (2 * length - 1).bit_length()
    spectrum = np.fft.rfft(seqs, n=nfft, axis=-1)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft, axis=-1)[..., :length]
    summed = autocorr.sum(axis=0).real
    counts = np.arange(length, 0, -1, dtype=np.float64)
    curve = summed / counts / float(num_blocks * block_dim)
    return curve


def _bin_edges(max_delta: int, num_bins: int) -> np.ndarray:
    if max_delta <= 1:
        return np.array([1, max_delta], dtype=int)
    raw = np.geomspace(1, max_delta, num=num_bins + 1)
    edges = np.unique(np.clip(np.round(raw).astype(int), 1, max_delta))
    if edges[0] != 1:
        edges = np.insert(edges, 0, 1)
    if edges[-1] != max_delta:
        edges = np.append(edges, max_delta)
    return edges


def envelope_descriptor(curve: np.ndarray, num_bins: int = ENVELOPE_COMPARE_BINS) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    max_delta = curve.size - 1
    if max_delta <= 0:
        return np.zeros(1, dtype=np.float64)

    edges = _bin_edges(max_delta, num_bins)
    values = []
    abs_curve = np.abs(curve)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < lo:
            continue
        segment = abs_curve[lo : hi + 1]
        if segment.size == 0:
            continue
        values.append(float(segment.mean()))
    return np.asarray(values, dtype=np.float64)


def normalized_power_spectrum(curve: np.ndarray, num_bins: int = SPECTRUM_COMPARE_BINS) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    signal = curve[1:] if curve.size > 1 else curve
    if signal.size == 0:
        return np.zeros(num_bins, dtype=np.float64)

    signal = signal - signal.mean()
    power = np.abs(np.fft.rfft(signal)) ** 2
    if power.size == 0:
        return np.zeros(num_bins, dtype=np.float64)

    # Re-bin to a fixed width so every run gets a comparable descriptor.
    bins = np.array_split(power, num_bins)
    rebinned = np.array([float(b.mean()) if b.size else 0.0 for b in bins], dtype=np.float64)
    total = float(rebinned.sum())
    return rebinned / total if total > 0 else rebinned


def alias_peak(curve: np.ndarray, tail_fraction: float = ALIAS_TAIL_FRACTION) -> float:
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size <= 1:
        return 0.0
    max_delta = curve.size - 1
    start = max(1, int(math.floor(tail_fraction * max_delta)))
    return float(np.max(np.abs(curve[start:])))


def effective_context_range(curve: np.ndarray) -> float:
    """
    Distance at which the curve has completed 90% of its total drop from k(0) toward the tail mean.

    This is a window-specific proxy for "how far out the kernel meaningfully changes" even when the
    reference never decays near zero.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size <= 1:
        return 0.0

    deltas = np.arange(curve.size, dtype=np.float64)
    tail_start = max(1, int(math.floor(ALIAS_TAIL_FRACTION * (curve.size - 1))))
    tail_mean = float(np.mean(curve[tail_start:]))
    total_drop = float(curve[0] - tail_mean)
    if abs(total_drop) < 1e-12:
        return 0.0

    target = float(curve[0] - 0.9 * total_drop)
    if total_drop > 0:
        hits = np.where(curve <= target)[0]
    else:
        hits = np.where(curve >= target)[0]
    return float(hits[0]) if hits.size else float(curve.size - 1)


def compare_curves(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} vs {reference.shape}")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("candidate curve contains non-finite values")

    diff = candidate - reference
    curve_rmse = float(np.sqrt(np.mean(diff ** 2)))
    curve_mae = float(np.mean(np.abs(diff)))
    max_abs_error = float(np.max(np.abs(diff)))
    zero_delta_abs_error = float(abs(diff[0]))

    cand_env = envelope_descriptor(candidate)
    ref_env = envelope_descriptor(reference)
    envelope_rmse = float(np.sqrt(np.mean((cand_env - ref_env) ** 2)))

    cand_spec = normalized_power_spectrum(candidate)
    ref_spec = normalized_power_spectrum(reference)
    spectrum_rmse = float(np.sqrt(np.mean((cand_spec - ref_spec) ** 2)))

    cand_alias = alias_peak(candidate)
    ref_alias = alias_peak(reference)
    alias_peak_abs_error = float(abs(cand_alias - ref_alias))

    cand_ecr = effective_context_range(candidate)
    ref_ecr = effective_context_range(reference)
    ecr_error = float(abs(cand_ecr - ref_ecr) / max(1, candidate.size - 1))

    # Weighted scalar objective. Lower is better.
    score = (
        0.40 * curve_rmse
        + 0.20 * envelope_rmse
        + 0.20 * spectrum_rmse
        + 0.10 * alias_peak_abs_error
        + 0.10 * ecr_error
    )

    return {
        "score": float(score),
        "curve_rmse": curve_rmse,
        "curve_mae": curve_mae,
        "max_abs_error": max_abs_error,
        "zero_delta_abs_error": zero_delta_abs_error,
        "envelope_rmse": envelope_rmse,
        "spectrum_rmse": spectrum_rmse,
        "alias_peak_abs_error": alias_peak_abs_error,
        "candidate_alias_peak": cand_alias,
        "reference_alias_peak": ref_alias,
        "effective_context_candidate": float(cand_ecr),
        "effective_context_reference": float(ref_ecr),
        "effective_context_error": ecr_error,
    }


def aggregate_window_metrics(window_rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not window_rows:
        raise ValueError("need at least one window row to aggregate")
    worst = max(window_rows, key=lambda row: float(row["score"]))
    long_rows = [row for row in window_rows if int(row["length"]) >= 1024]
    if not long_rows:
        long_rows = window_rows

    def avg(key: str) -> float:
        return float(np.mean([float(row[key]) for row in window_rows]))

    def avg_long(key: str) -> float:
        return float(np.mean([float(row[key]) for row in long_rows]))

    return {
        "overall_score": avg("score"),
        "mean_curve_rmse": avg("curve_rmse"),
        "mean_curve_mae": avg("curve_mae"),
        "mean_envelope_rmse": avg("envelope_rmse"),
        "mean_spectrum_rmse": avg("spectrum_rmse"),
        "mean_alias_peak_abs_error": avg("alias_peak_abs_error"),
        "mean_effective_context_error": avg("effective_context_error"),
        "long_context_score": avg_long("score"),
        "worst_window": int(worst["length"]),
        "worst_window_score": float(worst["score"]),
        "worst_window_curve_rmse": float(worst["curve_rmse"]),
    }


# ---------------------------------------------------------------------------
# Reference caching
# ---------------------------------------------------------------------------

def reference_cache_path(
    *,
    dim: int = DEFAULT_DIM,
    theta_base: float = REFERENCE_THETA_BASE,
    windows: tuple[int, ...] = TARGET_WINDOWS,
) -> Path:
    window_tag = "-".join(str(w) for w in windows)
    theta_tag = format(theta_base, ".6g").replace(".", "p")
    return CACHE_DIR / f"rope_reference_dim{dim}_theta{theta_tag}_windows{window_tag}.npz"


@lru_cache(maxsize=None)
def get_reference_curves(
    *,
    dim: int = DEFAULT_DIM,
    theta_base: float = REFERENCE_THETA_BASE,
    windows: tuple[int, ...] = TARGET_WINDOWS,
    refresh: bool = False,
) -> dict[int, np.ndarray]:
    path = reference_cache_path(dim=dim, theta_base=theta_base, windows=windows)
    if path.exists() and not refresh:
        data = np.load(path)
        return {int(key[1:]): data[key] for key in data.files if key.startswith("w")}

    import rope  # local import to keep the harness acyclic

    curves: dict[int, np.ndarray] = {}
    payload: dict[str, np.ndarray] = {}
    for length in windows:
        bank = get_position_bank(length)
        curve = rope.relative_kernel(bank=bank, dim=dim, theta_base=theta_base)
        curves[int(length)] = curve
        payload[f"w{length}"] = curve

    np.savez(path, **payload)
    return curves


# ---------------------------------------------------------------------------
# Fixed evaluation entrypoint used by experiment.py
# ---------------------------------------------------------------------------

def evaluate_candidate(
    build_curve: Callable[[PositionBank, int], np.ndarray],
    *,
    dim: int = DEFAULT_DIM,
    theta_base: float = REFERENCE_THETA_BASE,
    windows: tuple[int, ...] = TARGET_WINDOWS,
    candidate_name: str = "candidate",
) -> dict[str, object]:
    reference_curves = get_reference_curves(dim=dim, theta_base=theta_base, windows=windows)
    window_rows: list[dict[str, object]] = []
    candidate_curves: dict[int, np.ndarray] = {}

    for length in windows:
        bank = get_position_bank(length)
        reference = reference_curves[int(length)]
        candidate = np.asarray(build_curve(bank, dim), dtype=np.float64)
        metrics = compare_curves(candidate, reference)
        row = {
            "length": int(length),
            "axial_shape": list(bank.axial_shape),
            **metrics,
        }
        window_rows.append(row)
        candidate_curves[int(length)] = candidate

    aggregate = aggregate_window_metrics(window_rows)
    return {
        "meta": {
            "candidate_name": candidate_name,
            "dim": int(dim),
            "reference_theta_base": float(theta_base),
            "windows": [int(w) for w in windows],
        },
        "aggregate": aggregate,
        "windows": window_rows,
        "curves": {
            "reference": reference_curves,
            "candidate": candidate_curves,
        },
    }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def save_report(report: dict[str, object], *, report_path: Path, curves_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.parent.mkdir(parents=True, exist_ok=True)

    curve_payload: dict[str, np.ndarray] = {}
    for length, curve in report["curves"]["reference"].items():
        curve_payload[f"reference_{length}"] = np.asarray(curve, dtype=np.float64)
    for length, curve in report["curves"]["candidate"].items():
        curve_payload[f"candidate_{length}"] = np.asarray(curve, dtype=np.float64)
    np.savez(curves_path, **curve_payload)

    jsonable = {
        "meta": report["meta"],
        "aggregate": report["aggregate"],
        "windows": report["windows"],
        "artifacts": {
            "curves_npz": str(curves_path),
        },
    }
    report_path.write_text(json.dumps(jsonable, indent=2) + "\n", encoding="utf-8")


def print_report(report: dict[str, object], *, report_path: Path | None = None, curves_path: Path | None = None) -> None:
    meta = report["meta"]
    aggregate = report["aggregate"]
    rows = report["windows"]

    print("Meta:")
    print(f"  candidate_name:       {meta['candidate_name']}")
    print(f"  dim:                  {meta['dim']}")
    print(f"  reference_theta_base: {meta['reference_theta_base']}")
    print(f"  windows:              {meta['windows']}")
    print()

    print("Per-window diagnostics:")
    header = (
        "window  axial_shape  score      curve_rmse  envelope   spectrum   alias_err  "
        "ecr_cand  ecr_ref"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        shape = "x".join(str(x) for x in row["axial_shape"])
        print(
            f"{int(row['length']):>6d}  "
            f"{shape:>10s}  "
            f"{float(row['score']):>9.6f}  "
            f"{float(row['curve_rmse']):>10.6f}  "
            f"{float(row['envelope_rmse']):>9.6f}  "
            f"{float(row['spectrum_rmse']):>9.6f}  "
            f"{float(row['alias_peak_abs_error']):>9.6f}  "
            f"{float(row['effective_context_candidate']):>8.1f}  "
            f"{float(row['effective_context_reference']):>7.1f}"
        )
    print()

    print("---")
    print(f"overall_score:          {float(aggregate['overall_score']):.6f}")
    print(f"mean_curve_rmse:        {float(aggregate['mean_curve_rmse']):.6f}")
    print(f"long_context_score:     {float(aggregate['long_context_score']):.6f}")
    print(f"worst_window:           {int(aggregate['worst_window'])}")
    print(f"worst_window_score:     {float(aggregate['worst_window_score']):.6f}")
    print(f"worst_window_curve_rmse:{float(aggregate['worst_window_curve_rmse']):.6f}")

    for row in rows:
        length = int(row["length"])
        print(f"window_{length}_score:       {float(row['score']):.6f}")
        print(f"window_{length}_curve_rmse:  {float(row['curve_rmse']):.6f}")
        print(f"window_{length}_alias_error: {float(row['alias_peak_abs_error']):.6f}")
        print(f"window_{length}_ecr_cand:    {float(row['effective_context_candidate']):.1f}")
        print(f"window_{length}_ecr_ref:     {float(row['effective_context_reference']):.1f}")

    if report_path is not None:
        print(f"report_json:            {report_path}")
    if curves_path is not None:
        print(f"curves_npz:             {curves_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the fixed axial RoPE reference curves.")
    parser.add_argument("--refresh", action="store_true", help="Ignore any cached curves and rebuild them.")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Embedding dimension used for the kernel benchmark.")
    parser.add_argument("--theta-base", type=float, default=REFERENCE_THETA_BASE, help="Reference theta base.")
    args = parser.parse_args()

    curves = get_reference_curves(
        dim=int(args.dim),
        theta_base=float(args.theta_base),
        windows=TARGET_WINDOWS,
        refresh=bool(args.refresh),
    )
    path = reference_cache_path(dim=int(args.dim), theta_base=float(args.theta_base), windows=TARGET_WINDOWS)

    print(f"Cache path: {path}")
    print()
    for length in TARGET_WINDOWS:
        bank = get_position_bank(length)
        curve = curves[int(length)]
        print(
            f"window={length:>4d} | axial_shape={bank.axial_shape} | "
            f"curve_len={curve.size} | k(0)={curve[0]:.6f} | ecr={effective_context_range(curve):.1f}"
        )
    print()
    print("Done.")


if __name__ == "__main__":
    main()
