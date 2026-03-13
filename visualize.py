"""
Fixed plotting helpers for the kernel-matching autoresearch harness.

These utilities generate comparison dashboards that mirror the usual RoPE
visualizations, but for the actual axial-RoPE-vs-F-MonSTER setup used by the
fixed evaluation harness.

The main entrypoint is `save_all_plots(...)`, which writes one dashboard per
window plus a compact all-windows kernel-curve summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import f_monster
import prepare
import rope


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)
    return x / norms


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    return x / max(norm, 1e-12)


def _encode_blocks(transforms: np.ndarray, block_inputs: np.ndarray) -> np.ndarray:
    """
    Apply per-position block transforms to a repeated block input.

    Args:
        transforms: (L, B, M, M)
        block_inputs: (B, M) or (L, B, M)

    Returns:
        encoded: (L, B, M)
    """
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 4:
        raise ValueError("transforms must have shape (L, B, M, M)")

    length, num_blocks, block_dim, _ = transforms.shape
    block_inputs = np.asarray(block_inputs, dtype=np.float64)

    if block_inputs.ndim == 2:
        if block_inputs.shape != (num_blocks, block_dim):
            raise ValueError(
                f"block input shape mismatch: expected {(num_blocks, block_dim)}, got {block_inputs.shape}"
            )
        x = np.broadcast_to(block_inputs[None, :, :], (length, num_blocks, block_dim))
    elif block_inputs.ndim == 3:
        if block_inputs.shape != (length, num_blocks, block_dim):
            raise ValueError(
                f"block input shape mismatch: expected {(length, num_blocks, block_dim)}, got {block_inputs.shape}"
            )
        x = block_inputs
    else:
        raise ValueError("block_inputs must have shape (B, M) or (L, B, M)")

    return np.einsum("lbmn,lbn->lbm", transforms, x)


def _encode_vector(transforms: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Encode the same full embedding vector at every position."""
    transforms = np.asarray(transforms, dtype=np.float64)
    length, num_blocks, block_dim, _ = transforms.shape
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or vector.size != num_blocks * block_dim:
        raise ValueError(
            f"vector must have shape ({num_blocks * block_dim},), got {vector.shape}"
        )
    block_inputs = vector.reshape(num_blocks, block_dim)
    encoded = _encode_blocks(transforms, block_inputs)
    return encoded.reshape(length, num_blocks * block_dim)


def fixed_query_key(dim: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q0 = _normalize_vector(rng.standard_normal(dim))
    k0 = _normalize_vector(rng.standard_normal(dim))
    return q0, k0


def similarity_matrix_from_transforms(
    transforms: np.ndarray,
    *,
    q0: np.ndarray,
    k0: np.ndarray,
) -> np.ndarray:
    q_enc = _encode_vector(transforms, q0)
    k_enc = _encode_vector(transforms, k0)
    return q_enc @ k_enc.T


def _tangent_frames(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a deterministic orthonormal frame (u, v) tangent to each axis.

    For a spatial rotation around axis `a`, a probe along `u` rotates in the
    local (u, v) plane as `(cos θ, sin θ)`, which is the closest analogue to the
    usual RoPE `(1, 0) -> (cos θ, sin θ)` probe.
    """
    axes = _normalize_rows(np.asarray(axes, dtype=np.float64))
    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (axes.shape[0], 1))
    near_parallel = np.abs(np.sum(axes * ref, axis=-1)) > 0.90
    ref[near_parallel] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    u = np.cross(ref, axes)
    u = _normalize_rows(u)
    v = np.cross(axes, u)
    v = _normalize_rows(v)
    return u, v


def rope_phase_probe_heatmap_from_transforms(transforms: np.ndarray) -> np.ndarray:
    """
    RoPE probe analogous to the notebook's `(1, 0)` pair probe.

    Returns an array shaped (L, D) with channels ordered as `(sin, cos)` for each pair.
    """
    length, num_blocks, block_dim, _ = transforms.shape
    if block_dim != 2:
        raise ValueError("RoPE phase probe expects 2x2 block transforms")

    probe = np.zeros((num_blocks, block_dim), dtype=np.float64)
    probe[:, 0] = 1.0
    encoded = _encode_blocks(transforms, probe)

    heat = np.empty((length, 2 * num_blocks), dtype=np.float64)
    heat[:, 0::2] = encoded[:, :, 1]  # sin
    heat[:, 1::2] = encoded[:, :, 0]  # cos
    return heat


def monster_phase_probe_heatmap_from_transforms(
    transforms: np.ndarray,
    *,
    axes: np.ndarray,
) -> np.ndarray:
    """
    F-MonSTER phase probe expressed in each block's local tangent-plane basis.

    Each 4D F-MonSTER block has one boost axis and one spatial rotation axis.
    To recover a RoPE-like pure oscillation plot, we probe with a spatial vector
    orthogonal to the block axis and then project the encoded result into the
    local tangent-plane basis `(u, v)`. That produces a `(cos θ, sin θ)` pair per
    frequency block, independent of the axis orientation in global xyz coordinates.

    Returns an array shaped (L, 2 * num_freq) with channels ordered as `(sin, cos)`.
    """
    length, num_blocks, block_dim, _ = transforms.shape
    if block_dim != 4:
        raise ValueError("F-MonSTER phase probe expects 4x4 block transforms")
    if axes.shape != (num_blocks, 3):
        raise ValueError(f"axes shape mismatch: expected {(num_blocks, 3)}, got {axes.shape}")

    u, v = _tangent_frames(axes)

    probe = np.zeros((num_blocks, 4), dtype=np.float64)
    probe[:, 1:] = u
    encoded = _encode_blocks(transforms, probe)

    u4 = np.zeros((num_blocks, 4), dtype=np.float64)
    v4 = np.zeros((num_blocks, 4), dtype=np.float64)
    u4[:, 1:] = u
    v4[:, 1:] = v

    cos = np.einsum("lbm,bm->lb", encoded, u4)
    sin = np.einsum("lbm,bm->lb", encoded, v4)

    heat = np.empty((length, 2 * num_blocks), dtype=np.float64)
    heat[:, 0::2] = sin
    heat[:, 1::2] = cos
    return heat


def robust_symmetric_limit(x: np.ndarray, quantile: float = 0.995) -> float:
    x = np.asarray(x, dtype=np.float64)
    limit = float(np.quantile(np.abs(x), quantile))
    return max(limit, 1e-12)


def _curve_metrics_by_window(report: dict[str, object]) -> dict[int, dict[str, object]]:
    return {int(row["length"]): row for row in report["windows"]}


def save_window_dashboard(
    *,
    length: int,
    axial_shape: tuple[int, ...],
    candidate_name: str,
    rope_probe: np.ndarray,
    monster_probe: np.ndarray,
    rope_similarity: np.ndarray,
    monster_similarity: np.ndarray,
    reference_curve: np.ndarray,
    candidate_curve: np.ndarray,
    metrics: dict[str, object],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diff_similarity = monster_similarity - rope_similarity
    rope_lim = robust_symmetric_limit(rope_similarity)
    monster_lim = robust_symmetric_limit(monster_similarity)
    diff_lim = robust_symmetric_limit(diff_similarity)

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    fig.suptitle(
        f"{candidate_name} vs axial RoPE | window={length} | axial_shape={'x'.join(map(str, axial_shape))}",
        fontsize=14,
    )

    # Row 1: probe heatmaps + kernel curve.
    ax_probe_rope = fig.add_subplot(gs[0, 0])
    ax_probe_monster = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[0, 2])

    im_probe_rope = ax_probe_rope.imshow(
        rope_probe,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    ax_probe_rope.set_title("RoPE phase probe")
    ax_probe_rope.set_xlabel("pair channel (sin / cos)")
    ax_probe_rope.set_ylabel("token position")
    fig.colorbar(im_probe_rope, ax=ax_probe_rope, fraction=0.046, pad=0.04)

    im_probe_monster = ax_probe_monster.imshow(
        monster_probe,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    ax_probe_monster.set_title("F-MonSTER phase probe (local tangent coordinates)")
    ax_probe_monster.set_xlabel("local phase channel (sin / cos)")
    ax_probe_monster.set_ylabel("token position")
    fig.colorbar(im_probe_monster, ax=ax_probe_monster, fraction=0.046, pad=0.04)

    offsets = np.arange(reference_curve.size, dtype=np.int64)
    ax_curve.plot(offsets, reference_curve, label="axial RoPE", linewidth=2.0)
    ax_curve.plot(offsets, candidate_curve, label="F-MonSTER", linewidth=1.8)
    ax_curve.set_title(
        "Average similarity vs relative offset\n"
        f"score={float(metrics['score']):.4f} | curve_rmse={float(metrics['curve_rmse']):.4f} | "
        f"alias_err={float(metrics['alias_peak_abs_error']):.4f}"
    )
    ax_curve.set_xlabel("relative offset Δ")
    ax_curve.set_ylabel("K(Δ)")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="best")

    # Row 2: similarity matrices.
    ax_sim_rope = fig.add_subplot(gs[1, 0])
    ax_sim_monster = fig.add_subplot(gs[1, 1])
    ax_sim_diff = fig.add_subplot(gs[1, 2])

    im_rope = ax_sim_rope.imshow(
        rope_similarity,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
        cmap="coolwarm",
        vmin=-rope_lim,
        vmax=rope_lim,
    )
    ax_sim_rope.set_title(f"RoPE similarity matrix\nclipped to ±{rope_lim:.3g}")
    ax_sim_rope.set_xlabel("key position q")
    ax_sim_rope.set_ylabel("query position p")
    fig.colorbar(im_rope, ax=ax_sim_rope, fraction=0.046, pad=0.04)

    im_monster = ax_sim_monster.imshow(
        monster_similarity,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
        cmap="coolwarm",
        vmin=-monster_lim,
        vmax=monster_lim,
    )
    ax_sim_monster.set_title(f"F-MonSTER similarity matrix\nclipped to ±{monster_lim:.3g}")
    ax_sim_monster.set_xlabel("key position q")
    ax_sim_monster.set_ylabel("query position p")
    fig.colorbar(im_monster, ax=ax_sim_monster, fraction=0.046, pad=0.04)

    im_diff = ax_sim_diff.imshow(
        diff_similarity,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
        cmap="coolwarm",
        vmin=-diff_lim,
        vmax=diff_lim,
    )
    ax_sim_diff.set_title(f"Similarity difference (F-MonSTER − RoPE)\nclipped to ±{diff_lim:.3g}")
    ax_sim_diff.set_xlabel("key position q")
    ax_sim_diff.set_ylabel("query position p")
    fig.colorbar(im_diff, ax=ax_sim_diff, fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_curve_summary(
    *,
    report: dict[str, object],
    candidate_name: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(report["windows"], key=lambda row: int(row["length"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    fig.suptitle(f"Kernel curves by window | {candidate_name} vs axial RoPE", fontsize=14)

    for ax, row in zip(axes, rows):
        length = int(row["length"])
        reference_curve = np.asarray(report["curves"]["reference"][length], dtype=np.float64)
        candidate_curve = np.asarray(report["curves"]["candidate"][length], dtype=np.float64)
        offsets = np.arange(reference_curve.size, dtype=np.int64)

        ax.plot(offsets, reference_curve, label="axial RoPE", linewidth=2.0)
        ax.plot(offsets, candidate_curve, label="F-MonSTER", linewidth=1.8)
        ax.set_title(
            f"window={length} | shape={'x'.join(map(str, row['axial_shape']))}\n"
            f"score={float(row['score']):.4f} | curve_rmse={float(row['curve_rmse']):.4f}"
        )
        ax.set_xlabel("relative offset Δ")
        ax.set_ylabel("K(Δ)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_all_plots(
    *,
    report: dict[str, object],
    dim: int,
    theta_base: float,
    config: f_monster.MonsterConfig,
    candidate_name: str,
    windows: Iterable[int],
    out_dir: Path,
    seed: int = 0,
) -> dict[str, object]:
    windows = tuple(int(w) for w in windows)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_window = _curve_metrics_by_window(report)
    q0, k0 = fixed_query_key(dim, seed=seed)

    dashboards: dict[str, str] = {}
    for length in windows:
        bank = prepare.get_position_bank(length)

        rope_transforms = rope.build_block_transforms(bank=bank, dim=dim, theta_base=theta_base)
        monster_transforms = f_monster.build_block_transforms(bank=bank, dim=dim, config=config)
        monster_axes = f_monster.build_axes(dim // 4, config)

        rope_probe = rope_phase_probe_heatmap_from_transforms(rope_transforms)
        monster_probe = monster_phase_probe_heatmap_from_transforms(
            monster_transforms,
            axes=monster_axes,
        )

        rope_similarity = similarity_matrix_from_transforms(rope_transforms, q0=q0, k0=k0)
        monster_similarity = similarity_matrix_from_transforms(monster_transforms, q0=q0, k0=k0)

        out_path = out_dir / f"window_{length}_dashboard.png"
        save_window_dashboard(
            length=length,
            axial_shape=bank.axial_shape,
            candidate_name=candidate_name,
            rope_probe=rope_probe,
            monster_probe=monster_probe,
            rope_similarity=rope_similarity,
            monster_similarity=monster_similarity,
            reference_curve=np.asarray(report["curves"]["reference"][length], dtype=np.float64),
            candidate_curve=np.asarray(report["curves"]["candidate"][length], dtype=np.float64),
            metrics=metrics_by_window[length],
            out_path=out_path,
        )
        dashboards[str(length)] = str(out_path)

    summary_path = out_dir / "all_windows_kernel_curves.png"
    save_curve_summary(report=report, candidate_name=candidate_name, out_path=summary_path)

    manifest = {
        "candidate_name": candidate_name,
        "dim": int(dim),
        "reference_theta_base": float(theta_base),
        "seed": int(seed),
        "windows": list(windows),
        "candidate_config": asdict(config),
        "dashboards": dashboards,
        "summary_curve_grid": str(summary_path),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_candidate_curve_for_config(config: f_monster.MonsterConfig):
    def build_curve(bank: prepare.PositionBank, dim: int) -> np.ndarray:
        return f_monster.relative_kernel(bank=bank, dim=dim, config=config)
    return build_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate axial RoPE vs F-MonSTER comparison dashboards.")
    parser.add_argument("--windows", type=int, nargs="*", default=None, help="windows to visualize")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory for plots")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the fixed Q/K probe vectors")
    args = parser.parse_args()

    import experiment

    windows = tuple(args.windows) if args.windows else tuple(experiment.WINDOWS)
    out_dir = args.out_dir if args.out_dir is not None else experiment.RUN_ARTIFACTS / "plots"

    report = prepare.evaluate_candidate(
        _build_candidate_curve_for_config(experiment.CANDIDATE),
        dim=experiment.DIM,
        theta_base=experiment.REFERENCE_THETA_BASE,
        windows=windows,
        candidate_name=experiment.CANDIDATE_NAME,
    )
    manifest = save_all_plots(
        report=report,
        dim=experiment.DIM,
        theta_base=experiment.REFERENCE_THETA_BASE,
        config=experiment.CANDIDATE,
        candidate_name=experiment.CANDIDATE_NAME,
        windows=windows,
        out_dir=out_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
