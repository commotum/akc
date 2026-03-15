"""
Unified AKC visualizer.

Includes:
- Program.md v2 pair/track charts from benchmark_results.tsv
- compare.ipynb-style diagnostics (probe heatmaps, local-basis panels,
  synthetic similarity heatmaps, and offset curves)
- monster_hyperparam_grid.py-style grid sweep (ported to f_monster)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "run_artifacts" else SCRIPT_DIR
RUN_ARTIFACTS = PROJECT_ROOT / "run_artifacts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import benchmark_v2
import f_monster
import prepare


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
ETA4 = np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64)


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


@dataclass(frozen=True)
class HyperGridConfig:
    image_size: int = 16
    embed_dim: int = 768
    theta_base: float = 10_000.0
    top_delta: float = 16.0
    seed: int = 0
    query_x: int = 8
    query_y: int = 8
    query_on_value: float = 100.0
    key_t_value: float = 0.0
    query_t_values: tuple[float, ...] = (-8.0, -16.0, -24.0)
    tanh_k_values: tuple[float, ...] = (0.0001, 0.001, 0.01)
    unit_scale: float = 1.0
    axis_mode: str = "xy_circle"
    freq_scale: float = 1.0
    freq_exponent: float = 1.0
    boost_scale: float = 1.0
    rotation_scale: float = 1.0


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
        vals.append(np.nan if row is None else float(getattr(row, metric)))
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


def run_pair_track_reports(
    *,
    results_tsv: Path,
    benchmark_version: str | None,
    statuses: tuple[str, ...],
    out_dir: Path,
) -> dict[str, str]:
    rows = read_rows(results_tsv)
    if not rows:
        raise ValueError(f"no rows found in {results_tsv}")

    version = benchmark_version or latest_benchmark_version(rows)
    selected = select_rows(rows, benchmark_version=version, statuses=set(statuses))
    if not selected:
        raise ValueError(f"no rows matched benchmark version {version}")
    best = choose_best_by_pair_track(selected)

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
        "results_tsv": str(results_tsv),
        "statuses_used": list(statuses),
        "out_dir": str(out_dir),
        "score_plot": str(score_plot),
        "rope_metrics_plot": str(rope_plot),
        "time_metrics_plot": str(time_plot),
        "summary_markdown": str(summary_md),
        "best_rows": manifest_rows,
    }
    manifest_path = out_dir / f"manifest_{version}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "score_plot": str(score_plot),
        "rope_metrics": str(rope_plot),
        "time_metrics": str(time_plot),
        "summary": str(summary_md),
    }


def _apply_transforms_2d_batch(vec: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    blocks = vec.reshape(-1, 2)
    out = np.einsum("lbxy,by->lbx", transforms, blocks)
    return out.reshape(transforms.shape[0], -1)


def _apply_transforms_4d_batch(vec: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    blocks = vec.reshape(-1, 4)
    out = np.einsum("lfxy,fy->lfx", transforms, blocks)
    return out.reshape(transforms.shape[0], -1)


def _minkowski_similarity_matrix(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    q4 = q.reshape(q.shape[0], -1, 4)
    k4 = k.reshape(k.shape[0], -1, 4)
    return np.einsum("pfd,d,qfd->pq", q4, ETA4, k4)


def _minkowski_dot_batch(query_vec: np.ndarray, keys: np.ndarray) -> np.ndarray:
    q4 = query_vec.reshape(-1, 4)
    k4 = keys.reshape(keys.shape[0], -1, 4)
    return np.einsum("fd,d,pfd->p", q4, ETA4, k4)


def _offset_curve(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seq_len = sim.shape[0]
    offsets = np.arange(-(seq_len - 1), seq_len)
    means = np.array([np.mean(np.diag(sim, k=o)) for o in offsets], dtype=np.float64)
    return offsets, means


def _build_monster_config(
    *,
    spec: benchmark_v2.PairingSpec,
    theta_base: float,
    freq_scale: float,
    freq_exponent: float,
    t_unit: float,
    s_unit: float,
    boost_scale: float,
    rotation_scale: float,
    axis_mode: str | None,
    axis_blend: float,
    block_mode: str,
) -> f_monster.MonsterConfig:
    cfg = benchmark_v2.fixed_monster_config_for_pair(spec=spec, theta_base=theta_base)
    kwargs: dict[str, object] = {
        "freq_scale": float(freq_scale),
        "freq_exponent": float(freq_exponent),
        "t_unit": float(t_unit),
        "s_unit": float(s_unit),
        "boost_scale": float(boost_scale),
        "rotation_scale": float(rotation_scale),
        "axis_blend": float(axis_blend),
        "block_mode": str(block_mode),
    }
    if axis_mode is not None:
        kwargs["axis_mode"] = str(axis_mode)
    return f_monster.MonsterConfig(**(f_monster.config_dict(cfg) | kwargs))


def _demo_positions_for_pair(seq_len: int, active: tuple[str, ...]) -> np.ndarray:
    s = np.arange(seq_len, dtype=np.float64)
    coords = np.zeros((seq_len, 4), dtype=np.float64)
    for c in active:
        coords[:, benchmark_v2.COORD_INDEX[c]] = s
    return coords


def _build_local_frame(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.empty_like(axes)
    for k in range(axes.shape[0]):
        a = axes[k]
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = ref - float(np.dot(ref, a)) * a
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = ref - float(np.dot(ref, a)) * a
            n = float(np.linalg.norm(u))
        if n < 1e-12:
            raise ValueError("Failed to build stable local frame.")
        b[k] = u / n
    c = np.cross(axes, b)
    return b, c


def _project_local_basis(encoded: np.ndarray, axes: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    seq_len = encoded.shape[0]
    num_freq = axes.shape[0]
    blocks = encoded.reshape(seq_len, num_freq, 4)
    spatial = blocks[..., 1:]

    local = np.empty_like(blocks)
    local[..., 0] = blocks[..., 0]
    local[..., 1] = np.einsum("pfi,fi->pf", spatial, axes)
    local[..., 2] = np.einsum("pfi,fi->pf", spatial, b)
    local[..., 3] = np.einsum("pfi,fi->pf", spatial, c)
    return local


def run_compare_style_diagnostics(
    *,
    spec: benchmark_v2.PairingSpec,
    seq_len: int,
    demo_freq: int,
    theta_base: float,
    monster_cfg: f_monster.MonsterConfig,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    _, d_rope, d_monster = benchmark_v2.dims_for_track(spec=spec, track="matched-F", F=demo_freq, D=0)
    coords_txyz = _demo_positions_for_pair(seq_len, spec.monster_coords)

    rope_positions = benchmark_v2.select_coords(coords_txyz, spec.rope_coords)
    rope_tf = benchmark_v2.build_axial_rope_transforms(
        coords_active=rope_positions,
        dim=d_rope,
        theta_base=theta_base,
        freq_scale=1.0,
        freq_exponent=1.0,
    )

    masked_monster = benchmark_v2.mask_coords(coords_txyz, spec.monster_coords)
    monster_bank = benchmark_v2.make_monster_bank(masked_monster)
    monster_tf = f_monster.build_block_transforms(bank=monster_bank, dim=d_monster, config=monster_cfg)

    rope_probe = np.zeros(d_rope, dtype=np.float64)
    rope_probe[0::2] = 1.0
    monster_probe = np.zeros(d_monster, dtype=np.float64)
    monster_probe[0::4] = 1.0

    rope_probe_mat = _apply_transforms_2d_batch(rope_probe, rope_tf)
    monster_probe_mat = _apply_transforms_4d_batch(monster_probe, monster_tf)

    norm_plot = out_dir / f"compare_mean_norm_{spec.pair}.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    means = [
        float(np.mean(np.linalg.norm(rope_probe_mat, axis=1))),
        float(np.mean(np.linalg.norm(monster_probe_mat, axis=1))),
    ]
    labels = ["axial-rope", "monster"]
    ax.bar(labels, means, color=["#1f77b4", "#d62728"], alpha=0.9)
    ax.set_ylabel("mean output norm")
    ax.set_title(f"Mean encoded norm by method (pair {spec.pair}, seq={seq_len})")
    fig.savefig(norm_plot, dpi=160)
    plt.close(fig)

    probe_plot = out_dir / f"compare_probe_heatmaps_{spec.pair}.png"
    mats = {"axial-rope": rope_probe_mat, "monster": monster_probe_mat}
    vmin = min(float(np.min(m)) for m in mats.values())
    vmax = max(float(np.max(m)) for m in mats.values())
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, (name, mat) in zip(axes, mats.items()):
        im = ax.pcolormesh(mat, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}: probe heatmap")
        ax.set_xlabel("embedding dimension")
        ax.set_ylabel("position")
        ax.set_xlim((0, mat.shape[1]))
        ax.set_ylim((seq_len, 0))
    cbar = fig.colorbar(im, ax=axes, pad=0.01)
    cbar.set_label("encoding value")
    fig.suptitle(f"Probe positional heatmaps (pair {spec.pair})")
    fig.savefig(probe_plot, dpi=160)
    plt.close(fig)

    sim_plot = out_dir / f"compare_similarity_heatmaps_{spec.pair}.png"
    rng = np.random.default_rng(0)
    q0_rope = rng.normal(size=(d_rope,)).astype(np.float64)
    k0_rope = rng.normal(size=(d_rope,)).astype(np.float64)
    q0_monster = rng.normal(size=(d_monster,)).astype(np.float64)
    k0_monster = rng.normal(size=(d_monster,)).astype(np.float64)

    rope_q = _apply_transforms_2d_batch(q0_rope, rope_tf)
    rope_k = _apply_transforms_2d_batch(k0_rope, rope_tf)
    monster_q = _apply_transforms_4d_batch(q0_monster, monster_tf)
    monster_k = _apply_transforms_4d_batch(k0_monster, monster_tf)

    sims = {
        "axial-rope (euclidean)": rope_q @ rope_k.T,
        "monster (minkowski)": _minkowski_similarity_matrix(monster_q, monster_k),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, (name, sim) in zip(axes, sims.items()):
        local_abs_max = float(np.max(np.abs(sim)))
        if local_abs_max < 1e-12:
            local_abs_max = 1e-12
        im = ax.imshow(sim, cmap="coolwarm", aspect="auto", vmin=-local_abs_max, vmax=local_abs_max)
        ax.set_title(name)
        ax.set_xlabel("q position")
        ax.set_ylabel("p position")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("attention similarity")
    fig.suptitle(f"Synthetic q/k similarity (pair {spec.pair}, seq={seq_len})")
    fig.savefig(sim_plot, dpi=160)
    plt.close(fig)

    offset_plot = out_dir / f"compare_offset_curves_{spec.pair}.png"
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for name, sim in sims.items():
        offs, vals = _offset_curve(sim)
        ax.plot(offs, vals, linewidth=2, label=name)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.35)
    ax.set_title(f"Attention score vs positional offset (pair {spec.pair})")
    ax.set_xlabel("positional offset (p - q)")
    ax.set_ylabel("mean similarity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(offset_plot, dpi=160)
    plt.close(fig)

    axes_mon = f_monster.build_axes(d_monster // 4, monster_cfg)
    b_vec, c_vec = _build_local_frame(axes_mon)

    s = np.arange(seq_len, dtype=np.float64)
    spatial_dir = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    spatial_dir /= np.linalg.norm(spatial_dir)
    spatial_ramp = s[:, None] * spatial_dir[None, :]

    scenario_coords = {
        "boost_only_[1,0,0,0]": np.column_stack((s, np.zeros((seq_len, 3), dtype=np.float64))),
        "rotation_only_[0,b_f]": np.column_stack((np.zeros(seq_len, dtype=np.float64), spatial_ramp)),
        "combined_[1,b_f]": np.column_stack((s, spatial_ramp)),
    }

    diagnostics_out: dict[str, str] = {}
    for label, coords4 in scenario_coords.items():
        bank = benchmark_v2.make_monster_bank(coords4)
        tf = f_monster.build_block_transforms(bank=bank, dim=d_monster, config=monster_cfg)

        probe = np.zeros(d_monster, dtype=np.float64)
        blocks = probe.reshape(-1, 4)
        if label.startswith("boost_only"):
            blocks[:, 0] = 1.0
        elif label.startswith("rotation_only"):
            blocks[:, 1:] = b_vec
        else:
            blocks[:, 0] = 1.0
            blocks[:, 1:] = b_vec

        enc = _apply_transforms_4d_batch(probe, tf)
        local = _project_local_basis(enc, axes_mon, b_vec, c_vec)
        channels = [local[:, :, i] for i in range(4)]

        vmin = min(float(np.min(ch)) for ch in channels)
        vmax = max(float(np.max(ch)) for ch in channels)
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

        out_path = out_dir / f"compare_monster_local_basis_{spec.pair}_{label}.png"
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
        names = ["t_local", "axis_local", "b_local", "c_local"]
        for idx, ax in enumerate(axes):
            im = ax.pcolormesh(channels[idx], cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax)
            ax.set_title(names[idx])
            ax.set_xlabel("frequency block")
            ax.set_ylabel("position")
            ax.set_xlim((0, channels[idx].shape[1]))
            ax.set_ylim((seq_len, 0))
        cbar = fig.colorbar(im, ax=axes, pad=0.01)
        cbar.set_label("local value")
        fig.suptitle(f"MonSTER local basis: {label}")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        diagnostics_out[label] = str(out_path)

    return {
        "mean_norm": str(norm_plot),
        "probe_heatmaps": str(probe_plot),
        "similarity_heatmaps": str(sim_plot),
        "offset_curves": str(offset_plot),
        **{f"local_basis_{k}": v for k, v in diagnostics_out.items()},
    }


def _random_embedding(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, 1.0, d)
    v = v / np.linalg.norm(v) * np.sqrt(d)
    return v


def _make_query_image(size: int, query_x: int, query_y: int, on_value: float) -> np.ndarray:
    image = np.zeros((size, size), dtype=float)
    image[query_y, query_x] = on_value
    return image


def _centered_xy_coords(size: int) -> tuple[np.ndarray, np.ndarray]:
    vals = np.arange(-(size / 2) + 0.5, size / 2, 1.0)
    return np.meshgrid(vals, vals, indexing="xy")


def _tanh_time(t_value: float, k_value: float) -> float:
    return float(np.tanh(k_value * t_value))


def _make_standard_positions(size: int, t_value: float, k_value: float) -> np.ndarray:
    x, y = _centered_xy_coords(size)
    t = np.full_like(x, fill_value=_tanh_time(t_value, k_value), dtype=np.float64)
    z = np.zeros_like(x, dtype=np.float64)
    return np.stack((t, x, y, z), axis=-1).reshape(-1, 4)


def _encode_monster_vectors(base_vector: np.ndarray, positions: np.ndarray, cfg: f_monster.MonsterConfig) -> np.ndarray:
    bank = benchmark_v2.make_monster_bank(positions)
    tf = f_monster.build_block_transforms(bank=bank, dim=base_vector.size, config=cfg)
    return _apply_transforms_4d_batch(base_vector, tf)


def run_monster_hyper_grid(
    *,
    cfg: HyperGridConfig,
    out_dir: Path,
) -> dict[str, str]:
    if cfg.embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4")
    if not (0 <= cfg.query_x < cfg.image_size and 0 <= cfg.query_y < cfg.image_size):
        raise ValueError("query position must be inside the grid")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "monster_hyperparam_grid.png"

    rng = np.random.default_rng(cfg.seed)
    base_vector = _random_embedding(cfg.embed_dim, rng)
    query_index = cfg.query_y * cfg.image_size + cfg.query_x

    monster_cfg = f_monster.MonsterConfig(
        t_unit=cfg.unit_scale / cfg.top_delta,
        s_unit=cfg.unit_scale / cfg.top_delta,
        theta_base=cfg.theta_base,
        freq_scale=cfg.freq_scale,
        freq_exponent=cfg.freq_exponent,
        boost_scale=cfg.boost_scale,
        rotation_scale=cfg.rotation_scale,
        axis_mode=cfg.axis_mode,
        axis_blend=1.0,
        block_mode="lorentz",
        dual_freq_mode="interleaved",
    )

    row_maps: list[list[np.ndarray]] = []
    row_titles: list[list[str]] = []
    for k_value in cfg.tanh_k_values:
        maps_for_k: list[np.ndarray] = []
        titles_for_k: list[str] = []

        key_positions = _make_standard_positions(cfg.image_size, cfg.key_t_value, k_value)
        key_encoded = _encode_monster_vectors(base_vector, key_positions, monster_cfg)
        query_base = key_positions[query_index].copy()

        for query_t in cfg.query_t_values:
            query_pos = query_base.copy()
            query_pos[0] = _tanh_time(query_t, k_value)
            query_encoded = _encode_monster_vectors(base_vector, query_pos[None, :], monster_cfg)[0]
            logits = _minkowski_dot_batch(query_encoded, key_encoded) / math.sqrt(float(cfg.embed_dim))
            maps_for_k.append(logits.reshape(cfg.image_size, cfg.image_size))
            titles_for_k.append(f"q t={query_t:g} -> tanh={_tanh_time(query_t, k_value):.4f}")

        row_maps.append(maps_for_k)
        row_titles.append(titles_for_k)

    query_image = _make_query_image(cfg.image_size, cfg.query_x, cfg.query_y, cfg.query_on_value)
    all_panels = [panel for row in row_maps for panel in row]
    vmin = min(float(panel.min()) for panel in all_panels)
    vmax = max(float(panel.max()) for panel in all_panels)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    n_rows = len(cfg.tanh_k_values)
    fig = plt.figure(figsize=(13.7, 4.8 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        5,
        left=0.03,
        right=0.97,
        top=0.93,
        bottom=0.06,
        hspace=0.28,
        wspace=0.08,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.05],
    )

    cbar_ax = fig.add_subplot(gs[:, 4])
    right_last_image = None

    for row_idx, k_value in enumerate(cfg.tanh_k_values):
        ax_bin = fig.add_subplot(gs[row_idx, 0])
        ax_bin.imshow(query_image, cmap="viridis", vmin=0.0, vmax=cfg.query_on_value, interpolation="nearest")
        if row_idx == 0:
            ax_bin.set_title("Binary input", fontsize=11)
        ax_bin.axis("off")
        ax_bin.text(
            -0.12,
            0.5,
            f"k={k_value:g}",
            transform=ax_bin.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
        )

        for col in range(3):
            ax = fig.add_subplot(gs[row_idx, col + 1])
            panel = row_maps[row_idx][col]
            right_last_image = ax.imshow(panel, cmap="viridis", vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(row_titles[row_idx][col], fontsize=11)
            ax.axis("off")

    fig.colorbar(right_last_image, cax=cbar_ax)
    fig.suptitle(
        (
            f"MonSTER hyperparam grid | dim={cfg.embed_dim} | base={cfg.theta_base:g} | "
            f"top_delta={cfg.top_delta:g} | query=({cfg.query_x}, {cfg.query_y}) | rows=tanh(k*t)"
        ),
        fontsize=15,
    )
    fig.text(
        0.50,
        0.95,
        "Centered spatial coordinates, metric (-,+,+,+), columns are query t values",
        ha="center",
        va="center",
        fontsize=12,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"monster_hyperparam_grid": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AKC unified visualizer.")

    parser.add_argument(
        "--results-tsv",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results.tsv",
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
        help="statuses eligible for selecting best rows",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=RUN_ARTIFACTS / "plots_v2",
        help="directory for pair-track charts",
    )

    parser.add_argument("--pair", type=str, default="C", choices=sorted(benchmark_v2.PAIRINGS.keys()))
    parser.add_argument("--demo-seq-len", type=int, default=64)
    parser.add_argument("--demo-freq", type=int, default=24)
    parser.add_argument("--theta-base", type=float, default=float(prepare.REFERENCE_THETA_BASE))
    parser.add_argument("--freq-scale", type=float, default=1.0)
    parser.add_argument("--freq-exponent", type=float, default=1.0)
    parser.add_argument("--t-unit", type=float, default=1.0)
    parser.add_argument("--s-unit", type=float, default=1.0)
    parser.add_argument("--boost-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--axis-mode", type=str, default=None)
    parser.add_argument("--axis-blend", type=float, default=1.0)
    parser.add_argument("--block-mode", type=str, default="lorentz")
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=RUN_ARTIFACTS / "diagnostics",
        help="directory for compare-style outputs and hyper-grid chart",
    )

    parser.add_argument("--hyper-grid-dim", type=int, default=768)
    parser.add_argument("--hyper-grid-size", type=int, default=16)
    parser.add_argument("--hyper-grid-top-delta", type=float, default=16.0)

    parser.add_argument("--skip-pair-charts", action="store_true")
    parser.add_argument("--skip-compare-charts", action="store_true")
    parser.add_argument("--skip-hyper-grid", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = benchmark_v2.PAIRINGS[args.pair]

    monster_cfg = _build_monster_config(
        spec=spec,
        theta_base=float(args.theta_base),
        freq_scale=float(args.freq_scale),
        freq_exponent=float(args.freq_exponent),
        t_unit=float(args.t_unit),
        s_unit=float(args.s_unit),
        boost_scale=float(args.boost_scale),
        rotation_scale=float(args.rotation_scale),
        axis_mode=args.axis_mode,
        axis_blend=float(args.axis_blend),
        block_mode=str(args.block_mode),
    )

    results: dict[str, object] = {
        "project_root": str(PROJECT_ROOT),
        "pair": args.pair,
        "demo_seq_len": int(args.demo_seq_len),
        "demo_freq": int(args.demo_freq),
        "monster_config": f_monster.config_dict(monster_cfg),
    }

    if not args.skip_pair_charts:
        results["pair_track_reports"] = run_pair_track_reports(
            results_tsv=args.results_tsv,
            benchmark_version=args.benchmark_version,
            statuses=tuple(args.statuses),
            out_dir=args.plots_dir,
        )

    if not args.skip_compare_charts:
        results["compare_style"] = run_compare_style_diagnostics(
            spec=spec,
            seq_len=int(args.demo_seq_len),
            demo_freq=int(args.demo_freq),
            theta_base=float(args.theta_base),
            monster_cfg=monster_cfg,
            out_dir=args.diagnostics_dir,
        )

    if not args.skip_hyper_grid:
        hg_cfg = HyperGridConfig(
            image_size=int(args.hyper_grid_size),
            embed_dim=int(args.hyper_grid_dim),
            theta_base=float(args.theta_base),
            top_delta=float(args.hyper_grid_top_delta),
            freq_scale=float(args.freq_scale),
            freq_exponent=float(args.freq_exponent),
            boost_scale=float(args.boost_scale),
            rotation_scale=float(args.rotation_scale),
            axis_mode=args.axis_mode or "xy_circle",
        )
        results["monster_hyper_grid"] = run_monster_hyper_grid(cfg=hg_cfg, out_dir=args.diagnostics_dir)

    manifest_path = args.diagnostics_dir / "visualize_manifest.json"
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    results["manifest"] = str(manifest_path)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
