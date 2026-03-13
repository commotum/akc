"""
Program.md v2 benchmark harness.

This module replaces the legacy single-scalar "match RoPE exactly" objective with:
1) pairing-aware evaluation (A..G),
2) fairness tracks (matched-F, equal-D),
3) two score families (RoPE-likeness, MonSTER-ness),
4) structural gates for time-active MonSTER variants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Iterable

import numpy as np

import f_monster
import prepare

BENCHMARK_VERSION = "monster-v2.0"
COORD_INDEX = {"t": 0, "x": 1, "y": 2, "z": 3}
DEFAULT_WINDOWS = (64, 512, 1024, 2048)
DEFAULT_PAIR = "C"
DEFAULT_TRACK = "matched-F"
DEFAULT_F = 96
DEFAULT_D = 384

ETA_GATE = 1e-6
BOOST_COLLAPSE_GATE = 1e-3


@dataclass(frozen=True)
class PairingSpec:
    pair: str
    rope_coords: tuple[str, ...]
    monster_coords: tuple[str, ...]
    time_active: bool
    primary: bool


PAIRINGS: dict[str, PairingSpec] = {
    "A": PairingSpec("A", ("t",), ("t",), True, False),
    "B": PairingSpec("B", ("x",), ("x",), False, False),
    "C": PairingSpec("C", ("x", "y"), ("x", "y"), False, True),
    "D": PairingSpec("D", ("t", "x"), ("t", "x"), True, False),
    "E": PairingSpec("E", ("t", "x", "y"), ("t", "x", "y"), True, False),
    "F": PairingSpec("F", ("x", "y", "z"), ("x", "y", "z"), False, False),
    "G": PairingSpec("G", ("t", "x", "y", "z"), ("t", "x", "y", "z"), True, False),
}


def parse_windows(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return DEFAULT_WINDOWS
    out = tuple(int(v) for v in values)
    if not out:
        raise ValueError("windows must be non-empty")
    return out


def dims_for_track(
    *,
    spec: PairingSpec,
    track: str,
    F: int,
    D: int,
) -> tuple[int, int, int]:
    k = len(spec.rope_coords)
    if track == "matched-F":
        if F <= 0:
            raise ValueError("F must be positive")
        d_rope = 2 * k * int(F)
        d_monster = 4 * int(F)
        return int(F), d_rope, d_monster

    if track == "equal-D":
        d_rope = int(D)
        d_monster = int(D)
        if d_rope <= 0:
            raise ValueError("D must be positive")
        if d_rope % (2 * k) != 0:
            raise ValueError(f"D={d_rope} must be divisible by 2*k={2*k} for pair {spec.pair}")
        if d_monster % 4 != 0:
            raise ValueError(f"D={d_monster} must be divisible by 4 for MonSTER")
        # For reporting, track F on the MonSTER side.
        return d_monster // 4, d_rope, d_monster

    raise ValueError(f"unknown track: {track}")


def canonical_coords(length: int) -> np.ndarray:
    """
    Shared canonical (t, x, y, z) bank.
    t is linear sequence index.
    x, y, z come from near-cubic rasterization used by the fixed harness.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    t = np.arange(length, dtype=np.float64)
    xyz_shape = prepare.near_cubic_shape(length, axes=3)
    xyz = np.stack(np.unravel_index(np.arange(length), xyz_shape), axis=-1).astype(np.float64)
    return np.concatenate((t[:, None], xyz), axis=1)


def mask_coords(coords_txyz: np.ndarray, active: tuple[str, ...]) -> np.ndarray:
    out = np.zeros_like(coords_txyz, dtype=np.float64)
    for c in active:
        out[:, COORD_INDEX[c]] = coords_txyz[:, COORD_INDEX[c]]
    return out


def select_coords(coords_txyz: np.ndarray, active: tuple[str, ...]) -> np.ndarray:
    idx = [COORD_INDEX[c] for c in active]
    return coords_txyz[:, idx]


def build_axial_rope_transforms(
    *,
    coords_active: np.ndarray,
    dim: int,
    theta_base: float,
    freq_scale: float,
    freq_exponent: float,
) -> np.ndarray:
    length = coords_active.shape[0]
    k = coords_active.shape[1]
    if dim % (2 * k) != 0:
        raise ValueError(f"axial RoPE dim must be divisible by 2*k; got dim={dim}, k={k}")
    pair_count = dim // (2 * k)
    inv_freq = prepare.warped_frequencies(
        pair_count,
        theta_base=float(theta_base),
        exponent=float(freq_exponent),
        scale=float(freq_scale),
    )

    phase = coords_active[:, :, None] * inv_freq[None, None, :]
    cos = np.cos(phase)
    sin = np.sin(phase)

    transforms = np.empty((length, k * pair_count, 2, 2), dtype=np.float64)
    block = 0
    for axis_idx in range(k):
        sl = slice(block, block + pair_count)
        transforms[:, sl, 0, 0] = cos[:, axis_idx, :]
        transforms[:, sl, 0, 1] = -sin[:, axis_idx, :]
        transforms[:, sl, 1, 0] = sin[:, axis_idx, :]
        transforms[:, sl, 1, 1] = cos[:, axis_idx, :]
        block += pair_count
    return transforms


def make_monster_bank(coords_txyz: np.ndarray) -> prepare.PositionBank:
    length = int(coords_txyz.shape[0])
    xyz_shape = prepare.near_cubic_shape(length, axes=3)
    sequence_positions = np.arange(length, dtype=np.float64)
    axial_positions = np.stack(np.unravel_index(np.arange(length), xyz_shape), axis=-1).astype(np.float64)
    return prepare.PositionBank(
        length=length,
        axial_shape=tuple(int(x) for x in xyz_shape),
        sequence_positions=sequence_positions,
        axial_positions=axial_positions,
        monster_positions=np.asarray(coords_txyz, dtype=np.float64),
    )


def axis_mode_for_active(monster_coords: tuple[str, ...]) -> str:
    if monster_coords in {("x",), ("t",), ("t", "x")}:
        return "x_only"
    if monster_coords in {("x", "y"), ("t", "x", "y")}:
        return "xy_circle"
    return "fibonacci"


def fixed_monster_config_for_pair(
    *,
    spec: PairingSpec,
    theta_base: float,
    overrides: dict[str, object] | None = None,
) -> f_monster.MonsterConfig:
    cfg = f_monster.MonsterConfig(
        span=2.0 * math.pi,
        top_delta=float(max(DEFAULT_WINDOWS)),
        unit=1.0,
        theta_base=float(theta_base),
        freq_scale=1.0,
        freq_exponent=1.0,
        boost_scale=1.0,
        rotation_scale=1.0,
        axis_mode=axis_mode_for_active(spec.monster_coords),
        axis_blend=1.0,
        block_mode="lorentz",
        dual_freq_mode="interleaved",
    )
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def _pair_similarity(
    transforms: np.ndarray,
    p_idx: np.ndarray,
    q_idx: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    Block-agnostic similarity:
        S(p,q) = mean_blocks [ trace(T_p^T T_q) / block_dim ]
    """
    length, num_blocks, block_dim, _ = transforms.shape
    out = np.empty(p_idx.shape[0], dtype=np.float64)
    norm = float(num_blocks * block_dim)

    for start in range(0, p_idx.size, chunk_size):
        end = min(start + chunk_size, p_idx.size)
        tp = transforms[p_idx[start:end]]
        tq = transforms[q_idx[start:end]]
        # Frobenius inner product equals trace(T_p^T T_q).
        dots = np.einsum("pbmn,pbmn->p", tp, tq)
        out[start:end] = dots / norm
    return out


def _group_indices_from_deltas(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = deltas.shape[1]
    dtype = np.dtype([(f"f{i}", np.int64) for i in range(k)])
    structured = np.ascontiguousarray(deltas, dtype=np.int64).view(dtype).reshape(-1)
    uniq, inv, counts = np.unique(structured, return_inverse=True, return_counts=True)
    keys = np.column_stack([uniq[name] for name in uniq.dtype.names]).astype(np.int64)
    return keys, inv.astype(np.int64), counts.astype(np.int64)


def _group_means(values: np.ndarray, inv: np.ndarray, n_groups: int) -> np.ndarray:
    sums = np.bincount(inv, weights=values, minlength=n_groups)
    counts = np.bincount(inv, minlength=n_groups).astype(np.float64)
    return sums / np.maximum(counts, 1.0)


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    denom = float(np.sum(w))
    if denom <= 0:
        return 0.0
    return float(np.sum(x * w) / denom)


def _radial_profile(
    *,
    keys: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r2 = np.sum(keys.astype(np.float64) ** 2, axis=1)
    r2i = np.round(r2).astype(np.int64)
    uniq, inv = np.unique(r2i, return_inverse=True)
    weighted_sum = np.bincount(inv, weights=means * counts, minlength=uniq.size)
    cnt = np.bincount(inv, weights=counts.astype(np.float64), minlength=uniq.size)
    prof = weighted_sum / np.maximum(cnt, 1.0)
    return uniq.astype(np.float64), prof.astype(np.float64)


def _anisotropy(
    *,
    keys: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
) -> float:
    r2 = np.sum(keys.astype(np.float64) ** 2, axis=1)
    r2i = np.round(r2).astype(np.int64)
    uniq = np.unique(r2i)
    vars_: list[float] = []
    weights: list[float] = []
    for rr in uniq:
        mask = r2i == rr
        if np.sum(mask) < 2:
            continue
        m = means[mask]
        w = counts[mask].astype(np.float64)
        mu = _weighted_mean(m, w)
        var = _weighted_mean((m - mu) ** 2, w)
        vars_.append(var)
        weights.append(float(np.sum(w)))
    if not vars_:
        return 0.0
    return _weighted_mean(np.asarray(vars_, dtype=np.float64), np.asarray(weights, dtype=np.float64))


def _tail_error(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    n = min(profile_a.size, profile_b.size)
    if n <= 1:
        return 0.0
    a = profile_a[:n]
    b = profile_b[:n]
    start = max(1, int(math.floor(0.75 * n)))
    return float(abs(np.max(np.abs(a[start:])) - np.max(np.abs(b[start:]))))


def _spectrum_mismatch(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    n = min(profile_a.size, profile_b.size)
    if n <= 1:
        return 0.0
    a = prepare.normalized_power_spectrum(profile_a[:n], num_bins=32)
    b = prepare.normalized_power_spectrum(profile_b[:n], num_bins=32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _nonseparability(
    *,
    keys: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
) -> float:
    """
    Higher means "more non-separable" under a simple product-of-marginals baseline.
    """
    k = keys.shape[1]
    if k <= 1:
        return 0.0

    eps = 1e-12
    global_mean = _weighted_mean(means, counts.astype(np.float64))
    if abs(global_mean) < eps:
        global_mean = eps

    marginals: list[dict[int, float]] = []
    for i in range(k):
        vals = keys[:, i]
        uniq = np.unique(vals)
        md: dict[int, float] = {}
        for uu in uniq:
            mask = vals == uu
            md[int(uu)] = _weighted_mean(means[mask], counts[mask].astype(np.float64))
        marginals.append(md)

    pred = np.full_like(means, fill_value=1.0, dtype=np.float64)
    for i in range(k):
        pred *= np.asarray([marginals[i][int(v)] for v in keys[:, i]], dtype=np.float64)
    pred /= float(global_mean ** (k - 1))

    return _weighted_mean((means - pred) ** 2, counts.astype(np.float64))


def _cone_separation(
    *,
    keys: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    active_coords: tuple[str, ...],
) -> float:
    if "t" not in active_coords:
        return float("nan")
    time_idx = active_coords.index("t")
    spatial_idx = [i for i, c in enumerate(active_coords) if c != "t"]
    if not spatial_idx:
        return float("nan")

    dt = keys[:, time_idx].astype(np.float64)
    spatial = keys[:, spatial_idx].astype(np.float64)
    interval = -dt * dt + np.sum(spatial * spatial, axis=1)  # (-,+,+,+)
    timelike = interval < 0.0
    spacelike = interval > 0.0
    if not np.any(timelike) or not np.any(spacelike):
        return float("nan")

    w = counts.astype(np.float64)
    mu_t = _weighted_mean(means[timelike], w[timelike])
    mu_s = _weighted_mean(means[spacelike], w[spacelike])
    return float(abs(mu_t - mu_s))


def _minkowski_metrics(monster_transforms: np.ndarray) -> tuple[float, float]:
    eta = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)
    ident = np.eye(4, dtype=np.float64)

    t_eta_t = np.einsum("lbmi,mn,lbnj->lbij", monster_transforms, eta, monster_transforms)
    eta_err = np.linalg.norm(t_eta_t - eta[None, None, :, :], axis=(2, 3))

    gram = np.einsum("lbmi,lbmj->lbij", monster_transforms, monster_transforms)
    boost_err = np.linalg.norm(gram - ident[None, None, :, :], axis=(2, 3))

    return float(np.mean(eta_err)), float(np.mean(boost_err))


def evaluate_pairing(
    *,
    spec: PairingSpec,
    track: str,
    F: int,
    D: int,
    windows: tuple[int, ...],
    theta_base: float,
    rng_seed: int = 0,
    max_pairs_per_window: int = 200_000,
    monster_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    F_used, d_rope, d_monster = dims_for_track(spec=spec, track=track, F=F, D=D)
    monster_cfg = fixed_monster_config_for_pair(
        spec=spec,
        theta_base=theta_base,
        overrides=monster_overrides,
    )
    rng = np.random.default_rng(rng_seed)

    per_window_rows: list[dict[str, float | int | str | bool]] = []
    e_eta_vals: list[float] = []
    n_boost_vals: list[float] = []
    cone_vals: list[float] = []
    nonsep_vals: list[float] = []

    for length in windows:
        coords = canonical_coords(int(length))
        rope_coords = select_coords(coords, spec.rope_coords)
        monster_coords = mask_coords(coords, spec.monster_coords)

        rope_transforms = build_axial_rope_transforms(
            coords_active=rope_coords,
            dim=d_rope,
            theta_base=theta_base,
            freq_scale=1.0,
            freq_exponent=1.0,
        )
        monster_bank = make_monster_bank(monster_coords)
        monster_transforms = f_monster.build_block_transforms(
            bank=monster_bank,
            dim=d_monster,
            config=monster_cfg,
        )

        n_all = length * length
        n_pairs = int(min(max_pairs_per_window, n_all))
        p_idx = rng.integers(0, length, size=n_pairs, endpoint=False, dtype=np.int64)
        q_idx = rng.integers(0, length, size=n_pairs, endpoint=False, dtype=np.int64)

        delta = rope_coords[q_idx] - rope_coords[p_idx]
        delta_i = np.round(delta).astype(np.int64)
        keys, inv, counts = _group_indices_from_deltas(delta_i)
        n_groups = keys.shape[0]

        s_rope = _pair_similarity(rope_transforms, p_idx, q_idx)
        s_monster = _pair_similarity(monster_transforms, p_idx, q_idx)

        rope_means = _group_means(s_rope, inv, n_groups)
        monster_means = _group_means(s_monster, inv, n_groups)
        w = counts.astype(np.float64)

        e_rel = _weighted_mean((s_monster - monster_means[inv]) ** 2, np.ones_like(s_monster))

        # For time-active pairings, E_rope is assessed on the dt=0 spatial slice.
        if spec.time_active and "t" in spec.rope_coords:
            t_idx = spec.rope_coords.index("t")
            slice_mask = keys[:, t_idx] == 0
            if np.any(slice_mask):
                e_rope = _weighted_mean((monster_means[slice_mask] - rope_means[slice_mask]) ** 2, w[slice_mask])
            else:
                e_rope = float("nan")
        else:
            e_rope = _weighted_mean((monster_means - rope_means) ** 2, w)

        r_shells, rope_profile = _radial_profile(keys=keys, means=rope_means, counts=counts)
        _, monster_profile = _radial_profile(keys=keys, means=monster_means, counts=counts)
        spectrum = _spectrum_mismatch(monster_profile, rope_profile)
        tail = _tail_error(monster_profile, rope_profile)
        anisotropy = _anisotropy(keys=keys, means=monster_means, counts=counts)
        nonsep = _nonseparability(keys=keys, means=monster_means, counts=counts)
        cone = _cone_separation(
            keys=keys,
            means=monster_means,
            counts=counts,
            active_coords=spec.monster_coords,
        )
        if spec.time_active:
            e_eta, n_boost = _minkowski_metrics(monster_transforms)
        else:
            e_eta, n_boost = float("nan"), float("nan")

        per_window_rows.append(
            {
                "length": int(length),
                "num_pairs": int(n_pairs),
                "num_delta_groups": int(n_groups),
                "E_rel": float(e_rel),
                "E_rope": float(e_rope),
                "spectrum": float(spectrum),
                "tail": float(tail),
                "anisotropy": float(anisotropy),
                "E_eta": float(e_eta),
                "N_boost": float(n_boost),
                "cone_sep": float(cone),
                "nonseparability": float(nonsep),
            }
        )

        if spec.time_active:
            e_eta_vals.append(float(e_eta))
            n_boost_vals.append(float(n_boost))
            if np.isfinite(cone):
                cone_vals.append(float(cone))
        nonsep_vals.append(float(nonsep))

    def avg(row_key: str) -> float:
        vals = [float(row[row_key]) for row in per_window_rows if np.isfinite(float(row[row_key]))]
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    e_eta = float(np.mean(e_eta_vals)) if e_eta_vals else float("nan")
    n_boost = float(np.mean(n_boost_vals)) if n_boost_vals else float("nan")
    cone_sep = float(np.mean(cone_vals)) if cone_vals else float("nan")
    nonsep = float(np.mean(nonsep_vals)) if nonsep_vals else float("nan")

    collapse_fail = bool(spec.time_active and np.isfinite(e_eta) and np.isfinite(n_boost) and e_eta <= ETA_GATE and n_boost <= BOOST_COLLAPSE_GATE)

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "pair": spec.pair,
        "track": track,
        "active_coords": ",".join(spec.monster_coords),
        "F": int(F_used),
        "D_rope": int(d_rope),
        "D_monster": int(d_monster),
        "E_rel": avg("E_rel"),
        "E_rope": avg("E_rope"),
        "spectrum": avg("spectrum"),
        "tail": avg("tail"),
        "anisotropy": avg("anisotropy"),
        "E_eta": e_eta,
        "N_boost": n_boost,
        "cone_sep": cone_sep,
        "nonseparability": nonsep,
        "collapse_fail": collapse_fail,
        "monster_config": f_monster.config_dict(monster_cfg),
        "windows": per_window_rows,
        "time_active": spec.time_active,
    }


def append_tsv_row(path: Path, row: dict[str, object], *, commit: str, status: str, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "commit\tbenchmark_version\tpair\ttrack\tactive_coords\tF\tD_rope\tD_monster\t"
        "E_rel\tE_rope\tspectrum\ttail\tanisotropy\tE_eta\tN_boost\tcone_sep\t"
        "nonseparability\tcollapse_fail\tstatus\tdescription\n"
    )
    if not path.exists():
        path.write_text(header, encoding="utf-8")

    def fmt(x: object) -> str:
        if isinstance(x, bool):
            return "true" if x else "false"
        if x is None:
            return "NA"
        if isinstance(x, float):
            if not np.isfinite(x):
                return "NA"
            return f"{x:.6f}"
        return str(x)

    fields = [
        commit,
        row["benchmark_version"],
        row["pair"],
        row["track"],
        row["active_coords"],
        row["F"],
        row["D_rope"],
        row["D_monster"],
        row["E_rel"],
        row["E_rope"],
        row["spectrum"],
        row["tail"],
        row["anisotropy"],
        row["E_eta"],
        row["N_boost"],
        row["cone_sep"],
        row["nonseparability"],
        row["collapse_fail"],
        status,
        description,
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\t".join(fmt(x) for x in fields) + "\n")
