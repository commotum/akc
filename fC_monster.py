#!/usr/bin/env python3
"""
Standalone Pair-C (x,y) MonSTER benchmark.

This file is intentionally self-contained:
- no imports from prepare.py / benchmark_v2.py / f_monster.py
- all defaults and hyperparameters are declared here

Pair C definition:
- RoPE coords: (x, y)
- MonSTER coords: (x, y) embedded in 4D as (t=0, x, y, z=0)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Pair-C benchmark defaults (editable in one place)
# ---------------------------------------------------------------------------

BENCHMARK_VERSION = "fC-standalone-v1"
PAIR_NAME = "C"
ROPE_COORDS = ("x", "y")
MONSTER_COORDS = ("x", "y")
COORD_INDEX = {"t": 0, "x": 1, "y": 2, "z": 3}

DEFAULT_WINDOWS = (64, 512, 1024, 2048)
DEFAULT_TRACK = "matched-F"
DEFAULT_F = 96
DEFAULT_D = 384
DEFAULT_THETA_BASE = 10_000.0
DEFAULT_MAX_PAIRS_PER_WINDOW = 200_000
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# MonSTER hyperparameters (all in this file)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonsterConfig:
    # Coordinate units
    t_unit: float = 1.0
    s_unit: float = 1.0

    # Frequency family
    theta_base: float = DEFAULT_THETA_BASE
    freq_scale: float = 1.0
    freq_exponent: float = 1.0

    # Relative weights
    boost_scale: float = 1.0
    rotation_scale: float = 1.0

    # Axis distribution
    axis_mode: str = "xy_circle"  # fibonacci | cycle | hybrid | x_only | xy_circle
    axis_blend: float = 1.0

    # Block transform family
    block_mode: str = "lorentz"  # lorentz | dual_plane
    dual_freq_mode: str = "interleaved"  # interleaved | axis_grouped


def config_dict(config: MonsterConfig) -> dict[str, float | str | None]:
    return asdict(config)


def resolved_units(config: MonsterConfig) -> tuple[float, float]:
    t_unit = float(config.t_unit)
    s_unit = float(config.s_unit)
    if (not np.isfinite(t_unit)) or (not np.isfinite(s_unit)):
        raise ValueError("t_unit and s_unit must be finite")
    return t_unit, s_unit


# ---------------------------------------------------------------------------
# Geometry / frequency helpers
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


def near_cubic_shape(length: int, axes: int = 3) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("length must be positive")
    dims = [1] * axes
    for factor in sorted(prime_factors(length), reverse=True):
        idx = min(range(axes), key=lambda i: (dims[i], i))
        dims[idx] *= factor
    return tuple(sorted(dims))


def canonical_coords(length: int) -> np.ndarray:
    """
    Shared (t, x, y, z) coordinates.
    t is sequence index. x,y,z come from near-cubic rasterization.
    """
    t = np.arange(length, dtype=np.float64)
    xyz_shape = near_cubic_shape(length, axes=3)
    xyz = np.stack(np.unravel_index(np.arange(length), xyz_shape), axis=-1).astype(np.float64)
    return np.concatenate((t[:, None], xyz), axis=1)


def warped_frequencies(
    num_freq: int,
    *,
    theta_base: float,
    exponent: float = 1.0,
    scale: float = 1.0,
) -> np.ndarray:
    if num_freq <= 0:
        raise ValueError("num_freq must be positive")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    ranks = np.arange(num_freq, dtype=np.float64) / float(num_freq)
    warped = ranks ** float(exponent)
    return float(scale) * (float(theta_base) ** (-warped))


def select_pair_c_coords(coords_txyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pair C:
    - RoPE sees (x, y)
    - MonSTER sees 4D (t, x, y, z) with only x,y active
    """
    rope_xy = coords_txyz[:, [COORD_INDEX["x"], COORD_INDEX["y"]]]
    monster_txyz = np.zeros((coords_txyz.shape[0], 4), dtype=np.float64)
    monster_txyz[:, COORD_INDEX["x"]] = coords_txyz[:, COORD_INDEX["x"]]
    monster_txyz[:, COORD_INDEX["y"]] = coords_txyz[:, COORD_INDEX["y"]]
    return rope_xy, monster_txyz


# ---------------------------------------------------------------------------
# Axial RoPE (self-contained)
# ---------------------------------------------------------------------------


def build_axial_rope_transforms(
    *,
    coords_active: np.ndarray,  # (L,2) for pair C
    dim: int,
    theta_base: float,
    freq_scale: float = 1.0,
    freq_exponent: float = 1.0,
) -> np.ndarray:
    length = coords_active.shape[0]
    k = coords_active.shape[1]
    if dim % (2 * k) != 0:
        raise ValueError(f"axial RoPE dim must be divisible by 2*k; got dim={dim}, k={k}")
    pair_count = dim // (2 * k)
    inv_freq = warped_frequencies(
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


# ---------------------------------------------------------------------------
# MonSTER block transforms (self-contained)
# ---------------------------------------------------------------------------


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)
    return x / norms


def fibonacci_sphere(num_points: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    i = np.arange(num_points, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))

    z = 1.0 - 2.0 * (i + 0.5) / num_points
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = i * phi

    out = np.empty((num_points, 3), dtype=np.float64)
    out[:, 0] = np.cos(theta) * r
    out[:, 1] = np.sin(theta) * r
    out[:, 2] = z
    return _normalize_rows(out)


def cycle_axes(num_points: int) -> np.ndarray:
    base = np.eye(3, dtype=np.float64)
    reps = (num_points + 2) // 3
    return np.tile(base, (reps, 1))[:num_points]


def build_axes(num_freq: int, config: MonsterConfig) -> np.ndarray:
    mode = config.axis_mode.lower()
    fib = fibonacci_sphere(num_freq)
    cyc = cycle_axes(num_freq)

    if mode in {"x_only", "x"}:
        axes = np.zeros((num_freq, 3), dtype=np.float64)
        axes[:, 0] = 1.0
    elif mode in {"xy_circle", "xy"}:
        i = np.arange(num_freq, dtype=np.float64)
        theta = (2.0 * math.pi) * (i + 0.5) / max(1, num_freq)
        axes = np.zeros((num_freq, 3), dtype=np.float64)
        axes[:, 0] = np.cos(theta)
        axes[:, 1] = np.sin(theta)
    elif mode == "fibonacci":
        axes = fib
    elif mode in {"cycle", "cycle_xyz"}:
        axes = cyc
    elif mode == "hybrid":
        blend = float(np.clip(config.axis_blend, 0.0, 1.0))
        axes = _normalize_rows((1.0 - blend) * cyc + blend * fib)
    else:
        raise ValueError(f"unknown axis_mode: {config.axis_mode}")
    return _normalize_rows(axes)


def skew_matrices(axes: np.ndarray) -> np.ndarray:
    mats = np.zeros((axes.shape[0], 3, 3), dtype=np.float64)
    ax = axes[:, 0]
    ay = axes[:, 1]
    az = axes[:, 2]

    mats[:, 0, 1] = -az
    mats[:, 0, 2] = ay
    mats[:, 1, 0] = az
    mats[:, 1, 2] = -ax
    mats[:, 2, 0] = -ay
    mats[:, 2, 1] = ax
    return mats


def tangent_frames(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = _normalize_rows(np.asarray(axes, dtype=np.float64))
    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (axes.shape[0], 1))
    near_parallel = np.abs(np.sum(axes * ref, axis=-1)) > 0.90
    ref[near_parallel] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    u = np.cross(ref, axes)
    u = _normalize_rows(u)
    v = np.cross(axes, u)
    v = _normalize_rows(v)
    return u, v


def _build_lorentz_block_transforms(
    *,
    positions_4d: np.ndarray,  # (L,4)
    dim: int,
    config: MonsterConfig,
) -> np.ndarray:
    length = positions_4d.shape[0]
    num_freq = dim // 4
    t_unit, s_unit = resolved_units(config)
    inv_freq = warped_frequencies(
        num_freq,
        theta_base=float(config.theta_base),
        exponent=float(config.freq_exponent),
        scale=float(config.freq_scale),
    )
    axes = build_axes(num_freq, config)

    t = positions_4d[:, 0:1]         # (L,1)
    spatial = positions_4d[:, 1:4]   # (L,3)
    proj = spatial @ axes.T          # (L,F)

    phi = t * (t_unit * float(config.boost_scale)) * inv_freq[None, :]
    theta = proj * (s_unit * float(config.rotation_scale)) * inv_freq[None, :]

    ch = np.cosh(phi)
    sh = np.sinh(phi)
    c = np.cos(theta)
    s = np.sin(theta)

    outer = axes[:, :, None] * axes[:, None, :]   # (F,3,3)
    skew = skew_matrices(axes)                    # (F,3,3)
    i3 = np.eye(3, dtype=np.float64)[None, :, :]

    boost = np.zeros((length, num_freq, 4, 4), dtype=np.float64)
    boost[:, :, 0, 0] = ch
    boost[:, :, 0, 1:] = -sh[:, :, None] * axes[None, :, :]
    boost[:, :, 1:, 0] = -sh[:, :, None] * axes[None, :, :]
    boost[:, :, 1:, 1:] = i3[None, :, :, :] + (ch - 1.0)[:, :, None, None] * outer[None, :, :, :]

    rotate = np.zeros((length, num_freq, 4, 4), dtype=np.float64)
    rotate[:, :, 0, 0] = 1.0
    rotate[:, :, 1:, 1:] = (
        c[:, :, None, None] * i3[None, :, :, :]
        + s[:, :, None, None] * skew[None, :, :, :]
        + (1.0 - c)[:, :, None, None] * outer[None, :, :, :]
    )

    # Boost first, then rotation around same axis.
    return rotate @ boost


def _build_dual_plane_block_transforms(
    *,
    positions_4d: np.ndarray,
    dim: int,
    config: MonsterConfig,
) -> np.ndarray:
    length = positions_4d.shape[0]
    num_freq = dim // 4
    t_unit, s_unit = resolved_units(config)
    axes = build_axes(num_freq, config)
    u, v = tangent_frames(axes)

    spatial = positions_4d[:, 1:4]
    proj = spatial @ axes.T

    dual_freq_mode = config.dual_freq_mode.lower()
    if dual_freq_mode == "interleaved":
        inv_freq = warped_frequencies(
            2 * num_freq,
            theta_base=float(config.theta_base),
            exponent=float(config.freq_exponent),
            scale=float(config.freq_scale),
        )
        inv_phi = inv_freq[0::2]
        inv_theta = inv_freq[1::2]
    elif dual_freq_mode in {"axis_grouped", "grouped"}:
        axis_count = 3
        if num_freq % axis_count != 0:
            raise ValueError(f"axis_grouped requires num_freq divisible by {axis_count}; got {num_freq}")
        if config.axis_mode.lower() not in {"cycle", "cycle_xyz"}:
            raise ValueError("axis_grouped requires axis_mode='cycle'")

        blocks_per_axis = num_freq // axis_count
        freq_per_axis = 2 * blocks_per_axis
        axis_inv = warped_frequencies(
            freq_per_axis,
            theta_base=float(config.theta_base),
            exponent=float(config.freq_exponent),
            scale=float(config.freq_scale),
        )
        block_rank = np.arange(num_freq, dtype=np.int64) // axis_count
        inv_phi = axis_inv[2 * block_rank]
        inv_theta = axis_inv[2 * block_rank + 1]
    else:
        raise ValueError(f"unknown dual_freq_mode: {config.dual_freq_mode}")

    phi = proj * (t_unit * float(config.boost_scale)) * inv_phi[None, :]
    theta = proj * (s_unit * float(config.rotation_scale)) * inv_theta[None, :]

    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)

    blocks = np.zeros((length, num_freq, 4, 4), dtype=np.float64)
    blocks[:, :, 0, 0] = cp
    blocks[:, :, 0, 1] = -sp
    blocks[:, :, 1, 0] = sp
    blocks[:, :, 1, 1] = cp
    blocks[:, :, 2, 2] = ct
    blocks[:, :, 2, 3] = -st
    blocks[:, :, 3, 2] = st
    blocks[:, :, 3, 3] = ct

    e_time = np.zeros((num_freq, 4), dtype=np.float64)
    e_time[:, 0] = 1.0
    e_axis = np.zeros((num_freq, 4), dtype=np.float64)
    e_axis[:, 1:] = axes
    e_u = np.zeros((num_freq, 4), dtype=np.float64)
    e_u[:, 1:] = u
    e_v = np.zeros((num_freq, 4), dtype=np.float64)
    e_v[:, 1:] = v
    basis = np.stack((e_time, e_axis, e_u, e_v), axis=-1)

    transformed = np.einsum("fij,lfjk->lfik", basis, blocks)
    return np.einsum("lfij,fkj->lfik", transformed, basis)


def build_monster_block_transforms(
    *,
    positions_4d: np.ndarray,
    dim: int,
    config: MonsterConfig,
) -> np.ndarray:
    if dim % 4 != 0:
        raise ValueError(f"MonSTER requires dim divisible by 4; got dim={dim}")
    mode = config.block_mode.lower()
    if mode == "lorentz":
        return _build_lorentz_block_transforms(positions_4d=positions_4d, dim=dim, config=config)
    if mode in {"dual_plane", "dual"}:
        return _build_dual_plane_block_transforms(positions_4d=positions_4d, dim=dim, config=config)
    raise ValueError(f"unknown block_mode: {config.block_mode}")


# ---------------------------------------------------------------------------
# Metrics (same family as benchmark_v2, pair-C only)
# ---------------------------------------------------------------------------


def _pair_similarity(
    transforms: np.ndarray,  # (L,B,d,d)
    p_idx: np.ndarray,
    q_idx: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    _, num_blocks, block_dim, _ = transforms.shape
    out = np.empty(p_idx.shape[0], dtype=np.float64)
    norm = float(num_blocks * block_dim)
    for start in range(0, p_idx.size, chunk_size):
        end = min(start + chunk_size, p_idx.size)
        tp = transforms[p_idx[start:end]]
        tq = transforms[q_idx[start:end]]
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


def _normalized_power_spectrum(curve: np.ndarray, num_bins: int = 32) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    signal = curve[1:] if curve.size > 1 else curve
    if signal.size == 0:
        return np.zeros(num_bins, dtype=np.float64)
    signal = signal - signal.mean()
    power = np.abs(np.fft.rfft(signal)) ** 2
    if power.size == 0:
        return np.zeros(num_bins, dtype=np.float64)
    bins = np.array_split(power, num_bins)
    rebinned = np.array([float(b.mean()) if b.size else 0.0 for b in bins], dtype=np.float64)
    total = float(rebinned.sum())
    return rebinned / total if total > 0 else rebinned


def _spectrum_mismatch(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    n = min(profile_a.size, profile_b.size)
    if n <= 1:
        return 0.0
    a = _normalized_power_spectrum(profile_a[:n], num_bins=32)
    b = _normalized_power_spectrum(profile_b[:n], num_bins=32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _nonseparability(
    *,
    keys: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
) -> float:
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


def _rope_like_score(row: dict[str, object]) -> float:
    keys = ("E_rel", "E_rope", "spectrum", "tail", "anisotropy")
    vals = []
    for k in keys:
        v = float(row[k])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("inf")
    return float(sum(vals))


# ---------------------------------------------------------------------------
# Pair C evaluation
# ---------------------------------------------------------------------------


def dims_for_track(track: str, F: int, D: int) -> tuple[int, int, int]:
    k = len(ROPE_COORDS)  # pair C -> 2
    if track == "matched-F":
        if F <= 0:
            raise ValueError("F must be positive")
        d_rope = 2 * k * int(F)   # 4F for pair C
        d_monster = 4 * int(F)
        return int(F), d_rope, d_monster
    if track == "equal-D":
        d_rope = int(D)
        d_monster = int(D)
        if d_rope <= 0:
            raise ValueError("D must be positive")
        if d_rope % (2 * k) != 0:
            raise ValueError(f"D={d_rope} must be divisible by 2*k={2*k} for pair C")
        if d_monster % 4 != 0:
            raise ValueError(f"D={d_monster} must be divisible by 4 for MonSTER")
        return d_monster // 4, d_rope, d_monster
    raise ValueError(f"unknown track: {track}")


def evaluate_pair_c(
    *,
    track: str,
    F: int,
    D: int,
    windows: tuple[int, ...],
    seed: int,
    max_pairs_per_window: int,
    monster_cfg: MonsterConfig,
) -> dict[str, object]:
    f_used, d_rope, d_monster = dims_for_track(track=track, F=F, D=D)
    rng = np.random.default_rng(seed)
    per_window: list[dict[str, float | int]] = []
    nonsep_vals: list[float] = []

    for length in windows:
        coords = canonical_coords(int(length))
        rope_xy, monster_txyz = select_pair_c_coords(coords)

        rope_tf = build_axial_rope_transforms(
            coords_active=rope_xy,
            dim=d_rope,
            theta_base=monster_cfg.theta_base,
            freq_scale=1.0,
            freq_exponent=1.0,
        )
        monster_tf = build_monster_block_transforms(
            positions_4d=monster_txyz,
            dim=d_monster,
            config=monster_cfg,
        )

        n_all = length * length
        n_pairs = int(min(max_pairs_per_window, n_all))
        p_idx = rng.integers(0, length, size=n_pairs, endpoint=False, dtype=np.int64)
        q_idx = rng.integers(0, length, size=n_pairs, endpoint=False, dtype=np.int64)

        delta = rope_xy[q_idx] - rope_xy[p_idx]
        delta_i = np.round(delta).astype(np.int64)
        keys, inv, counts = _group_indices_from_deltas(delta_i)
        n_groups = keys.shape[0]

        s_rope = _pair_similarity(rope_tf, p_idx, q_idx)
        s_monster = _pair_similarity(monster_tf, p_idx, q_idx)

        rope_means = _group_means(s_rope, inv, n_groups)
        monster_means = _group_means(s_monster, inv, n_groups)
        w = counts.astype(np.float64)

        e_rel = _weighted_mean((s_monster - monster_means[inv]) ** 2, np.ones_like(s_monster))
        e_rope = _weighted_mean((monster_means - rope_means) ** 2, w)

        _, rope_profile = _radial_profile(keys=keys, means=rope_means, counts=counts)
        _, monster_profile = _radial_profile(keys=keys, means=monster_means, counts=counts)
        spectrum = _spectrum_mismatch(monster_profile, rope_profile)
        tail = _tail_error(monster_profile, rope_profile)
        anisotropy = _anisotropy(keys=keys, means=monster_means, counts=counts)
        nonsep = _nonseparability(keys=keys, means=monster_means, counts=counts)
        nonsep_vals.append(float(nonsep))

        per_window.append(
            {
                "length": int(length),
                "num_pairs": int(n_pairs),
                "num_delta_groups": int(n_groups),
                "E_rel": float(e_rel),
                "E_rope": float(e_rope),
                "spectrum": float(spectrum),
                "tail": float(tail),
                "anisotropy": float(anisotropy),
                "nonseparability": float(nonsep),
            }
        )

    def avg(key: str) -> float:
        vals = [float(row[key]) for row in per_window if np.isfinite(float(row[key]))]
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    row: dict[str, object] = {
        "benchmark_version": BENCHMARK_VERSION,
        "pair": PAIR_NAME,
        "track": track,
        "active_coords": ",".join(MONSTER_COORDS),
        "F": int(f_used),
        "D_rope": int(d_rope),
        "D_monster": int(d_monster),
        "E_rel": avg("E_rel"),
        "E_rope": avg("E_rope"),
        "spectrum": avg("spectrum"),
        "tail": avg("tail"),
        "anisotropy": avg("anisotropy"),
        "nonseparability": float(np.mean(nonsep_vals)) if nonsep_vals else float("nan"),
        "monster_config": config_dict(monster_cfg),
        "windows": per_window,
        "time_active": False,
    }
    row["score"] = _rope_like_score(row)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_windows(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return DEFAULT_WINDOWS
    out = tuple(int(v) for v in values)
    if not out:
        raise ValueError("windows must be non-empty")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone Pair-C MonSTER benchmark.")
    p.add_argument("--track", type=str, default=DEFAULT_TRACK, choices=("matched-F", "equal-D"))
    p.add_argument("--F", type=int, default=DEFAULT_F, help="frequency count for matched-F")
    p.add_argument("--D", type=int, default=DEFAULT_D, help="embedding dim for equal-D")
    p.add_argument("--windows", type=int, nargs="*", default=list(DEFAULT_WINDOWS))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-pairs-per-window", type=int, default=DEFAULT_MAX_PAIRS_PER_WINDOW)

    # MonSTER hyperparameters
    p.add_argument("--t-unit", type=float, default=1.0)
    p.add_argument("--s-unit", type=float, default=1.0)
    p.add_argument("--theta-base", type=float, default=DEFAULT_THETA_BASE)
    p.add_argument("--freq-scale", type=float, default=1.0)
    p.add_argument("--freq-exponent", type=float, default=1.0)
    p.add_argument("--boost-scale", type=float, default=1.0)
    p.add_argument("--rotation-scale", type=float, default=1.0)
    p.add_argument("--axis-mode", type=str, default="xy_circle")
    p.add_argument("--axis-blend", type=float, default=1.0)
    p.add_argument("--block-mode", type=str, default="lorentz", choices=("lorentz", "dual_plane"))
    p.add_argument("--dual-freq-mode", type=str, default="interleaved", choices=("interleaved", "axis_grouped"))

    p.add_argument("--json-out", type=Path, default=None, help="optional path to save JSON report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)

    monster_cfg = MonsterConfig(
        t_unit=float(args.t_unit),
        s_unit=float(args.s_unit),
        theta_base=float(args.theta_base),
        freq_scale=float(args.freq_scale),
        freq_exponent=float(args.freq_exponent),
        boost_scale=float(args.boost_scale),
        rotation_scale=float(args.rotation_scale),
        axis_mode=str(args.axis_mode),
        axis_blend=float(args.axis_blend),
        block_mode=str(args.block_mode),
        dual_freq_mode=str(args.dual_freq_mode),
    )

    row = evaluate_pair_c(
        track=str(args.track),
        F=int(args.F),
        D=int(args.D),
        windows=windows,
        seed=int(args.seed),
        max_pairs_per_window=int(args.max_pairs_per_window),
        monster_cfg=monster_cfg,
    )

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
    print(f"  E_rel:             {float(row['E_rel']):.6f}")
    print(f"  E_rope:            {float(row['E_rope']):.6f}")
    print(f"  spectrum:          {float(row['spectrum']):.6f}")
    print(f"  tail:              {float(row['tail']):.6f}")
    print(f"  anisotropy:        {float(row['anisotropy']):.6f}")
    print(f"  nonseparability:   {float(row['nonseparability']):.6f}")
    print(f"  score:             {float(row['score']):.6f}")
    print()
    print("Per-window metrics:")
    for wr in row["windows"]:
        print(
            f"  L={int(wr['length']):4d}  "
            f"E_rel={float(wr['E_rel']):.6f}  "
            f"E_rope={float(wr['E_rope']):.6f}  "
            f"spectrum={float(wr['spectrum']):.6f}  "
            f"tail={float(wr['tail']):.6f}  "
            f"anis={float(wr['anisotropy']):.6f}  "
            f"nonsep={float(wr['nonseparability']):.6f}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"json_out:            {args.json_out}")


if __name__ == "__main__":
    main()
