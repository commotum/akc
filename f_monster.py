"""
Agent-editable F-MonSTER candidate encoder.

Edit this file and/or experiment.py during autoresearch.
The fixed metric and the fixed reference live in prepare.py and rope.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

import prepare


@dataclass(frozen=True)
class MonsterConfig:
    # Geometry / span
    span: float = 2.0 * math.pi
    top_delta: float = float(max(prepare.TARGET_WINDOWS))
    unit: float | None = None

    # Frequency family
    theta_base: float = prepare.REFERENCE_THETA_BASE
    freq_scale: float = 1.0
    freq_exponent: float = 1.0

    # Relative weight of the hyperbolic and rotational pieces
    boost_scale: float = 1.0
    rotation_scale: float = 1.0

    # Axis distribution
    axis_mode: str = "fibonacci"   # fibonacci | cycle | hybrid
    axis_blend: float = 1.0        # for hybrid: 1 => fibonacci, 0 => cycle

    # Block transform family
    block_mode: str = "lorentz"    # lorentz | dual_plane
    dual_freq_mode: str = "interleaved"  # interleaved | axis_grouped


DEFAULT_CONFIG = MonsterConfig()


def config_dict(config: MonsterConfig) -> dict[str, float | str | None]:
    return asdict(config)


def resolved_unit(config: MonsterConfig) -> float:
    if config.unit is not None:
        return float(config.unit)
    if config.top_delta == 0:
        raise ValueError("top_delta must be non-zero when unit is not provided")
    return float(config.span) / float(config.top_delta)


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

    if mode == "fibonacci":
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
    bank: prepare.PositionBank,
    dim: int,
    config: MonsterConfig,
) -> np.ndarray:
    length = bank.length
    num_freq = dim // 4
    unit = resolved_unit(config)
    inv_freq = prepare.warped_frequencies(
        num_freq,
        theta_base=float(config.theta_base),
        exponent=float(config.freq_exponent),
        scale=float(config.freq_scale),
    )
    axes = build_axes(num_freq, config)

    t = bank.monster_positions[:, 0:1]                   # (L, 1)
    spatial = bank.monster_positions[:, 1:4]             # (L, 3)
    proj = spatial @ axes.T                              # (L, F)

    phi = t * (unit * float(config.boost_scale)) * inv_freq[None, :]
    theta = proj * (unit * float(config.rotation_scale)) * inv_freq[None, :]

    ch = np.cosh(phi)
    sh = np.sinh(phi)
    c = np.cos(theta)
    s = np.sin(theta)

    outer = axes[:, :, None] * axes[:, None, :]          # (F, 3, 3)
    skew = skew_matrices(axes)                           # (F, 3, 3)
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

    # Apply boost first, then rotate around the same axis.
    return rotate @ boost


def _build_dual_plane_block_transforms(
    *,
    bank: prepare.PositionBank,
    dim: int,
    config: MonsterConfig,
) -> np.ndarray:
    """
    Build 4x4 blocks as two independent Euclidean rotations:
    - one rotation in the (time, axis) plane
    - one rotation in the local tangent (u, v) plane

    This packs two frequency tracks per 4D block.
    """
    length = bank.length
    num_freq = dim // 4
    unit = resolved_unit(config)
    axes = build_axes(num_freq, config)
    u, v = tangent_frames(axes)

    spatial = bank.monster_positions[:, 1:4]             # (L, 3)
    proj = spatial @ axes.T                              # (L, F)

    dual_freq_mode = config.dual_freq_mode.lower()
    if dual_freq_mode == "interleaved":
        inv_freq = prepare.warped_frequencies(
            2 * num_freq,
            theta_base=float(config.theta_base),
            exponent=float(config.freq_exponent),
            scale=float(config.freq_scale),
        )
        inv_phi = inv_freq[0::2]
        inv_theta = inv_freq[1::2]
    elif dual_freq_mode in {"axis_grouped", "grouped"}:
        axis_count = prepare.AXIAL_COORD_DIMS
        if num_freq % axis_count != 0:
            raise ValueError(
                f"dual axis_grouped packing requires num_freq divisible by {axis_count}; got {num_freq}"
            )
        if config.axis_mode.lower() not in {"cycle", "cycle_xyz"}:
            raise ValueError("dual axis_grouped packing requires axis_mode='cycle'")

        blocks_per_axis = num_freq // axis_count
        freq_per_axis = 2 * blocks_per_axis
        axis_inv = prepare.warped_frequencies(
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

    phi = proj * (unit * float(config.boost_scale)) * inv_phi[None, :]
    theta = proj * (unit * float(config.rotation_scale)) * inv_theta[None, :]

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

    # Per-frequency orthonormal basis [time, axis, u, v].
    e_time = np.zeros((num_freq, 4), dtype=np.float64)
    e_time[:, 0] = 1.0
    e_axis = np.zeros((num_freq, 4), dtype=np.float64)
    e_axis[:, 1:] = axes
    e_u = np.zeros((num_freq, 4), dtype=np.float64)
    e_u[:, 1:] = u
    e_v = np.zeros((num_freq, 4), dtype=np.float64)
    e_v[:, 1:] = v
    basis = np.stack((e_time, e_axis, e_u, e_v), axis=-1)  # (F, 4, 4), basis vectors as columns.

    transformed = np.einsum("fij,lfjk->lfik", basis, blocks)
    return np.einsum("lfij,fkj->lfik", transformed, basis)


def build_block_transforms(
    *,
    bank: prepare.PositionBank,
    dim: int,
    config: MonsterConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    if dim % 4 != 0:
        raise ValueError(f"F-MonSTER requires dim divisible by 4; got dim={dim}")

    mode = config.block_mode.lower()
    if mode == "lorentz":
        return _build_lorentz_block_transforms(bank=bank, dim=dim, config=config)
    if mode in {"dual_plane", "dual"}:
        return _build_dual_plane_block_transforms(bank=bank, dim=dim, config=config)
    raise ValueError(f"unknown block_mode: {config.block_mode}")


def relative_kernel(
    *,
    bank: prepare.PositionBank,
    dim: int,
    config: MonsterConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    transforms = build_block_transforms(bank=bank, dim=dim, config=config)
    return prepare.relative_curve_from_block_transforms(transforms)
