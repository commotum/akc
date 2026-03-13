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


def build_block_transforms(
    *,
    bank: prepare.PositionBank,
    dim: int,
    config: MonsterConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    if dim % 4 != 0:
        raise ValueError(f"F-MonSTER requires dim divisible by 4; got dim={dim}")

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

    t = bank.monster_positions[:, 0:1]                    # (L, 1)
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
    I3 = np.eye(3, dtype=np.float64)[None, :, :]

    boost = np.zeros((length, num_freq, 4, 4), dtype=np.float64)
    boost[:, :, 0, 0] = ch
    boost[:, :, 0, 1:] = -sh[:, :, None] * axes[None, :, :]
    boost[:, :, 1:, 0] = -sh[:, :, None] * axes[None, :, :]
    boost[:, :, 1:, 1:] = I3[None, :, :, :] + (ch - 1.0)[:, :, None, None] * outer[None, :, :, :]

    rotate = np.zeros((length, num_freq, 4, 4), dtype=np.float64)
    rotate[:, :, 0, 0] = 1.0
    rotate[:, :, 1:, 1:] = (
        c[:, :, None, None] * I3[None, :, :, :]
        + s[:, :, None, None] * skew[None, :, :, :]
        + (1.0 - c)[:, :, None, None] * outer[None, :, :, :]
    )

    # Apply boost first, then rotate around the same axis.
    return rotate @ boost


def relative_kernel(
    *,
    bank: prepare.PositionBank,
    dim: int,
    config: MonsterConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    transforms = build_block_transforms(bank=bank, dim=dim, config=config)
    return prepare.relative_curve_from_block_transforms(transforms)
