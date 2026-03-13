"""
Read-only axial RoPE reference encoder.

This is the fixed benchmark target. Autoresearch agents should not modify this file.
"""

from __future__ import annotations

import numpy as np

import prepare


def validate_dim(dim: int, coord_dims: int = prepare.AXIAL_COORD_DIMS) -> None:
    if dim % (2 * coord_dims) != 0:
        raise ValueError(
            f"axial RoPE requires dim divisible by 2 * coord_dims; got dim={dim}, coord_dims={coord_dims}"
        )


def build_block_transforms(
    *,
    bank: prepare.PositionBank,
    dim: int,
    theta_base: float = prepare.REFERENCE_THETA_BASE,
) -> np.ndarray:
    validate_dim(dim, bank.axial_positions.shape[1])

    length = bank.length
    coord_dims = bank.axial_positions.shape[1]
    pair_count = dim // (2 * coord_dims)
    inv_freq = prepare.base_frequencies(pair_count, theta_base)

    phase = bank.axial_positions[:, :, None] * inv_freq[None, None, :]
    cos = np.cos(phase)
    sin = np.sin(phase)

    transforms = np.empty((length, coord_dims * pair_count, 2, 2), dtype=np.float64)
    block = 0
    for axis_idx in range(coord_dims):
        block_slice = slice(block, block + pair_count)
        transforms[:, block_slice, 0, 0] = cos[:, axis_idx, :]
        transforms[:, block_slice, 0, 1] = -sin[:, axis_idx, :]
        transforms[:, block_slice, 1, 0] = sin[:, axis_idx, :]
        transforms[:, block_slice, 1, 1] = cos[:, axis_idx, :]
        block += pair_count

    return transforms


def relative_kernel(
    *,
    bank: prepare.PositionBank,
    dim: int,
    theta_base: float = prepare.REFERENCE_THETA_BASE,
) -> np.ndarray:
    transforms = build_block_transforms(bank=bank, dim=dim, theta_base=theta_base)
    return prepare.relative_curve_from_block_transforms(transforms)
