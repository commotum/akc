from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np

from v12 import TriadMonSTERFastVec, apply_monster_triad_fast_vec


# ============================================================================
# Editable hyperparameters
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "monster_hyperparam_grid.png"

IMAGE_SIZE = 16
EMBED_DIM = 768
THETA_BASE = 10_000.0
TOP_DELTA = 16.0
SEED = 0

QUERY_X = 8
QUERY_Y = 8
QUERY_ON_VALUE = 100.0

KEY_T_VALUE = 0.0
QUERY_T_VALUES = [-8.0, -16.0, -24.0]
TANH_K_VALUES = [0.0001, 0.001, 0.01]
UNIT_SCALE = 1.0

FIGSIZE = (13.7, 4.8)
AXES_PAD = 0.08
CBAR_RATIO = 0.05

ETA4_NEGPOS = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)


def random_embedding(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, 1.0, d)
    v = v / np.linalg.norm(v) * np.sqrt(d)
    return v


def make_query_image(size: int, query_x: int, query_y: int, on_value: float) -> np.ndarray:
    image = np.zeros((size, size), dtype=float)
    image[query_y, query_x] = on_value
    return image


def centered_xy_coords(size: int) -> tuple[np.ndarray, np.ndarray]:
    vals = np.arange(-(size / 2) + 0.5, size / 2, 1.0)
    return np.meshgrid(vals, vals, indexing="xy")


def tanh_time(t_value: float, k_value: float) -> float:
    return float(np.tanh(k_value * t_value))


def make_standard_positions(size: int, t_value: float, k_value: float) -> np.ndarray:
    x, y = centered_xy_coords(size)
    t = np.full_like(x, fill_value=tanh_time(t_value, k_value), dtype=np.float64)
    z = np.zeros_like(x, dtype=np.float64)
    return np.stack((t, x, y, z), axis=-1).reshape(-1, 4)


def metric_dot_batch(query_vec: np.ndarray, keys: np.ndarray, metric: np.ndarray) -> np.ndarray:
    query_metric = query_vec.reshape(-1, 4) @ metric
    return np.sum(keys.reshape(keys.shape[0], -1, 4) * query_metric[None, :, :], axis=(1, 2))


def transform_positions(
    base_vector: np.ndarray,
    positions: np.ndarray,
    monster: TriadMonSTERFastVec,
) -> np.ndarray:
    transformed_keys = np.empty((positions.shape[0], monster.dim), dtype=np.float64)
    for idx, position in enumerate(positions):
        transformed_keys[idx] = apply_monster_triad_fast_vec(
            base_vector,
            monster.forward(position),
            dim=monster.dim,
        )
    return transformed_keys


def transform_vector_at_position(
    base_vector: np.ndarray,
    position: np.ndarray,
    monster: TriadMonSTERFastVec,
) -> np.ndarray:
    return apply_monster_triad_fast_vec(
        base_vector,
        monster.forward(position),
        dim=monster.dim,
    )


def score_transformed_vectors(
    transformed_query: np.ndarray,
    transformed_keys: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    logits = metric_dot_batch(transformed_query, transformed_keys, metric) / np.sqrt(transformed_query.size)
    side = int(np.sqrt(transformed_keys.shape[0]))
    return logits.reshape(side, side)


def plot_grid(
    query_image: np.ndarray,
    row_k_values: list[float],
    panel_titles_by_row: list[list[str]],
    negpos_maps_by_row: list[list[np.ndarray]],
    output_path: Path,
) -> None:
    num_rows = len(row_k_values)
    if num_rows == 0:
        raise RuntimeError("Expected at least one tanh-k row.")

    all_panels = [panel for row in negpos_maps_by_row for panel in row]
    if not all_panels:
        raise RuntimeError("Expected MonSTER panel images.")

    fig = plt.figure(figsize=(FIGSIZE[0], FIGSIZE[1] * num_rows))
    negpos_norm = Normalize(
        vmin=min(float(panel.min()) for panel in all_panels),
        vmax=max(float(panel.max()) for panel in all_panels),
    )
    grid = GridSpec(
        num_rows,
        5,
        figure=fig,
        left=0.03,
        right=0.97,
        top=0.93,
        bottom=0.06,
        hspace=0.28,
        wspace=AXES_PAD,
        width_ratios=[
            1.0,
            1.0,
            1.0,
            1.0,
            CBAR_RATIO,
        ],
    )
    negpos_cbar_ax = fig.add_subplot(grid[:, 4])

    fig.suptitle(
        (
            f"MonSTERs comparison | dim={EMBED_DIM} | base={THETA_BASE:g} | "
            f"top_delta={TOP_DELTA:g} | query=({QUERY_X}, {QUERY_Y}) | rows=tanh(k*t)"
        ),
        fontsize=15,
    )

    right_last_image = None
    for row_idx, (k_value, row_titles, row_panels) in enumerate(
        zip(row_k_values, panel_titles_by_row, negpos_maps_by_row)
    ):
        binary_ax = fig.add_subplot(grid[row_idx, 0])
        binary_ax.imshow(query_image, cmap="viridis", vmin=0.0, vmax=QUERY_ON_VALUE, interpolation="nearest")
        if row_idx == 0:
            binary_ax.set_title("Binary input", fontsize=11)
        binary_ax.axis("off")
        binary_ax.text(
            -0.12,
            0.5,
            f"k={k_value:g}",
            transform=binary_ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
        )

        negpos_axes = [fig.add_subplot(grid[row_idx, idx]) for idx in (1, 2, 3)]
        for axis, title, panel in zip(negpos_axes, row_titles, row_panels):
            right_last_image = axis.imshow(panel, cmap="viridis", norm=negpos_norm)
            if row_idx == 0:
                axis.set_title(title, fontsize=11)
            axis.axis("off")

    fig.colorbar(right_last_image, cax=negpos_cbar_ax)
    fig.text(
        0.50,
        0.95,
        "Standard coordinates | metric (-,+,+,+) | columns are query t values",
        ha="center",
        va="center",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if EMBED_DIM % 12 != 0:
        raise ValueError("EMBED_DIM must be divisible by 12 for MonSTERs.")
    if not (0 <= QUERY_X < IMAGE_SIZE and 0 <= QUERY_Y < IMAGE_SIZE):
        raise ValueError("Query position must lie within the image grid.")

    rng = np.random.default_rng(SEED)
    base_vector = random_embedding(EMBED_DIM, rng)
    query_index = QUERY_Y * IMAGE_SIZE + QUERY_X
    monster = TriadMonSTERFastVec(dim=EMBED_DIM, base=THETA_BASE, top_delta=TOP_DELTA)
    monster.unit = UNIT_SCALE / TOP_DELTA

    row_k_values: list[float] = []
    panel_titles_by_row: list[list[str]] = []
    negpos_maps_by_row: list[list[np.ndarray]] = []

    for k_value in TANH_K_VALUES:
        panel_titles: list[str] = []
        negpos_maps: list[np.ndarray] = []

        standard_positions = make_standard_positions(IMAGE_SIZE, KEY_T_VALUE, k_value)
        standard_keys = transform_positions(base_vector, standard_positions, monster)
        standard_query_base = standard_positions[query_index].copy()

        for query_t in QUERY_T_VALUES:
            query_position = standard_query_base.copy()
            query_position[0] = tanh_time(query_t, k_value)
            transformed_query = transform_vector_at_position(base_vector, query_position, monster)
            negpos_maps.append(score_transformed_vectors(transformed_query, standard_keys, ETA4_NEGPOS))
            panel_titles.append(f"q t={query_t:g} -> tanh={tanh_time(query_t, k_value):.4f}")

        row_k_values.append(k_value)
        panel_titles_by_row.append(panel_titles)
        negpos_maps_by_row.append(negpos_maps)

    query_image = make_query_image(IMAGE_SIZE, QUERY_X, QUERY_Y, QUERY_ON_VALUE)
    plot_grid(query_image, row_k_values, panel_titles_by_row, negpos_maps_by_row, OUTPUT_PATH)

    print(f"Saved MonSTERs comparison grid to {OUTPUT_PATH}")
    print(f"Key t value: {KEY_T_VALUE}")
    print(f"Query t values: {QUERY_T_VALUES}")
    print(f"Tanh k values (rows): {TANH_K_VALUES}")
    for k_value in TANH_K_VALUES:
        transformed = [round(tanh_time(query_t, k_value), 6) for query_t in QUERY_T_VALUES]
        print(f"k={k_value:g} -> query tanh values: {transformed}")
    print(f"Unit scale: {UNIT_SCALE:.6g} / top_delta")
    print("All MonSTER panels use standard centered (x, y, z=0) coordinates.")
    print("The plot uses metric (-,+,+,+).")


if __name__ == "__main__":
    main()
