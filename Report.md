# Kernel Autoresearch Report

Date: March 13, 2026  
Branch: `autoresearch/2026-03-13` (kept separate from `master`)

## Scope

This report summarizes every experiment marked `keep` in `results.tsv` while tuning `f_monster.py` + `experiment.py` against the fixed harness in `prepare.py`.

Target metric:

- Minimize `overall_score` (mean of window scores for 64, 512, 1024, 2048).
- Also track `worst_window_score`.

## Kept Experiment Timeline

| Step | Commit | Hypothesis | Key Change(s) | overall_score | worst_window_score | Delta vs previous keep |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 0 | `51404ea` | Baseline | Default config (`fibonacci`, `unit=None`, Lorentz-like mode) | 7.159192 | 27.935448 | - |
| 1 | `b5e164a` | Remove boost mismatch + align axes | `unit=1.0`, `boost_scale=0.0`, `axis_mode="cycle"` | 0.049328 | 0.062717 | -7.109864 (99.31% better) |
| 2 | `5126e79` | Warp frequency ranks | `freq_exponent=1.2` | 0.037833 | 0.048386 | -0.011495 (23.30% better) |
| 3 | `46ea182` | Stronger warp + higher scale | `freq_exponent=1.4`, `freq_scale=1.3` | 0.021784 | 0.026226 | -0.016049 (42.42% better) |
| 4 | `89df608` | Structural change: dual-plane blocks | Added `block_mode="dual_plane"` and enabled it in config | 0.004696 | 0.005388 | -0.017088 (78.44% better) |
| 5 | `4e04a0e` | Tune dual-plane boost | `boost_scale=1.1` | 0.002991 | 0.004247 | -0.001705 (36.31% better) |
| 6 | `f0c388d` | Fine tune plane balance | `boost_scale=1.12`, `rotation_scale=0.98` | 0.002909 | 0.004166 | -0.000082 (2.74% better) |
| 7 | `39c0034` | Axis-grouped dual-plane frequencies | Added `dual_freq_mode="axis_grouped"` and enabled it with equal plane scales | 0.000000 | 0.000000 | -0.002909 (100.00% better) |

## Details By Kept Hypothesis

### 1) `b5e164a` — cycle axes, unit=1, boost=0

- Hypothesis: most baseline error comes from temporal boost drift and axis-distribution mismatch.
- Edits in `experiment.py`:
- Set `unit=1.0` (instead of auto `span/top_delta`).
- Set `boost_scale=0.0`.
- Switched from `axis_mode="fibonacci"` to `axis_mode="cycle"`.
- Result: biggest early win, cutting catastrophic long-context error.

### 2) `5126e79` — frequency rank warp

- Hypothesis: RoPE alignment improves if high-rank frequencies are warped.
- Edit in `experiment.py`:
- `freq_exponent: 1.0 -> 1.2`.
- Result: consistent across windows, especially long windows.

### 3) `46ea182` — stronger warp + frequency scale

- Hypothesis: stronger warp plus mild scaling better fits the reference envelope.
- Edits in `experiment.py`:
- `freq_exponent: 1.2 -> 1.4`.
- `freq_scale: 1.0 -> 1.3`.
- Result: major additional drop before structural changes.

### 4) `89df608` — dual-plane block transform family

- Hypothesis: one Lorentz-style block is too constrained; two Euclidean rotation planes per 4D block should match axial RoPE kernel better.
- Edits in `f_monster.py`:
- Added config field `block_mode` (`lorentz | dual_plane`).
- Added `tangent_frames(...)`.
- Refactored existing path into `_build_lorentz_block_transforms(...)`.
- Added `_build_dual_plane_block_transforms(...)`:
- rotation in `(time, axis)` plane and `(u, v)` tangent plane.
- dispatch in `build_block_transforms(...)` by `block_mode`.
- Edits in `experiment.py`:
- Enabled `block_mode="dual_plane"`.
- Reset tuned scalar knobs to neutral (`freq_scale=1.0`, `freq_exponent=1.0`, `boost_scale=1.0`).
- Result: structural jump to low-millisecond error regime.

### 5) `4e04a0e` — dual-plane boost tune

- Hypothesis: small anisotropy in plane scales improves fit.
- Edit in `experiment.py`:
- `boost_scale: 1.0 -> 1.1`.
- Result: meaningful but smaller gain.

### 6) `f0c388d` — fine boost/rotation tuning

- Hypothesis: near-1.0 asymmetric scaling is slightly better than symmetric.
- Edits in `experiment.py`:
- `boost_scale: 1.1 -> 1.12`.
- `rotation_scale: 1.0 -> 0.98`.
- Result: marginal improvement; near local optimum for interleaved dual-plane packing.

### 7) `39c0034` — axis-grouped dual-plane packing (exact match)

- Hypothesis: dual-plane frequencies should be grouped by cycle-axis families (not interleaved globally) to mirror axial RoPE structure exactly.
- Edits in `f_monster.py`:
- Added config field `dual_freq_mode` (`interleaved | axis_grouped`).
- In dual-plane path, added `axis_grouped` branch:
- validates divisibility by `AXIAL_COORD_DIMS`.
- requires cycle-axis mode.
- assigns per-axis frequency bank and maps by `block_rank`.
- Edits in `experiment.py`:
- Enabled `dual_freq_mode="axis_grouped"`.
- Set `boost_scale=1.0`, `rotation_scale=1.0`.
- Result: exact kernel match on all benchmark windows (`overall_score=0`, `worst_window_score=0`).

## Final Best Config (Exact Match)

Current best candidate (commit `39c0034`, preserved on this branch):

- `axis_mode="cycle"`
- `unit=1.0`
- `freq_scale=1.0`
- `freq_exponent=1.0`
- `block_mode="dual_plane"`
- `dual_freq_mode="axis_grouped"`
- `boost_scale=1.0`
- `rotation_scale=1.0`

## Branch Policy

This work is intentionally kept off `master` to preserve `master` as the reference implementation baseline.  
The research branch now contains both the full tuning history (`results.tsv`) and the exact-match implementation path.
