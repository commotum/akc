# kernel autoresearch

This is a minimal autoresearch setup for tuning `f_monster.py` so its induced relative-position attention kernel matches the fixed axial RoPE reference as closely as possible.

The fixed evaluation harness is in `prepare.py`.  
The fixed benchmark target is in `rope.py`.  
The editable files are `experiment.py` and `f_monster.py`.

## Goal

Minimize the mismatch between

- `K_f-monster(Δ)`
- `K_axial-rope(Δ)`

across the target context windows:

- `64`
- `512`
- `1024`
- `2048`

The benchmark is sequence-relative. For each context window, positions are flattened into a near-cubic 3-axis layout:

- `64  -> 4 x 4 x 4`
- `512 -> 8 x 8 x 8`
- `1024 -> 8 x 8 x 16`
- `2048 -> 8 x 16 x 16`

The axial RoPE reference uses those fixed axial coordinates.  
The F-MonSTER candidate uses the corresponding fixed `(t, x, y, z)` bank where `t` is the linear sequence position and `(x, y, z)` are the axial coordinates.

## Fixed metric

The score is computed from the induced similarity curve:

`K(Δ) = average_{valid position pairs} average_{basis vectors} <E(p)x, E(q)x> / dim`

Lower is better.

For each window, the fixed harness computes:

- average similarity curve mismatch
- decay-envelope mismatch
- oscillation / spectrum mismatch
- aliasing error in the far tail
- effective-context-range error

and combines them into a single scalar:

`window_score = 0.40 * curve_rmse + 0.20 * envelope_rmse + 0.20 * spectrum_rmse + 0.10 * alias_error + 0.10 * effective_context_error`

The main optimization target is:

`overall_score = mean(window_score over {64, 512, 1024, 2048})`

The harness also prints `long_context_score` and `worst_window_score`.

## Setup

1. Agree on a run tag based on today’s date.
2. Create a fresh branch like `autoresearch/<tag>`.
3. Read these files for full context:
   - `prepare.py` — fixed position banks, cached axial RoPE reference curves, and the fixed mismatch metric.
   - `rope.py` — fixed axial RoPE benchmark target. Do not modify.
   - `f_monster.py` — editable candidate encoder.
   - `experiment.py` — the editable experiment runner.
4. Run the fixed prep step once:
   - `python prepare.py`
5. Initialize `results.tsv` with this header row:

```tsv
commit	overall_score	worst_window_score	status	description
```

6. Confirm setup looks good, then start the loop.

## What you can edit

You may edit:

- `experiment.py`
- `f_monster.py`

Everything else is fixed.

## What you should tune

Prioritize hypotheses around:

- `span`
- `top_delta`
- `unit`
- `theta_base`
- frequency spacing / warping
- boost vs rotation balance
- axis distribution (`fibonacci`, `cycle`, `hybrid`)
- any simple structural change in `f_monster.py` that improves kernel fit

Keep changes simple unless a more complex change clearly wins.

## Running an experiment

Run one experiment as:

```bash
python experiment.py > run.log 2>&1
```

Then inspect the score:

```bash
grep "^overall_score:\|^long_context_score:\|^worst_window_score:\|^window_[0-9]\+_score:" run.log
```

If the grep output is empty, the run crashed. Read the traceback with:

```bash
tail -n 50 run.log
```

## Optional visual diagnostics

To generate RoPE-style comparison plots for the current candidate, run:

```bash
python experiment.py --plots > run.log 2>&1
```

That writes dashboards under `run_artifacts/plots/` and adds these lines to the log:

```text
plot_manifest:          run_artifacts/plots/manifest.json
plot_summary:           run_artifacts/plots/all_windows_kernel_curves.png
plot_window_64:         run_artifacts/plots/window_64_dashboard.png
plot_window_512:        run_artifacts/plots/window_512_dashboard.png
plot_window_1024:       run_artifacts/plots/window_1024_dashboard.png
plot_window_2048:       run_artifacts/plots/window_2048_dashboard.png
```

Each per-window dashboard includes:

- axial RoPE phase-probe heatmap
- F-MonSTER phase-probe heatmap in local tangent-plane coordinates
- average similarity curve `K(Δ)`
- RoPE similarity matrix `Q(p) · K(q)`
- F-MonSTER similarity matrix `Q(p) · K(q)`
- similarity-matrix difference

For a quicker visual pass, restrict the dashboards to a subset:

```bash
python experiment.py --plots --plot-windows 64 2048 > run.log 2>&1
```

## Output format

A successful run prints a summary like:

```text
---
overall_score:          0.123456
mean_curve_rmse:        0.098765
long_context_score:     0.150000
worst_window:           2048
worst_window_score:     0.201234
window_64_score:        0.012345
window_512_score:       0.067890
window_1024_score:      0.103210
window_2048_score:      0.201234
report_json:            run_artifacts/latest_report.json
curves_npz:             run_artifacts/latest_curves.npz
```

Lower is better.

## Logging results

After each run, append a row to `results.tsv`:

```tsv
commit	overall_score	worst_window_score	status	description
a1b2c3d	0.123456	0.201234	keep	baseline fibonacci axes with default boost
b2c3d4e	0.102300	0.160000	keep	reduce boost scale and warp frequencies slower
c3d4e5f	0.140000	0.250000	discard	switch to cycle axes only
d4e5f6g	0.000000	0.000000	crash	typo in transform assembly
```

Use:

- `keep` if `overall_score` improved
- `discard` if it did not improve
- `crash` if the experiment failed

## Experiment loop

Loop forever:

1. Check git state and current best commit.
2. Propose one concrete hypothesis.
3. Edit `experiment.py` and/or `f_monster.py`.
4. Commit the change.
5. Run:
   - `python experiment.py > run.log 2>&1`
6. Inspect:
   - `overall_score`
   - `long_context_score`
   - `worst_window_score`
   - per-window scores
7. If the run crashed, read the traceback and either:
   - fix the bug and re-run if the idea still makes sense, or
   - log a crash and move on
8. Append the result to `results.tsv`
9. If `overall_score` improved, advance the branch
10. If it did not improve, reset back to the previous best commit

## Heuristics for good hypotheses

Good low-complexity first moves:

- weaken the boost component relative to the rotation component
- retune `span` / `top_delta` so the slowest and fastest frequencies align better with the RoPE windows
- swap `fibonacci` axes for `cycle` or a `hybrid`
- warp the frequency ranks (`freq_exponent`)
- change the candidate `theta_base`

If progress stalls, try more structural ideas inside `f_monster.py`, but keep the fixed evaluation untouched.

## Never stop

Once the loop begins, do not pause for approval between experiments. Keep proposing, implementing, running, and keeping only improvements until interrupted.
