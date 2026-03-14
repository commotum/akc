# AKC MonSTER Benchmark Findings (through `monster-v2.1`)

## Executive Summary
We moved the project from "make MonSTER look exactly like RoPE" to a two-family benchmark that balances RoPE-like behavior with MonSTER-specific structure. The strongest result is that we now have stable, reproducible tuned settings for all seven pairings (`A..G`) under both fairness tracks (`matched-F` and `equal-D`), without deleting boosts or collapsing the time-active model into Euclidean-only behavior.

The most important technical turning point was time-active stability: default boosts were numerically unstable at long windows, so we tuned into a small-but-nonzero boost regime (`boost_scale=0.0005`) and then optimized frequency schedules on top of that.

## What We Changed

1. Implemented the new benchmark program in code (pairings `A..G`, fairness tracks, coordinate-space evaluation, dual score families).
1. Tuned spatial anchors first (`C` primary, then `B` and `F`).
1. Tuned time-active pairings (`A`, `D`, `E`, `G`) with hard emphasis on feasibility and structural retention (`E_eta`, `N_boost`, cone behavior).
1. Hardened automatic acceptance logic:
   - numerically blown-up / non-finite time-active runs now fail feasibility and are auto-discarded.
1. Introduced `benchmark_version = monster-v2.1` because metric behavior changed:
   - added a synthetic cone-separation fallback for cases where sampled coordinate deltas contain only one causal class.
   - scoped auto-comparison to the same benchmark version only.

## Hypotheses and Outcomes

### H1. Primary `C` can be improved mainly by frequency schedule tuning.
Outcome: **Supported**.
- `C` improved substantially from early baseline to best region around `freq_exponent ~ 1.6`.
- Extra knobs (`freq_scale`, `rotation_scale`, `theta_base`) did not produce a clearly robust win over the `freq_exponent`-led solution.
- Multi-seed checks showed near-ties in the top region, so we kept the simpler tuned setting as reference.

### H2. Spatial pairs `B` and `F` have different optimal exponent bands.
Outcome: **Supported**.
- `B` preferred `freq_exponent ~ 1.5`.
- `F` preferred `freq_exponent ~ 1.71`.

### H3. Time-active defaults are feasible as-is.
Outcome: **Rejected**.
- With default boost settings, time-active runs blew up numerically (`sinh/cosh` overflow) and produced invalid metrics.
- Stable regime required `boost_scale=0.0005` (small but nonzero), which preserved nonzero boost activity.

### H4. Time-active cone separation for `D (t,x)` should appear naturally from sampled windows.
Outcome: **Rejected for current sampling geometry**.
- In current coordinate sampling, `D` frequently had only timelike deltas, so cone separation was `NA`.
- Added synthetic cone fallback in `monster-v2.1`; `D` now reports finite cone separation.

## Final Tuned Reference Settings (v2.1)

These are the current "kept" reference settings used to generate fresh `monster-v2.1` rows.

| Pair | Setting (key knobs) | matched-F score* | equal-D score* |
|---|---|---:|---:|
| A | `exp=1.0`, `boost=0.0005` | `0.678829` | `0.688041` |
| B | `exp=1.5` | `0.063600` | `0.063723` |
| C | `exp=1.625` | `0.040978` | `0.040978` |
| D | `exp=2.7`, `boost=0.0005` | `0.271742` | `0.271742` |
| E | `exp=3.5`, `boost=0.0005` | `0.173042` | `0.165863` |
| F | `exp=1.7125` | `0.044542` | `0.043298` |
| G | `exp=3.0`, `boost=0.0005` | `0.110936` | `0.109830` |

\*score = `E_rel + E_rope + spectrum + tail + anisotropy` (the same scalar used for run ranking within a benchmark version).

## Time-Active Structural Findings

- `E_eta` remained effectively zero in the tuned stable regime (good Minkowski consistency).
- `N_boost` stayed nonzero and meaningful:
  - `D ~ 0.324664`
  - `E ~ 0.403705`
  - `G ~ 0.356699`
- `collapse_fail=False` for tuned rows.
- Cone separation is now finite in all time-active tuned baselines under `v2.1`:
  - `D ~ 0.214584` (via synthetic fallback)
  - `E ~ 0.062936`
  - `G ~ 0.028114`

## Practical Takeaways

1. The main spatial anchor (`C`) is healthy and competitive without erasing MonSTER identity.
1. Time-active MonSTER is viable, but only inside a controlled boost regime; otherwise it fails numerically.
1. Higher exponents are helpful for some time-active pairs (`D/E/G`) once stability is enforced.
1. `monster-v2.1` is a better baseline than `v2.0` because it correctly handles numerical failure and causal-separation observability.

## Artifacts and Commits

- Primary results log: `benchmark_results.tsv`
- Key implementation files: `benchmark_v2.py`, `experiment.py`, `f_monster.py`
- Recent checkpoint commits:
  - `ef96e4b` (phase 2/3 tuning + numerical gate hardening)
  - `685180f` (equal-D and multiseed validation rows)
  - `d9f41db` (phase 3 local refinement + dense cone probe)
  - `6de82bd` (`monster-v2.1` cone fallback + version-scoped auto status)

