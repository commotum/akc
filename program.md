# MonSTER autoresearch reset

This file **replaces** the old single-scalar “make MonSTER match RoPE exactly” program.

The old objective was secretly an **exact RoPE imitation score**. That objective rewarded deleting the MonSTER-specific pieces (`sinh/cosh`, Lorentz boosts, Minkowski structure) until the 4D block became a disguised axial-RoPE clone. That is a **metric failure**, not a search failure.

The new goal is different:

> Build a **useful, genuinely different MonSTER family** that keeps RoPE-like strengths
> (stable relative structure, multiscale behavior, good long-context kernels)
> **while preserving real MonSTER structure**
> (Lorentz boost/rotation coupling, Minkowski geometry, non-separable spatial reasoning).

`master` remains the reference implementation. This branch is for **benchmark design, principled comparison, and safe autoresearch**.

---

## 1. Non-goals and hard guardrails

Do **not** optimize MonSTER into plain RoPE with extra steps.

### Forbidden mainline wins

The following are **not acceptable** as “improvements” for any **time-active** MonSTER variant (`t` is active):

- removing `sinh/cosh`
- zeroing out or bypassing Lorentz boosts
- replacing the Lorentz block with two Euclidean rotation planes
- forcing `T^T T ≈ I` everywhere as the main path to a better score
- turning the 4D MonSTER block into an exact axial-RoPE replica

Such runs may be logged **only as ablations / controls**, clearly labeled as RoPE-clone controls. They must **not** be adopted as the mainline target.

### Required MonSTER ingredients for time-active variants

If `t` is active, the candidate must keep:

- explicit Minkowski signature `(-,+,+,+)`
- real Lorentz-style boost behavior (`sinh/cosh` or exact equivalent)
- coupled boost + rotation behavior inside the 4D block
- explicit, numerically consistent unit handling (`c=1` normalized lattice units or equivalent)

Spatial-only MonSTER variants (`x`, `x,y`, `x,y,z`) do **not** need boost activity, because no time axis is active.

---

## 2. What we compare

We evaluate the following seven pairings.

| ID | RoPE side | MonSTER side | Optimization target |
|---|---|---|---|
| A | 1D RoPE | `t` only MonSTER | structural control; RoPE comparison is descriptive only |
| B | 1D RoPE | `x` only MonSTER | direct 1D spatial anchor |
| C | 2D axial RoPE | `x,y` only MonSTER | **primary calibration benchmark** |
| D | 2D axial RoPE | `t,x` only MonSTER | RoPE on `Δt = 0` slice + time-active structural metrics |
| E | 3D axial RoPE | `t,x,y` only MonSTER | RoPE on `Δt = 0` slice + time-active structural metrics |
| F | 3D axial RoPE | `x,y,z` only MonSTER | 3D spatial anchor |
| G | 4D axial RoPE | `t,x,y,z` MonSTER | RoPE on `Δt = 0` slice + time-active structural metrics |

### Primary benchmark

The main “are we sane?” comparator is:

- **2D axial RoPE vs `x,y` only MonSTER**

Reason:

- exact 4-to-4 width match
- exact frequency-block match at equal dimension
- no incentive to delete boosts just to mimic a Euclidean time axis
- directly relevant to multidimensional / non-flattened spatial reasoning

---

## 3. Stop comparing raw sub-blocks directly

Do **not** compare 2D RoPE pairs to 4D MonSTER blocks directly.

Compare only:

- **matched frequency count**
- **block-agnostic observables**

The families allocate frequency blocks differently:

- 1D RoPE: `F = d / 2`
- `k`-axis axial RoPE: `F = d / (2k)`
- MonSTER: `F = d / 4`

So “same embedding width” is only structurally exact when `k = 2`.

### Primary fairness rule: match frequency count `F`

Use:

```text
D_rope    = 2 k F
D_monster = 4 F
```

This gives both families the same number of frequency ranks.

Implications:

- 1D: RoPE dim `= 2F`, MonSTER dim `= 4F`
- 2D: RoPE dim `= 4F`, MonSTER dim `= 4F`
- 3D: RoPE dim `= 6F`, MonSTER dim `= 4F`
- 4D: RoPE dim `= 8F`, MonSTER dim `= 4F`

### Secondary fairness rule: report equal-width runs too

For engineering relevance, also report an **equal total width** track:

- hold total embedding width `D` fixed for both families
- compare only block-agnostic observables
- never use equal-width alone to claim structural equivalence except in the 2D case

### Required reporting

Every serious benchmark should state:

- pairing ID (`A`–`G`)
- fairness track (`matched-F` or `equal-D`)
- active coordinate subset
- `F`, `D_rope`, and `D_monster`
- frequency schedule details (`theta_base`, warps, exponents, tied vs untied settings)

---

## 4. Compare in coordinate space, not flattened offsets

Do **not** use a flattened 1D sequence offset as the primary geometry benchmark.

The benchmark must operate on actual displacement vectors:

```text
Δ ∈ Z^k
```

Use a shared canonical position bank generator that produces `(t,x,y,z)` coordinates for all windows.

For each pairing:

1. generate canonical coordinate positions
2. zero inactive coordinates on both sides
3. evaluate shell-averaged observables in coordinate space
4. use flattened sequence order only as an optional diagnostic, not as the main truth source

### Coordinate-shell benchmark types

- 1D: `Δ = (δ)`
- 2D: `Δ = (δx, δy)`
- 3D: `Δ = (δx, δy, δz)`
- spacetime: `Δ = (δt, δx, ...)`

### Block-agnostic observables

Use observables such as:

```text
S̄(Δ) = E_p [ < E(p) u , E(p + Δ) v > ]
```

with fixed random probes `u, v`, and/or dimension-normalized trace kernels such as:

```text
K(p, q) = trace(T_p^T T_q) / block_dim
```

These work regardless of whether the internal block is 2D, 4D, 6D, or 8D.

---

## 5. Two score families, not one

There is no single scalar “win” metric anymore.

We use **two score families**:

1. **RoPE-likeness**: are we keeping the useful RoPE properties?
2. **MonSTER-ness**: are we still genuinely MonSTER, especially when time is active?

Success is a **Pareto improvement**, not “zero error to RoPE.”

### 5.1 RoPE-likeness scores

These are the baseline-alignment scores.

#### Relative-law error

Measure how close interactions are to depending only on relative displacement:

```text
E_rel = E_{p,Δ} [ ( S(p, p+Δ) - S̄(Δ) )^2 ]
```

Lower is better.

#### Spatial RoPE mismatch

For the relevant **spatial slice** of a pairing, compare shell-averaged MonSTER responses to the paired RoPE baseline:

```text
E_rope = E_{Δ in D_spatial} [ ( S̄_monster(Δ) - S̄_rope(Δ) )^2 ]
```

Lower is better.

#### Spectrum mismatch

Compare the Fourier / spectral structure of `S̄(Δ)`.

Lower is better.

#### Attenuation / locality / tail error

Compare long-range decay, aliasing, and tail behavior.

Lower is better.

#### Spatial anisotropy

Measure shell variance:

```text
A(r) = Var_{||Δ|| = r} S̄(Δ)
```

Lower is better if the goal is improved isotropy over axis-factorized axial RoPE.

#### Preservation checks

Also report:

- shift invariance
- extensibility to higher dimensions / active subsets
- norm or metric preservation on the relevant slice

### 5.2 MonSTER-ness scores

These are the structure-preservation scores.

#### Minkowski preservation

For time-active variants, require low error under the active Minkowski metric:

```text
E_eta = E_Δ [ || T(Δ)^T η T(Δ) - η ||_F ]
```

Lower is better.

#### Boost activity

For time-active variants, require non-trivial non-Euclidean behavior:

```text
N_boost = E_Δ [ || T(Δ)^T T(Δ) - I ||_F ]
```

Higher means the transform is not collapsing back to pure Euclidean orthogonal rotation.

#### Cone / signature discrimination

For time-active variants, compare matched-size timelike vs spacelike displacements.

Higher separation is better.

#### Non-separability score

Measure how far the kernel is from a purely axis-factorized product form.

Higher means we are keeping genuinely coupled MonSTER structure.

#### Causal anisotropy probe

Report whether the encoder distinguishes causal classes or signatures in a stable, interpretable way.

### Hard failure condition for time-active variants

Any **time-active** MonSTER candidate automatically fails if:

- `E_eta` is acceptable **but**
- `N_boost` is near zero

This is the Euclidean-collapse detector.

In plain terms:

> a candidate does **not** count as a win if it preserves the Minkowski form only by collapsing into plain Euclidean RoPE-like rotations.

---

## 6. Optimization rules by pairing

### Spatial-only pairings

This applies to:

- B: `x`
- C: `x,y`
- F: `x,y,z`

Optimize primarily for:

- `E_rel`
- `E_rope`
- spectrum mismatch
- tail / attenuation error
- anisotropy / isotropy where relevant

Report non-separability as a positive secondary property.

### Time-active pairings

This applies to:

- A: `t`
- D: `t,x`
- E: `t,x,y`
- G: `t,x,y,z`

Do **not** optimize these to full axial RoPE with a single scalar.

Instead:

- compare **spatial slices** to RoPE
- compare **time-containing behavior** to MonSTER-only structural metrics

Required decision order:

1. **feasibility gate**
   - reject candidates that fail Minkowski preservation or Euclidean-collapse detection
2. **RoPE-like sanity on spatial slices**
   - among feasible candidates, prefer lower spatial-slice RoPE mismatch
3. **MonSTER structural quality**
   - prefer higher boost activity, cone separation, and useful non-separability

For pair A (`t` only), RoPE is a descriptive reference only. It is **not** a zero-error target.

---

## 7. Constrain the MonSTER axis family for fair ablations

The axis family must be restricted to the active spatial subspace.

Otherwise inactive directions leak into the structural prior and the ablation is not clean.

Use:

- `x` only / `t,x` only: axis fixed to `e_x`
- `x,y` only / `t,x,y` only: axes sampled on the unit circle in the `xy` plane
- `x,y,z` only / `t,x,y,z`: axes sampled on the sphere
- `t` only: no spurious spatial-axis prior

That way “x-only” really means x-only.

---

## 8. Benchmark phases

### Phase 0 — Rewrite the harness

Before any large search loop, the harness must support:

- coordinate-space displacement evaluation
- pairing IDs `A`–`G`
- matched-`F` and equal-`D` tracks
- RoPE-likeness + MonSTER-ness metrics
- feasibility gates for time-active variants
- spatial-slice evaluation for time-active pairings

The old single `overall_score` / exact-RoPE setup is now **legacy only**.

### Phase 1 — Primary calibration

Start with:

- **C: 2D axial RoPE vs `x,y` only MonSTER**

This is the cleanest direct anchor.

### Phase 2 — Spatial expansion

Then extend to:

- B: `x`
- F: `x,y,z`

### Phase 3 — Time-active structural benchmarks

Then extend to:

- A: `t`
- D: `t,x`
- E: `t,x,y`
- G: `t,x,y,z`

These are governed by the feasibility-gated two-family scorecard, **not** by exact RoPE imitation.

---

## 9. What the agent may edit

Editable:

- `experiment.py`
- `f_monster.py`
- any new benchmark / harness files explicitly created for the new protocol

Fixed unless intentionally versioned:

- benchmark definitions
- pairing matrix
- fairness-track definitions
- score-family definitions
- feasibility-gate logic

If a fixed benchmark definition changes, bump the benchmark version and log it clearly.

---

## 10. Required run logging

The old scalar-only `results.tsv` format is obsolete.

Use a per-run artifact that records at minimum:

- git commit
- benchmark version
- pairing ID
- fairness track
- active coordinates
- `F`, `D_rope`, `D_monster`
- `E_rel`
- `E_rope` or “N/A” when not primary
- spectrum mismatch
- tail / attenuation error
- anisotropy
- `E_eta`
- `N_boost`
- cone separation
- non-separability
- pass / fail for Euclidean-collapse detection
- short hypothesis description

A TSV is fine, but JSON or one-row-per-pair TSV is better than collapsing everything into one scalar.

Recommended TSV header:

```tsv
commit	benchmark_version	pair	track	active_coords	F	D_rope	D_monster	E_rel	E_rope	spectrum	tail	anisotropy	E_eta	N_boost	cone_sep	nonseparability	collapse_fail	status	description
```

### Status rules

- `keep` = Pareto improvement on the intended benchmark phase, without violating hard gates
- `discard` = no meaningful Pareto improvement
- `ablation` = intentionally RoPE-clone / collapse control
- `crash` = run failed

---

## 11. Experiment loop

Loop forever:

1. Check git state and current benchmark phase.
2. Pick one pairing and one fairness track to target.
3. State one concrete hypothesis.
4. Implement it.
5. Run the benchmark.
6. Reject any time-active candidate that fails the structural gates.
7. Compare against the current best for that pairing / phase.
8. Log the full metric row(s).
9. Keep only Pareto improvements.
10. Continue.

### Critical discipline

Do **not** accept a candidate just because some legacy “overall score” went down.

The correct question is:

> Did we get closer to RoPE where RoPE should be the reference, **without** destroying the MonSTER-specific structure that motivates the project?

---

## 12. Practical heuristics

### Good early moves

- start with pair C (`x,y` only) under matched-`F`
- restrict the axis family correctly before tuning anything else
- tune frequency scheduling before making the transform more complicated
- treat equal-`D` results as secondary until matched-`F` behavior is understood

### For time-active models

- keep boost magnitudes numerically stable but non-zero
- improve spatial-slice RoPE behavior **subject to** structural constraints
- use cone / signature probes to verify the model is still doing something genuinely non-Euclidean

### Bad moves

- deleting boosts to make a score smaller
- treating pair A (`t` only) as an exact RoPE target
- optimizing full `(t,x)` or `(t,x,y)` MonSTER directly to full axial RoPE with one scalar
- comparing raw 2D RoPE pairs to raw 4D MonSTER blocks

---

## 13. Bottom line

Success is **not**:

- “overall score goes to zero vs RoPE”

Success **is**:

- close enough to RoPE on the behaviors we care about
- while preserving distinctive Lorentz / Minkowski structure
- and doing so under a benchmark that is fair across unlike block structures

In short:

1. match **frequency count** first
2. compare **block-agnostic observables**, not raw sub-blocks
3. use **2D axial RoPE vs `x,y` only MonSTER** as the main anchor benchmark
4. for time-active MonSTERs, compare to RoPE only on **spatial slices**
5. enforce **hard MonSTER structural constraints** so boosts cannot silently vanish
6. evaluate in **coordinate space**, not flattened sequence offsets

This is the new program.
