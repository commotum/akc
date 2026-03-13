# Big Picture: MonSTERs vs RoPE (Reset)

## 1) What we are actually trying to do

We are **not** trying to rebuild RoPE exactly with extra machinery.  
We are trying to build a **useful, genuinely different MonSTER family** that:

1. keeps RoPE-like strengths (stable relative structure, multiscale behavior, good long-context kernels), and  
2. preserves MonSTER-specific structure (Lorentz boost/rotation coupling, Minkowski geometry, non-separable spatial reasoning).

`master` remains the reference implementation. This branch is for benchmark design and principled comparison.

## 2) Updated assumptions (current, not legacy)

- Benchmark target is now **axial RoPE**, not only flattened 1D RoPE.
- The metric convention is now **\(-,+,+,+\)**.
- MonSTER should keep real MonSTER ingredients: **`sinh/cosh` boost behavior + spatial rotation**, not collapse into pure Euclidean RoPE clones.
- Unit handling should be explicit (`c=1` normalized lattice units or equivalent consistent scaling), so boost terms are meaningful and numerically stable.

## 3) Fair comparison set (the 7 requested matchups)

We compare each RoPE family to a MonSTER variant with the same active coordinate subset:

| Pair | RoPE side | MonSTER side (active coords) |
|---|---|---|
| A | 1D RoPE | `t` only MonSTER |
| B | 1D RoPE | `x` only MonSTER |
| C | 2D Axial RoPE | `x,y` only MonSTER |
| D | 2D Axial RoPE | `t,x` only MonSTER |
| E | 3D Axial RoPE | `t,x,y` only MonSTER |
| F | 3D Axial RoPE | `x,y,z` only MonSTER |
| G | 4D Axial RoPE | `t,x,y,z` MonSTER |

## 4) How to compare unlike sub-blocks fairly

Problem: RoPE uses 2D pairs; axial RoPE uses \(2n\)-dim axis-factorized structure; MonSTER uses fixed 4D blocks.

Use two fairness tracks:

### Track 1: Equal total embedding width (architectural fairness)
- Fix total dimension `D` for both models.
- RoPE block count = `D/2`; MonSTER block count = `D/4`.
- Compare induced kernels using **dimension-normalized trace similarity** (already the right style):  
  \[
  K(\Delta)=\mathbb{E}\left[\mathrm{trace}(T_p^\top T_q)/\text{block\_dim}\right]
  \]
This removes direct bias from block size.

### Track 2: Equal spectral degrees of freedom (capacity fairness)
- Match effective frequency budget across families (e.g., untied MonSTER boost/rotation frequencies vs RoPE pair frequencies).
- Report both tied and untied MonSTER frequency variants so we can separate “capacity gain” from “geometry gain”.

Both tracks should be reported; neither alone is sufficient.

## 5) Evaluation protocol

1. Use one shared position bank generator producing canonical `(t,x,y,z)` coordinates per window.  
2. For each pairing A–G, mask inactive coordinates to zero on both sides.  
3. Keep windows fixed (e.g., `64, 512, 1024, 2048`) and evaluate all pairings on identical windows.  
4. Run each pairing with matched frequency schedules (`theta_base`, exponent/warp settings logged explicitly).  
5. Report mean + worst-window values.

## 6) Metrics: not just “error to RoPE”

We need **two score families**:

### RoPE-likeness (baseline alignment)
- Curve RMSE to paired RoPE baseline
- Envelope RMSE
- Spectrum RMSE
- Long-context alias/tail error
- Effective-context distance error

### MonSTER-ness (what we do not want to lose)
- **Boost utilization**: non-trivial rapidity statistics (`|phi|` distribution, contribution vs rotation)
- **Minkowski invariance check** under \(-,+,+,+\)
- **Non-separability score**: how far the kernel is from axis-factorized product form
- **Causal anisotropy probe**: distinguish timelike vs spacelike displacements at matched norm

Final reporting should be a **Pareto view**: RoPE-likeness vs MonSTER-ness, not a single scalar that always pushes toward RoPE cloning.

## 7) Immediate practical recommendation

If we need one quick “clean” comparator first, start with:

- **2D Axial RoPE vs MonSTER(x,y only)**  

Reason: both naturally map to 4D structural blocks and avoid the most severe block-mismatch ambiguity.  
Then expand to A, D, E, F, G under the same protocol.

---

Bottom line: success is **not** “score goes to zero vs RoPE.”  
Success is: “close enough to RoPE on useful behaviors, while retaining distinctive Lorentz/Minkowski structure that could unlock better multidimensional reasoning.”

## 8) Required Properties for Any New Positional Encoding (Including MonSTER)

Any new positional/structural encoding should preserve the core properties that make RoPE useful.  
For MonSTER specifically, we need to preserve these while keeping real MonSTER structure (boosts, Minkowski metric, non-separable geometry).

### 8.1 Equivalence of Formulations

- Show that applying transforms at absolute positions yields the same relative-position behavior.
- For MonSTER, the absolute-vs-relative identity should hold under the chosen metric and transform family.

### 8.2 Norm/Metric Preservation

- Show that the transform preserves the relevant metric form.
- For MonSTER, this means preserving the Minkowski form under the active signature.

### 8.3 Shift Invariance

- Show that interactions depend on relative displacement, not absolute origin.
- For any global shift `d`, pairwise scores for `(m,n)` should match those for `(m+d,n+d)`.

### 8.4 Extensibility to Higher Dimensions

- Show that the method scales cleanly as dimensionality increases.
- For MonSTER, this means clean handling of active subsets (`t`, `x`, `y`, `z`) without ad hoc rewrites.

### 8.5 Multiscale Coverage and Remote Attenuation

- Show that geometric frequency schedules still provide broad scale coverage.
- Show that large separations naturally decorrelate in a controlled way (remote attenuation/locality bias).

## 9) RoPE Demo - Proof Checklist (Reference)

To show that a RoPE implementation works in all cases, you would need to demonstrate the following:

### 9.1 Equivalence of Formulations

- Show that applying separate rotations to absolute positions yields the same result as the complex relative position formula. For any 2D vectors `q` and `k` at positions `m` and `n`:

- `np.dot(apply_rotation_2d(q, m), apply_rotation_2d(k, n))`
- must equal `complex_dot_product_2d(q, k, m-n)`.

### 9.2 Norm Preservation

- Show that the rotation doesn't change the vector's length.
- The magnitude of a vector should remain constant after applying the positional encoding.

- `np.linalg.norm(q)` must equal `np.linalg.norm(apply_rotation_2d(q, m))`.

### 9.3 Shift Invariance

- Show that the dot product is the same for any pair of tokens that have the same relative distance, regardless of where they are in the sequence.
- For any shift `d`:
- `np.dot(apply_rotation_2d(q, m), apply_rotation_2d(k, n))` must equal
- `np.dot(apply_rotation_2d(q, m+d), apply_rotation_2d(k, n+d))`.

### 9.4 Extensibility to N-Dimensional Vectors

- Show that by applying rotations blockwise to pairs of dimensions RoPE naturally extends to higher-dimensional embeddings, interpreting the embedding vector as complex pairs:

- For an embedding vector `q = (q_1, q_2, q_3, q_4, ..., q_d)`,
- treat it as complex-valued pairs:
- `q = (q_1 + i q_2, q_3 + i q_4, ..., q_{d-1} + i q_d) in C^(d/2)`.

### 9.5 Multiscale Coverage and Remote Attenuation

- Show that by using the same geometric frequency schedule as the original sinusoidal positional encodings, `theta_i = 10000^{-2i/d}` we ensure:
- Multiscale Coverage: Frequencies span multiple positional scales.
- Remote Attenuation: Increasing positional distances naturally attenuate similarity due to phase misalignment, enhancing the model's locality bias.
