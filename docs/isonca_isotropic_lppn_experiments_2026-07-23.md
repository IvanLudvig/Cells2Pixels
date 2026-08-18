# IsoNCA and Isotropic LPPN Experiments

Date: 2026-07-23

## Research Objective

Design a high-resolution Local Pattern Producing Network (LPPN) suitable for
isotropic self-organizing systems.

The immediate test system is the successful single-seed `lap_gradnorm` IsoNCA.
The eventual scope also includes reaction-diffusion models.

The research contribution is the decoder, not a new IsoNCA training method. The
goal is to retain the high-resolution detail provided by a coordinate-conditioned
Cells2Pixels LPPN without giving the system a preferred global direction.

For a scalar state field `z`, decoder `D`, and rotation or reflection `T`, the
desired property is:

```text
D(T(z)) ~= T(D(z))
```

The same fixed decoder is used on both sides. The decoder parameters are not
rotated.

The main hypothesis is that a local, permutation-invariant neighborhood decoder
can approach the visual quality of a Cartesian-coordinate LPPN while preserving
the rotation/reflection equivariance of a coordinate-free decoder.

This follows the future-work direction in the Cells2Pixels paper:

> alternative formulations better suited to isotropic settings ... such as
> coordinate-free LPPN variants that avoid explicit intra-primitive coordinates
> and instead condition on the interpolated cell state together with
> nearest-neighbor cell states

References:

- Cells2Pixels paper: <https://arxiv.org/html/2506.22899v3>
- IsoNCA article: <https://google-research.github.io/self-organising-systems/isonca/>

## Controlled IsoNCA Backbone

Use the already established IsoNCA route as a controlled backbone:

```text
model type: lap_gradnorm
state channels: 16
hidden width: 128
update probability: 0.5
initial condition: one living center cell
loss: rotation/reflection-invariant polar FFT loss
auxiliary target: binary
```

Its perception consists of the current state, an isotropic Laplacian, and the
magnitude of the spatial gradient:

```text
perception = [state, laplacian(state), sqrt(gx^2 + gy^2)]
```

The gradient magnitude does not expose the separate global `x` and `y`
directions to the update MLP.

The successful direct-output reference is:

```text
notebooks-executed/blogpost_isonca_single_seed_pytorch.ipynb
final loss: 9.1643 at 50,000 steps
checkpoint: lap_gradnorm_lizard_binary_0050000.pt
```

That run grows recognizable lizards in arbitrary rotations/reflections and is
the reference for the NCA architecture and training semantics.

## Experiments So Far

### 1. Earlier Laplacian and structured-seed route

The earlier route used:

```text
MODEL_TYPE = LAPLACIAN
LOSS_FN = FIXED
SEED_N = 2
```

This failed in both the project notebooks and a patched version of the original
structured-seed notebook. It produced blurry symmetric blobs rather than a
stable lizard.

Conclusion:

- Do not use this route as the baseline.
- The confirmed single-seed `lap_gradnorm` route must remain the controlled
  backbone.

### 2. Initial LPPN port

The first LPPN port allowed the decoder to learn while the NCA received
effectively zero useful gradient. The visible symptom was a nearly constant
green render. The explicit guard reported:

```text
RuntimeError: no_coords NCA gradient stayed near zero for 10 consecutive steps
```

Bootstrap stages, temporary decoder freezing, and related attempts did not
resolve the underlying issue reliably.

The useful diagnosis was that the project already had a working growing-LPPN
route in `training/tasks/growing_2d.py`. The port needed to preserve that route
instead of treating the decoder as an independent image fitter.

### 3. Corrected joint-training route

The following changes restored useful NCA learning:

- Train IsoNCA and LPPN jointly from step zero.
- Gate the rendered image with the upsampled NCA living mask.
- Supervise NCA alpha directly.
- Put rendered RGBA, the binary auxiliary channel, and NCA alpha into the same
  invariant-loss alignment decision.
- Keep the successful single-seed pool schedule.
- Use random rollout lengths from 64 to 96 steps.
- Keep asynchronous updates with probability `0.5`.
- Remove the zero-gradient abort once the route was corrected.
- Keep float32 because the FFT loss and long unroll overflowed under AMP/fp16 on
  a Colab T4.

The aligned alpha point is important. Alpha cannot be compared at a default
orientation while RGBA is compared at the best rotation/reflection. All
supervised channels must use the same selected target transform.

### 4. Completed 50k no-coordinate LPPN run

Artifacts:

```text
notebooks-executed/50k-run/isonca_lizard_lppn_gradnorm.ipynb
notebooks-executed/50k-run/latest.pt
notebooks-executed/50k-run/model.pth
notebooks-executed/50k-run/siren.pth
notebooks-executed/50k-run/run_meta.pth
```

Important configuration:

```text
target: data/morphology_png/lizard.png
target size: 512 x 512, plus 32-pixel transparent padding per side
NCA grid: 72 x 72
render scale: 8
IsoNCA channels: 16
IsoNCA hidden width: 128
LPPN hidden width: 96
LPPN hidden layers: 2
decoder input: interpolated NCA state, no local coordinates
batch size: 2
pool size: 256
training steps: 50,000
rollout range: 64 to 96
damage training: disabled
checkpoint interval: 5,000
hardware: Tesla T4
precision: float32
```

Final logged values:

```text
loss: 35.9307
target loss: 35.8563
overflow loss: 0.00516
state difference: 0.00228
NCA gradient norm: 670.73
LPPN gradient norm: 111.99
learning rate: 1.024e-5
final rollout length: 95
```

Observed result:

- The run completed successfully and saved all expected checkpoints.
- The output is a clearly recognizable lizard with a complete body, head, tail,
  limbs, and spots.
- The binary auxiliary output has the intended two-sided division.
- Both NCA and LPPN gradients remain healthy at the end of training.
- The loss falls sharply early and then enters a broad, noisy plateau.
- The result is visibly softer and less detailed than the direct blogpost
  output.
- Extending this exact run beyond 50k is unlikely to address the main quality
  limitation.

Interpretation:

- The Cells2Pixels training route works with the `lap_gradnorm` IsoNCA.
- An interpolated-state-only decoder is an isotropic-compatible baseline.
- Removing coordinates alone sacrifices the main source of high-resolution
  detail in the original LPPN.
- This quality gap motivates an expressive isotropic LPPN rather than more
  tuning of this baseline.

## Current Equivariance Results and Caveat

The completed notebook reports very low errors for exact square-grid
transformations:

| Transform | State relative L2 | Render relative L2 |
| --- | ---: | ---: |
| 90-degree rotation | 0.002755 | 0.002224 |
| 180-degree rotation | 0.002124 | 0.001692 |
| 270-degree rotation | 0.002755 | 0.002224 |
| horizontal reflection | 0.002564 | 0.002108 |
| vertical reflection | 0.002155 | 0.001872 |

Interpolated arbitrary-angle results are poor:

| Transform | State relative L2 | Render relative L2 |
| --- | ---: | ---: |
| 30-degree rotation | 1.068 | 0.911 |
| 45-degree rotation | 1.002 | 1.000 |
| 60-degree rotation | 1.068 | 0.911 |

These numbers must not currently be used as evidence for or against the proposed
LPPN.

The evaluation sets the NCA update probability to `1.0`. Starting from a
rotationally symmetric single-cell seed, the deterministic rollout remains
highly symmetric and renders a radial disk instead of a lizard. Exact
90-degree/reflection errors are consequently easy to make small, but they do not
measure equivariance of the learned lizard morphology.

Arbitrary-angle results are additionally confounded by interpolation of the
single-cell seed and the square lattice.

The current evaluation tests the entire rollout:

```text
D(N(T(seed))) ~= T(D(N(seed)))
```

That is a useful end-to-end property, but it conflates NCA dynamics, stochastic
update masks, grid resampling, and decoder behavior. It does not isolate the
LPPN.

## What Has Been Demonstrated

- The direct-output single-seed `lap_gradnorm` IsoNCA baseline works.
- The same IsoNCA route can be trained jointly with the project LPPN renderer.
- Living-mask gating and aligned NCA-alpha supervision are required for useful
  automaton gradients in this experiment.
- The 50k no-coordinate LPPN run is a valid trainable baseline.
- The no-coordinate baseline grows a recognizable lizard but loses substantial
  high-resolution sharpness.
- Checkpointing and resume support work at 5,000-step intervals.

## What Has Not Been Demonstrated

- The current end-to-end metric does not yet demonstrate lizard-morphology
  equivariance.
- The ordinary Cartesian-coordinate LPPN has not yet been measured as a matched
  quality/anisotropy baseline.
- No proposed isotropic neighborhood LPPN has been implemented or trained.
- We have not shown that an isotropic LPPN improves detail over the no-coordinate
  baseline.
- We have not shown resolution generalization, regeneration, long-rollout
  stability, or reaction-diffusion compatibility for the proposed decoder.
- One successful training run is not enough for statistical claims.

## Decoder Variants for the Main Study

### A. Interpolated-state-only baseline

This is the completed `no_coords` run:

```text
y(p) = rho(z_bar(p))
```

Here `z_bar(p)` is the bilinearly interpolated NCA state at query point `p`.

Expected behavior:

- Strong exact square-grid rotation/reflection compatibility.
- Limited ability to add high-frequency subcell detail.
- Blurry output.

### B. Original Cartesian LPPN baseline

This is the existing Cells2Pixels decoder:

```text
y(p) = rho(z_bar(p), x_local(p), y_local(p))
```

Expected behavior:

- Better detail and sharper output.
- A generic MLP can distinguish the local horizontal and vertical axes.
- Rotation/reflection equivariance is not guaranteed.

This variant is necessary as the quality baseline even though it is not the
proposed solution.

### C. Proposed isotropic neighborhood LPPN

For each high-resolution query point `p`:

1. Compute the interpolated state `z_bar(p)`.
2. Gather the states `z_j` of cells in a local neighborhood.
3. Process each neighbor using shared weights.
4. Aggregate messages without orientation-dependent neighbor ordering.
5. Decode the interpolated state and aggregate into output channels.

One candidate formulation is:

```text
m_j(p) = phi(z_j, z_j - z_bar(p), distance(p, p_j))
m(p)   = sum_j m_j(p)
y(p)   = rho(z_bar(p), m(p))
```

Properties:

- `phi` is shared across all neighbors.
- Sum or mean aggregation is invariant to neighbor ordering.
- Distance is a rotation/reflection-invariant scalar.
- No neighbor is labeled north, south, east, or west.
- The local state configuration supplies context that the current
  interpolated-state-only decoder lacks.

The strict coordinate-free version should also be tested:

```text
m_j(p) = phi(z_j, z_j - z_bar(p))
```

This most directly follows the Cells2Pixels future-work paragraph. Adding radial
distance is a controlled extension if the strict version is not expressive
enough.

Simply concatenating the four neighbors in a fixed grid order is not acceptable:
that would reintroduce preferred directions.

## Experimental Plan

### Phase 1: Define and test the decoder symmetry contract

Add a decoder-only evaluation independent of NCA rollout:

```text
D(T(z)) versus T(D(z))
```

Use arbitrary non-symmetric state fields as inputs so that a radial attractor
cannot make the test trivial.

Required tests:

- 90-, 180-, and 270-degree rotations.
- Horizontal and vertical reflections.
- Random permutation of neighbor order for the proposed set decoder.
- 30-, 45-, and 60-degree rotations as approximate, interpolation-confounded
  diagnostics.

Exact D4 transforms are the primary architectural pass/fail test. Arbitrary
angles should be reported separately.

### Phase 2: Establish matched decoder baselines

Run the original Cartesian-coordinate decoder using the same lizard setup as the
completed no-coordinate run.

Keep fixed:

- IsoNCA architecture and update semantics.
- Single-cell initial condition.
- Target and output resolution.
- Invariant loss and binary auxiliary target.
- Alpha supervision and living-mask gating.
- Pool, rollout, optimizer, and checkpoint schedules.
- Model capacity as closely as practical.

Train NCA and decoder jointly from scratch because they co-adapt. "Same IsoNCA"
means the same architecture and training recipe, not necessarily the same frozen
weights.

### Phase 3: Implement the isotropic neighborhood LPPN

Implement the decoder as a reusable model, not notebook-only code.

Start with:

- Four enclosing primitive cells or a small fixed-radius neighborhood.
- Shared neighbor encoder.
- Sum/mean aggregation.
- Interpolated state as the query-specific input.
- No raw Cartesian coordinates.

Then ablate:

- Strict coordinate-free messages.
- Radial-distance messages.
- Four-cell primitive versus a larger neighborhood.
- Sum versus mean aggregation.

Avoid adding attention or larger neighborhoods until the minimal version is
measured; they add compute and can obscure the architectural result.

### Phase 4: Controlled lizard comparison

Train:

```text
interpolated_only
cartesian_coords
isotropic_neighbors
```

Use the same initial training random seed for the first engineering comparison.
For final claims, repeat with multiple training seeds.

Measure fit:

- Invariant target loss.
- Aligned RGBA reconstruction metrics.
- Perceptual/sharpness metric or a clearly specified edge/high-frequency metric.
- Visual quality at the training resolution.
- Rendering at higher and lower resolutions without retraining.
- Primitive-boundary artifacts.

Measure symmetry:

- Decoder-only D4 equivariance on arbitrary state fields.
- Decoder-only approximate arbitrary-angle equivariance.
- End-to-end stochastic IsoNCA equivariance.
- Distribution of output orientations/reflections across multiple rollouts.

The desired result is not merely the lowest reconstruction loss. The proposed
decoder should improve quality over `interpolated_only` while retaining its
symmetry behavior, and should improve symmetry over `cartesian_coords` while
approaching its quality.

### Phase 5: Correct end-to-end IsoNCA evaluation

Keep asynchronous updates at probability `0.5`.

For pathwise comparisons, provide the NCA with explicit update masks and apply
the same spatial transform to each mask in the transformed rollout:

```text
D(N(T(seed), T(mask_sequence)))
    versus
T(D(N(seed, mask_sequence)))
```

This preserves stochastic symmetry breaking while making the two trajectories
comparable.

Also evaluate multiple independent stochastic rollouts. Isotropy does not mean
that every run must remain radial; it means the system has no preferred external
orientation and transformed conditions produce correspondingly transformed
behavior.

### Phase 6: Robustness and generalization

After the main lizard comparison works:

- Evaluate long rollouts beyond the 64-96 training horizon.
- Evaluate regeneration after damage.
- Test render resolutions not seen during training.
- Test a second morphology target.
- Apply the same decoder to an isotropic reaction-diffusion model.

Reaction-diffusion is a generalization experiment for the decoder, not a
prerequisite for establishing the initial result.

## Success Criteria

The first-stage objective is met when:

1. The proposed LPPN is rotation/reflection equivariant by construction or by a
   clearly justified local symmetry argument.
2. Decoder-only exact D4 tests confirm that property on non-symmetric state
   fields.
3. It produces a materially sharper, more detailed lizard than the current
   interpolated-state-only baseline.
4. Its equivariance error is substantially lower than the Cartesian-coordinate
   LPPN under matched evaluation.
5. The result holds across multiple training runs rather than one favorable
   checkpoint.

The broader paper direction is supported when the same decoder also improves
high-resolution output for another isotropic system, preferably a
reaction-diffusion model.

## Immediate Next Actions

1. Preserve the completed 50k `no_coords` artifacts as baseline A.
2. Add decoder-only D4/reflection tests using arbitrary state fields.
3. Run the matched Cartesian-coordinate lizard baseline B.
4. Implement the minimal permutation-invariant neighborhood decoder C.
5. Smoke-test C, then train it with the established 50k lizard route.
6. Compare A, B, and C using fit, detail, boundary-artifact, and equivariance
   metrics.

Do not return to the failed `LAPLACIAN + FIXED + structured seed` route, and do
not move to the chameleon before the decoder comparison is established.

## Prepared Next Run (2026-07-24)

The matched Colab experiment is prepared in:

```text
notebooks/isonca_isotropic_lppn_experiments.ipynb
```

Reusable implementation and tests are in:

```text
models/isotropic_lppn.py
tests/test_isotropic_lppn.py
```

The notebook trains one decoder variant per run while preserving the established
50k `lap_gradnorm` training route. Run `cartesian_coords` first, then
`isotropic_neighbors`. `isotropic_neighbors_radial` is the next controlled
ablation; do not substitute it for the strict coordinate-free primary run.

The notebook performs two distinct symmetry evaluations:

- Decoder-only D4/reflection tests on arbitrary non-symmetric state fields,
  including a random neighbor-order permutation check.
- End-to-end pathwise tests with update probability `0.5`, using the same
  stochastic mask sequence transformed along with the seed.

The latter replaces the earlier deterministic `update_prob=1.0` radial-disk
diagnostic. No experimental result is claimed until the new Colab runs finish.
