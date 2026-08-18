# IsoNCA LPPN Decoder Comparison

## Research question

Can a Cells2Pixels LPPN be used with an isotropic NCA without reintroducing a
preferred orientation?

All experiments use the same `lap_gradnorm` IsoNCA backbone and training
recipe. The controlled variable is the LPPN decoder:

```text
interpolated_only             interpolated state only
cartesian_coords              interpolated state + local x/y encoding
isotropic_neighbors           unordered four-cell neighborhood
isotropic_neighbors_radial    unordered neighborhood + radial distance
```

## Existing methods versus new methods

| Decoder | Role | Description |
| --- | --- | --- |
| `cartesian_coords` | Existing Cells2Pixels baseline | Normal LPPN using the interpolated cell state and local Cartesian subcell coordinates. |
| `interpolated_only` | Simple coordinate-free baseline | Earlier ablation that removes the Cartesian coordinates and decodes only the interpolated state. |
| `isotropic_neighbors` | **New proposed decoder** | Adds an unordered learned summary of the four surrounding cell states without directional coordinates. |
| `isotropic_neighbors_radial` | **New proposed ablation** | Adds directionless radial distance to the neighborhood messages. |

The main research comparison is:

```text
existing Cartesian LPPN versus new isotropic-neighborhood LPPN
```

`interpolated_only` tests whether the new neighborhood information improves
over merely deleting coordinates. The radial variant tests whether invariant
distance information improves the new decoder.

## Primary metric: decoder-only exact D4 equivariance

This metric isolates the LPPN from NCA dynamics. For a non-symmetric state
field `z` and exact square-grid transform `T`, measure:

```text
relative_L2 = ||D(T(z)) - T(D(z))||_2 / ||T(D(z))||_2
```

Transforms are 90-, 180-, and 270-degree rotations plus horizontal and
vertical reflections.

| Decoder | Exact D4 relative L2 range |
| --- | ---: |
| `interpolated_only` | 1.83e-7 to 2.07e-7 |
| `cartesian_coords` | 2.77e-2 to 4.02e-2 |
| `isotropic_neighbors` | 2.55e-7 to 2.81e-7 |
| `isotropic_neighbors_radial` | 2.10e-7 to 2.30e-7 |

The coordinate-free neighborhood decoders preserve exact D4 equivariance to
floating-point precision. The Cartesian decoder does not.

The neighborhood-order permutation relative L2 errors are:

```text
isotropic_neighbors:        1.39e-7
isotropic_neighbors_radial: 1.18e-7
```

## Secondary metric: paired-mask end-to-end equivariance

This evaluates the complete stochastic system. The transformed rollout uses
the spatially transformed version of the same update-mask sequence:

```text
D(N(T(seed), T(masks))) versus T(D(N(seed, masks)))
```

| Decoder | Render relative L2 range |
| --- | ---: |
| `interpolated_only` | 3.46e-6 to 7.32e-6 |
| `cartesian_coords` | 8.16e-3 to 1.16e-2 |
| `isotropic_neighbors` | 7.70e-7 to 9.06e-7 |
| `isotropic_neighbors_radial` | 1.80e-6 to 3.41e-6 |

The NCA state error remains near numerical precision for the Cartesian run,
while its render error is much larger. This identifies the Cartesian decoder,
not the IsoNCA dynamics, as the source of the orientation bias.

## Fit and visual quality

Training uses the rotation/reflection-invariant polar FFT target loss. Because
individual batches and rollout trajectories are noisy, the best loss and the
median over the last 1,000 steps are more informative than the final batch.

| Decoder | Best target loss | Last-1k median target loss |
| --- | ---: | ---: |
| `interpolated_only` | 31.01 | 47.02 |
| `cartesian_coords` | 30.83 | 46.27 |
| `isotropic_neighbors` | **29.03** | **44.11** |
| `isotropic_neighbors_radial` | 30.93 | 47.72 |

The strict `isotropic_neighbors` decoder has the best loss statistics in these
runs. The Cartesian decoder does not produce a visible sharpness advantage.
All four outputs remain relatively soft, and neither neighborhood decoder has
yet recovered the sharp high-resolution detail targeted by the broader
research objective.

## Current conclusion

The experiment is successful in the narrower sense:

- A coordinate-free neighborhood LPPN trains jointly with `lap_gradnorm`
  IsoNCA.
- It preserves exact square-grid rotation/reflection equivariance.
- It performs at least as well as the ordinary Cartesian LPPN in this run.
- The Cartesian LPPN introduces measurable orientation bias without a visible
  quality benefit.

The broader high-resolution objective is not yet met: the isotropic decoder is
still visually blurry and is not materially sharper than `interpolated_only`.

## Limitations

- There is one training run per decoder, all using training seed 43.
- The decoder-only D4 table currently uses one random non-symmetric probe field.
- Cross-decoder evaluation should be repeated with several common stochastic
  mask sequences and summarized with mean, standard deviation, and maximum.
- Arbitrary-angle rotations are excluded from the headline metric because
  square-grid resampling confounds them.
- The results establish D4 equivariance, not exact continuous SO(2)
  equivariance.

## Code

Executed notebooks:

```text
notebooks-executed/weights/isonca_lizard_lppn_no_coords/isonca_lizard_lppn_gradnorm.ipynb
notebooks-executed/weights/isonca_lizard_lppn_coords/isonca_isotropic_lppn_experiments.ipynb
notebooks-executed/weights/isonca_lizard_lppn_isotropic_neighbors/isonca_isotropic_lppn_experiments.ipynb
notebooks-executed/weights/isonca_lizard_isotropic_neighbors_radial/isonca_isotropic_lppn_experiments.ipynb
```