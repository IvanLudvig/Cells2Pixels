# LPPN Decoder Notes

## Basic picture

The NCA is a coarse grid whose nodes are cell centers. Each node holds a
persistent 16D state:

```text
NCA state: 72 x 72 x 16
```

The LPPN samples a finer grid with 8 x 8 output pixels per NCA cell:

```text
Rendered output: 576 x 576 x 5
Output channels: RGBA + binary auxiliary
```

Subcells are only fine-grid query positions. They do not have persistent
states.

For every fine-grid position `p`, the four surrounding cell states are
bilinearly interpolated:

```text
z_bar(p) = sum_j w_j(p) z_j
```

The four weights depend on the query position and sum to one. The LPPN converts
the resulting local information into one output pixel.

## Decoder approaches

### 1. Interpolated state only

```text
input:  interpolated 16D state z_bar(p)
output: RGBA + auxiliary

y(p) = rho(z_bar(p))
```

- Does not expose horizontal or vertical directions to the learned decoder.
- Compatible with square-grid rotation/reflection symmetry.
- Has limited information for producing detailed subcell patterns.
- This is the completed `no_coords` / `interpolated_only` baseline.

### 2. Cartesian subcell coordinates

```text
input:  z_bar(p) + local x/y coordinate encoding
output: RGBA + auxiliary

y(p) = rho(z_bar(p), x_local(p), y_local(p))
```

- Can distinguish all positions inside an 8 x 8 cell primitive.
- Usually provides sharper and more detailed output.
- Gives every cell the same horizontal and vertical axes.
- A fixed decoder can therefore render a rotated state differently, breaking
  rotation/reflection equivariance.
- This is the `cartesian_coords` quality baseline.

### 3. Isotropic neighborhood LPPN

For each fine-grid position, retain the four individual surrounding cell
states in addition to their interpolation:

```text
m_j(p) = phi(z_j, z_j - z_bar(p))
m(p)   = mean_j m_j(p)
y(p)   = rho(z_bar(p), m(p))
```

- `phi` is shared by all four neighbors.
- Mean aggregation discards neighbor order.
- No neighbor is labeled left, right, above, or below.
- A rotation or reflection only permutes the messages, leaving their aggregate
  unchanged.
- Provides more local information than `interpolated_only` without supplying
  directional coordinates.
- This is the proposed strict `isotropic_neighbors` decoder.

If all four surrounding cell states are identical, this strict decoder cannot
distinguish the subcell positions. That is part of its symmetry constraint.

### 4. Isotropic neighborhood with radial distance

Add the distance from the fine-grid query to each surrounding cell center:

```text
r_j(p) = ||p - p_j||
m_j(p) = phi(z_j, z_j - z_bar(p), r_j(p))
```

- Distance indicates proximity without specifying a direction.
- Can distinguish more subcell configurations than the strict variant.
- Still does not expose separate horizontal and vertical coordinates.
- This is the `isotropic_neighbors_radial` ablation, not the primary run.

## Intended comparison

```text
interpolated_only    symmetric but visually soft
cartesian_coords     detailed but potentially direction-biased
isotropic_neighbors intended to be both symmetric and more detailed
```

The symmetry guarantee is exact for square-lattice rotations and reflections
(the D4 group). Arbitrary-angle rotations remain approximate because they
require interpolation on a square grid.
