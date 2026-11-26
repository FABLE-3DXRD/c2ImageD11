# AoS vs SoA data layout

Some c2ImageD11 functions accept their trailing `(n, 3)`-style matrix
arguments in either of two memory layouts. The C code detects which layout
was passed and selects the right kernel, so no `.T.copy()` or manual
transpose is needed.

## The two layouts

- **AoS (array of structures)** — shape `(n, 3)`. Each *row* is one point:
  the three components of a single point live in adjacent memory.
- **SoA (structure of arrays)** — shape `(3, n)`. Each *column* is one point:
  the components are separated, one array of x-values, one of y-values, one
  of z-values.

Either layout describes the same point cloud; only the memory order differs.

## Functions that accept either layout

A function that accepts a transposed (SoA) layout for a given argument is
annotated with a `layout` note on its reference page. The argument(s) that
may be transposed are named there, e.g.:

- `score` / `score_and_refine` — `gv` accepts `(ng, 3)` AoS or `(3, ng)` SoA.
- `compute_geometry` / `compute_gv` — `xlylzl` and the vector output accept
  `(n, 3)` AoS or `(3, n)` SoA.

## Split SoA variants

A few functions (`compute_geometry_soa`, `compute_gv_soa`) take the
components as *separate* arrays instead of one matrix: separate `xl`, `yl`,
`zl` inputs (and, for the outputs, `tth`, `eta`, `ds`, `gx`, `gy`, `gz`).
These are provided for callers that already keep the components split; they
are equivalent to passing an SoA matrix.

## Why this matters

These are the hot loops used by ImageD11's indexing and refinement. The
dispatch avoids allocating a transposed copy in Python, keeping the C
kernels operating directly on the caller's buffer.
