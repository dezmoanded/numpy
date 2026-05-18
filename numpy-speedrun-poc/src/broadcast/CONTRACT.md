# Broadcast module — CONTRACT

Scope and responsibility
- Provide NumPy-style right-aligned broadcasting helpers for ranks 0..2.
- No tensor data access; depends only on core shape utilities.

Package
- :mini-array.broadcast (nickname :ma.broadcast)
- Depends on: :mini-array.core (rank, valid-shape-p, product [product not currently used]).
- Exports: broadcast-shape, all-indices, project-broadcast-index

API
- broadcast-shape(a-shape, b-shape) -> shape
  - Inputs: SHAPEs are NIL (0D), (n), or (r c), positive integers only, rank<=2.
  - Rule per dimension (after left-padding smaller rank with 1s): equal wins; else if one is 1 -> other wins; else error.
  - NIL broadcasts to any shape; returns the other shape.
  - Errors for invalid shapes or rank>2.

- all-indices(shape) -> list of indices (row-major)
  - NIL -> (NIL)
  - (n) -> ((0) ... (n-1))
  - (r c) -> ((0 0) (0 1) ... (r-1 c-1))
  - Errors for invalid shapes or rank>2.

- project-broadcast-index(out-index, in-shape, out-shape) -> index-for-in
  - Validates shapes and that OUT-INDEX length == rank(OUT-SHAPE).
  - Ensures IN-SHAPE is broadcastable to OUT-SHAPE (broadcast-shape(in, out) == out).
  - Left-pads IN-SHAPE with 1s to OUT rank; for each dim: if in-dim==1 -> pick 0; else copy from OUT-INDEX; drop padding.
  - 0D returns NIL.

Invariants and simplifications (V1)
- Rank limited to 0..2.
- Shapes disallow zero-sized dimensions.
- Row‑major index order only.
- No views/slices; indices are full-rank.

Errors
- broadcast-shape: invalid shapes; incompatible shapes.
- all-indices: invalid shape; rank>2.
- project-broadcast-index: invalid shapes; OUT-INDEX rank mismatch; in-shape not broadcastable to out-shape.

Exclusions
- Higher-rank broadcasting (>2).
- Negative/zero strides; non-contiguous views; advanced indexing.
- Keepdims/axes—out of scope here.

Testing
- Module tests runnable via: sbcl --script src/broadcast/tests.lisp
- Load only minimal upstream files: core/package.lisp, core/shape.lisp; then this module’s package + impl files.
