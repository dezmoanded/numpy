Module: reductions

Responsibilities
- Provide 2D axis reductions (sum) for the mini array runtime.
- Public API: sum-axis (primary), sum_axis (temporary alias).

API
- (sum-axis a axis) -> ndarray
  - a: ndarray or array-like convertible via tensor:asarray
  - axis: integer, 0 or 1
  - Returns a new contiguous ndarray with dtype :float64 and shape:
    - axis=0 -> (c)
    - axis=1 -> (r)
- (sum_axis a axis) -> same as sum-axis (alias kept temporarily)

Dependencies
- :mini-array.core for rank/product utilities
- :mini-array.tensor for ndarray accessors and constructors

Invariants/Assumptions
- Global rank support: only 0D/1D/2D; here we accept only 2D.
- Row-major contiguous, positive strides only.
- Zero-sized dims are globally disallowed.
- Input dtype may be :float64 or :bool; output dtype is always :float64.
  - Coercion: numbers -> double-float; T->1.0d0; NIL->0.0d0.

Explicit exclusions
- keepdims, multi-axis reductions, 0D/1D inputs.
- NaN-special semantics; dtype-preserving sum; integer/other dtypes.
- Non-contiguous views; negative strides.

Errors
- Non-2D input -> error
- Axis not in {0,1} -> error
- Invalid element types not convertible to a numeric sum -> error

Notes
- Implementation is independent of broadcasting and ufunc modules.
- Alias sum_axis will be removed after tests migrate.
