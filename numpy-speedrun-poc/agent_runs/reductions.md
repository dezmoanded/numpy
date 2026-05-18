# reductions

## 1. Summary
Implemented 2D sum reductions under :mini-array.reductions. Only axis 0 or 1 is supported. Output dtype is :float64. Elements are coerced to double-float during accumulation; if T/NIL appear, they are treated as 1.0d0/0.0d0, respectively.

## 2. Files/docs inspected
- src/core/package.lisp — confirmed exports used by reductions (product, rank, etc.).
- agent_runs/central-interfaces-and-behavior.md — validated exports include sum-axis/sum_axis and 2D-only reduction scope.
- Repo structure under src/ — ensured reductions depends only on core and tensor per DAG.

## 3. Key architectural ideas
- Reductions are independent of broadcasting/ufunc; depend only on core (shape utils) and tensor (array creation/indexing).
- Iterate row-major over 2D inputs; accumulate per output position.
- Always return contiguous outputs with default strides and dtype :float64.

## 4. Minimal behavior to port
- Input must be 2D ndarray (or array-like convertible via tensor:asarray).
- axis in {0,1}:
  - axis=0 => sum down rows, output shape (c)
  - axis=1 => sum across columns, output shape (r)
- Coercion during accumulation: number -> double-float; T -> 1.0d0; NIL -> 0.0d0.
- Errors on any other rank or axis values.

Pseudocode
- Validate rank=2 and axis in {0,1}
- out = make-array-from-flat(out-shape, :float64, zeroed-vector)
- if axis=0:
  - for j=0..c-1: s=0; for i=0..r-1: s+=coerce(aref-nd(a, (i j))); out[j]=s
- else axis=1:
  - for i=0..r-1: s=0; for j=0..c-1: s+=coerce(aref-nd(a, (i j))); out[i]=s

## 5. Implementation handoff
- Package: :mini-array.reductions; exports sum-axis and sum_axis (alias).
- Functions:
  - sum-axis(a, axis) — main API
  - sum_axis(a, axis) — alias to sum-axis
- Dependencies: :mini-array.core (rank, product), :mini-array.tensor (asarray, aref-nd, make-array-from-flat, shape-of, to-list for tests).
- Error cases: non-2D input; axis not 0/1; unsupported element types during accumulation.
- Simplifications: 2D only; :float64 outputs; no keepdims; no zero-sized dims; contiguous only.

## 6. Explicit exclusions
- Reductions for ranks other than 2D
- keepdims, multi-axis, dtype-preserving outputs
- NaN/signed-zero special cases
- Non-contiguous/view semantics

## 7. Suggested tests
- Numeric 2x3 matrix:
  - axis 0 -> (5.0d0 7.0d0 9.0d0)
  - axis 1 -> (6.0d0 15.0d0)
- Errors: 1D input; axis=2

Note: 2D boolean array construction via tensor:asarray is not required in V1 per tensor spec; boolean coercion is implemented defensively if booleans are encountered.

## 8. Open questions
- Confirm tensor:make-array-from-flat signature (shape dtype data) and that :float64 is accepted.
- Confirm tensor:to-list returns a flat list for 1D results.

Status
- Saved: src/reductions/sum.lisp, src/reductions/CONTRACT.md, src/reductions/tests.lisp.
- Tests depend on missing upstream modules (core/dtype.lisp and tensor/*); will run once they land.
