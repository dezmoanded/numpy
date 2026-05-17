# central-interfaces-and-behavior

## 0. Clarifications (updated)

- Canonical file layout now:
  - src/mini-array.lisp = thin loader + core utils/ndarray/broadcasting (temporarily houses core/tensor until split)
  - src/ops.lisp = add/mul/binary-ufunc (signature: (a b op))
  - src/reductions.lisp = sum-axis (2D only; axis in {0,1})
  - src/package.lisp = exports (add, mul, binary-ufunc, sum-axis, sum_axis, broadcast-shape, etc.)
- Naming note: This plan previously said reduce.lisp; the actual file in repo is reductions.lisp. Prefer reductions.lisp. If a reduce.lisp appears during split, update the loader accordingly.
- Public helpers added: product and rank are available and exported for tests.
- Temporary alias: sum_axis is exported and delegates to sum-axis. Keep until subgroup tests are green, then remove.
- Old modules under src/old/* are deprecated. Do not :load or :use them.
- Current status: SBCL test tests/test-mini-array.lisp passes end-to-end after loader/export/reduction fixes.

## 1. Summary

We will reduce surface area by grouping the V1 implementation into four sub‑packages with focused, narrow tests. This note defines the interfaces and responsibilities for each group, the loader/package expectations, and the exact tests to write. Later agents should implement each group and its tests without re‑crawling.

Target groups (source files):
- core.lisp = utils + shape/strides
- tensor.lisp = ndarray struct + asarray/to-list + indexing helpers
- ops.lisp = broadcasting + elementwise ufuncs (add/mul)
- reduce(s)/reductions.lisp = reductions (sum-axis)
- mini-array.lisp (loader only) + package.lisp (exports)

Constraints: V1 only supports float64, 0D/1D/2D, row-major contiguous, positive strides, broadcasting, add/mul, and sum along a 2D axis.

## 2. Files/docs inspected

- agent_runs/00-final-architecture-port-plan.md — scope, invariants, and exclusions
- agent_runs/01-core-array-implementation.md — ndarray metadata, contiguous layout
- agent_runs/02-broadcasting-implementation.md — broadcasting/right‑aligned rules
- agent_runs/03-elementwise-ufunc-implementation.md — ufunc iteration over broadcast shape
- agent_runs/04-reduction-implementation.md — axis semantics and exclusions
- tests/test-mini-array.lisp — current test cases to split/narrow

## 3. Key architectural ideas

- Keep ndarray metadata minimal: shape, strides (element-based), dtype (fixed :float64), data (contiguous double-float vector).
- Default strides are row-major: NIL (0D), (1) for 1D, (c 1) for 2D.
- Broadcasting is right-aligned per NumPy: dimensions must match or one must be 1; otherwise raise error.
- Elementwise ops iterate over the broadcasted output shape and project indices back to inputs.
- Reductions V1: only 2D axis 0 or 1; errors for 0D/1D/axis out of range; return contiguous outputs.

## 4. Minimal behavior to port

Group: core.lisp
- product(xs) -> integer (product of dims); product(NIL) -> 1
- rank(shape) -> length
- valid-shape-p(shape) -> rank<=2 and every dim>0 (no zero-sized)
- %ensure-double(x) -> double-float
- %make-double-vector(n, optional init=0.0d0) -> simple-array double-float of length n filled with init
- default-strides(shape):
  - NIL -> NIL; (n) -> (1); (r c) -> (c 1); else error rank>2

Group: tensor.lisp
- defstruct ndarray: shape, strides, dtype(:float64), data(simple-array double-float)
- asarray(obj):
  - number -> 0D: shape NIL, strides NIL
  - 1D list of numbers -> shape (n)
  - 2D rectangular list of numbers -> shape (r c)
  - errors: empty lists, ragged rows, non-numeric entries, rank>2
- to-list(a): number for 0D, flat list for 1D, list of lists for 2D
- flat-offset(index, strides, optional shape): bounds-check when shape given; 0D -> 0
- aref-nd(a, index) and %set-aref-nd(a, index, value): use flat-offset

Group: ops.lisp
- broadcast-shape(a, b): treat NIL as 0D; right-align; dimension rule max(di,dj) if equal or one is 1; else error
- all-indices(shape):
  - NIL -> (NIL)
  - (n) -> ((0) ... (n-1))
  - (r c) -> ((0 0) ... (r-1 c-1))
- project-broadcast-index(out-idx, in-shape, out-shape): right-align in-shape to out; if in-dim==1 use 0; else copy out-idx
- binary-ufunc(a, b, op): accepts ndarray or scalars; returns contiguous ndarray of broadcasted shape. Signature is (a b op).
- add(a, b), mul(a, b): wrappers for binary-ufunc

Group: reductions.lisp
- sum-axis(a, axis):
  - Only for 2D; axis in {0,1}; return (c) for axis=0 or (r) for axis=1
  - Errors: 0D/1D inputs; axis not 0/1; rank>2 (future-proof error)

Loader and package
- package.lisp must export: ndarray, asarray, to-list, product, rank, valid-shape-p, default-strides, flat-offset, aref-nd, add, mul, sum-axis, sum_axis, broadcast-shape, all-indices, project-broadcast-index, binary-ufunc.
- mini-array.lisp is a thin loader that loads ops.lisp and reductions.lisp now; as core/tensor split out, update loader to load core.lisp, tensor.lisp first, then ops and reductions in that order.

## 5. Implementation handoff

Data structures
- ndarray struct as above; no views or negative strides.

Functions
- Implement exactly the functions listed per group. Keep signatures stable for narrow tests and integration.

Algorithm outlines
- broadcasting: pad-left with 1s to match ranks; iterate dims; choose winner or error.
- ufunc: for each index in all-indices(out), compute projected indices for inputs, fetch via aref-nd, apply op, store in output via flat offset.
- sum-axis: nested loops over fixed axis while accumulating in contiguous output buffer.

Error cases
- Shapes with zero-sized dims; broadcasting incompatibility; reductions on unsupported ranks or axes; flat-offset bounds violations; ragged rows in 2D input.

Simplifications (V1)
- Only float64; only positive, row-major contiguous strides; ranks limited to 0/1/2; no slicing, no advanced indexing, no negative strides, no zero-sized dims.

## 6. Explicit exclusions
- Dtypes other than float64; object arrays; masked arrays; negative/zero strides; non-contiguous views; advanced indexing; slicing; axis reductions beyond 2D or along keepdims; NaN special handling; BLAS/LAPACK; SIMD.

## 7. Suggested tests

Create separate files under tests/ and run them individually before integration.

- tests/core-tests.lisp
  - (assert (= 1 (product nil)))
  - (assert (= 6 (product '(2 3))))
  - (assert (valid-shape-p '(2)))
  - (assert (not (valid-shape-p '(0))))
  - (assert (equal '(1) (default-strides '(5))))
  - (assert (equal '(3 1) (default-strides '(2 3))))

- tests/tensor-tests.lisp
  - (assert (eql :float64 (ndarray-dtype (asarray 7))))
  - (assert (null (ndarray-shape (asarray 7))))
  - (assert (equal '(3) (ndarray-shape (asarray '(1 2 3)))))
  - (assert (equal '((1 2) (3 4)) (to-list (asarray '((1 2) (3 4))))))
  - flat-offset bounds: expect error for (list 2) on shape '(2 2)

- tests/ops-tests.lisp
  - (assert (equal '(2 3) (broadcast-shape '(2 1) '(1 3))))
  - add with scalar: asarray '(1 2 3) + 10 -> '(11 12 13)
  - broadcast 1x3 + 2x1 -> 2x3 expected matrix
  - incompatible shapes should raise error

- tests/reduce-tests.lisp
  - sum-axis on '((1 2 3) (4 5 6)) axis 0 -> '(5 7 9)
  - sum-axis on same axis 1 -> '(6 15)
  - sum-axis on 1D should error; invalid axis should error

- tests/integration-tests.lisp
  - Compose: sum-axis(add(A, B), 1) with small 2x2 A,B

Approx-equality helpers
- For float comparisons, use abs tolerance 1e-6 where non-integer ops appear.

## 8. Open questions
- Should shape-of/strides-of remain public or be folded into tests via struct accessors? Default: keep public for convenience during V1; we can narrow later.
- Confirm package exports required by tests; keep internals private unless tests require them.
- Timeline to remove sum_axis alias: after all subgroup tests pass and have migrated to sum-axis.
