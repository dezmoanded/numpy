# reductions

## Overview
The reductions module provides 2D axis-wise summation for ndarrays. Use it to compute column-wise (axis=0) or row-wise (axis=1) sums, producing a 1D contiguous :float64 ndarray. V1 supports only 2D inputs (no keepdims, no multi-axis reductions), row‑major contiguous layout, and positive element strides.

## Public exports
- sum-axis — Sum a 2D ndarray along a given axis (0 for columns, 1 for rows). Returns a 1D :float64 ndarray.
- sum_axis — Temporary alias of sum-axis for transition; same contract and return type.

## Functions and data structures

- sum-axis
  - Signature: (sum-axis a axis)
  - Arguments:
    - a — ndarray or array-like convertible via asarray; must be rank-2 with shape (r c).
      - Elements must be numeric; boolean arrays are not accepted for reductions in V1.
      - Output dtype is always :float64.
    - axis — integer; must be 0 (column-wise) or 1 (row-wise).
  - Returns: ndarray (rank-1, :float64, contiguous, row-major)
    - If axis=0: shape (c); each entry is the sum over r rows in that column.
    - If axis=1: shape (r); each entry is the sum over c columns in that row.
  - Errors and edge cases:
    - Signals an error if input is not 2D (0D or 1D), or if axis ∉ {0,1}.
    - Signals an error if elements are non-numeric or boolean.
    - Zero-sized dimensions are disallowed in V1.
  - Notes:
    - V1 constraints: ranks limited to 0D/1D/2D; row‑major contiguous only; positive element strides.
    - No dtype promotion/policy beyond :float64 result.
    - No keepdims; no multi-axis tuples; no NaN-special handling.

- sum_axis
  - Signature: (sum_axis a axis)
  - Behavior: Alias of sum-axis; identical arguments, return, and errors.
  - Notes: Transitional alias slated for removal after downstream tests migrate.

## Errors
Common invalid inputs and their effects (one-line forms shown; wrap with handler-case in scripts/tests when asserting failures):
- (sum-axis (asarray 7) 0)              ;; error: not a 2D ndarray
- (sum-axis (asarray '(1 2 3)) 0)       ;; error: not a 2D ndarray
- (sum-axis (asarray '((1 2) (3 4))) 2) ;; error: axis must be 0 or 1
- (sum-axis (asarray '((t nil) (t t))) 0) ;; error: boolean elements not allowed for reductions in V1
- (sum_axis (asarray '((1 2) (3 4))) -1)  ;; error: axis must be 0 or 1

## Examples
All snippets run from the repository root and use the integration loader.

```lisp
;; From repo root
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Column-wise sums (axis=0)
(let* ((a (asarray '((1 2 3) (4 5 6))))
       (s (sum-axis a 0)))
  (to-list s))                   ;; => (5.0d0 7.0d0 9.0d0)
```

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Row-wise sums (axis=1)
(let* ((a (asarray '((1 2 3) (4 5 6))))
       (s (sum-axis a 1)))
  (to-list s))                   ;; => (6.0d0 15.0d0)
```

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Alias usage (sum_axis)
(let* ((a (asarray '((1 2) (3 4))))
       (s (sum_axis a 0)))
  (to-list s))                   ;; => (4.0d0 6.0d0)
```

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Negative examples: invalid ranks and axes
(handler-case
    (sum-axis (asarray '(1 2 3)) 0)
  (error (e) (format t "error: ~a~%" e)))
(handler-case
    (sum-axis (asarray '((1 2) (3 4))) 2)
  (error (e) (format t "error: ~a~%" e)))
```

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; End-to-end: add two matrices, then sum rows
(let* ((a (asarray '((1 2) (3 4))))
       (b (asarray '((10 20) (30 40))))
       (c (add a b))                ;; => ((11 22) (33 44))
       (row-sums (sum-axis c 1)))
  (to-list row-sums))              ;; => (33.0d0 77.0d0)
```

## See also
- Core (shape/rank/strides): docs/modules/core.md; tests: tests/core-tests.lisp, src/core/tests.lisp
- Tensor (ndarray, asarray, to-list): docs/modules/tensor.md; tests: tests/tensor-tests.lisp
- Broadcasting helpers: docs/modules/broadcast.md; tests: tests/broadcast-tests.lisp
- Elementwise ufuncs (add, mul; boolean inputs disallowed): docs/modules/ufunc.md; tests: tests/ufunc-tests.lisp
- Indexing (1D boolean-select; all-false mask is an error): docs/modules/indexing.md; tests: src/indexing/tests.lisp
- Reductions tests: src/reductions/tests.lisp and tests/reduce-tests.lisp