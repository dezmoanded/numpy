# reductions

## Overview
Reductions provides 2D axis-wise summation for ndarrays. Use it to compute column-wise (axis=0) or row-wise (axis=1) sums, returning a 1D contiguous :float64 ndarray. V1 supports only 2D inputs and does not implement keepdims or higher-rank reductions.

## Public exports
- sum-axis — Sum a 2D ndarray along a given axis (0 for columns, 1 for rows). Returns a 1D :float64 ndarray.
- sum_axis — Temporary alias of sum-axis for transition. Same behavior and contract.

## Functions and data structures
- sum-axis
  - Signature: (sum-axis a axis)
  - Arguments:
    - a: ndarray or array-like convertible via asarray. Must be 2D of shape (r c).
      - Elements must be numeric; dtype may be :float64; boolean inputs are not accepted for 2D reductions in V1.
      - Output dtype is always :float64.
    - axis: integer, must be 0 or 1.
  - Returns:
    - ndarray (rank-1, :float64, contiguous, row-major)
      - If axis=0: shape (c), each entry is the sum over r rows in that column.
      - If axis=1: shape (r), each entry is the sum over c columns in that row.
  - Errors and edge cases:
    - Signals error if input is not 2D (e.g., 0D or 1D).
    - Signals error if axis is not 0 or 1.
    - Signals error if an element is not numeric.
    - Zero-sized dimensions are not supported in V1.
  - Notes:
    - V1 constraints: ranks limited to 0D/1D/2D, positive row-major strides, contiguous storage.
    - No dtype promotion: result dtype is :float64 regardless of input.
    - No keepdims, no reducing over multiple axes, no NaN-special handling.

- sum_axis
  - Signature: (sum_axis a axis)
  - Arguments/Returns/Errors: Identical to sum-axis.
  - Notes:
    - Transitional alias of sum-axis; scheduled for removal after downstream updates.

## Examples
- Setup (from repo root):
  - (in-package :cl)
  - (load "src/mini-array.lisp")
  - (in-package :mini-array)

- Column-wise sums (axis=0)
  - (let* ((a (asarray '((1 2 3) (4 5 6))))
          (s (sum-axis a 0)))
      (to-list s))
  - => (5.0d0 7.0d0 9.0d0)

- Row-wise sums (axis=1)
  - (let* ((a (asarray '((1 2 3) (4 5 6))))
          (s (sum-axis a 1)))
      (to-list s))
  - => (6.0d0 15.0d0)

- Alias usage (sum_axis)
  - (let* ((a (asarray '((1 2) (3 4))))
          (s (sum_axis a 0)))
      (to-list s))
  - => (4.0d0 6.0d0)

- Error on non-2D input
  - (handler-case
        (sum-axis (asarray '(1 2 3)) 0)
      (error (e) (format t "errored: ~a~%" e)))
  - => prints an error message (expected)

## See also
- Related modules:
  - ../modules/tensor.md — ndarray struct, asarray, to-list, indexing helpers.
  - ../modules/broadcast.md — broadcasting rules (not used by reductions in V1).
  - ../modules/ufunc.md — elementwise add/mul.
  - ../modules/core.md — shape, rank, and dtype utilities.
  - ../modules/indexing.md — boolean-select policy (mask dtype :bool).
- Source and tests:
  - src/reductions/package.lisp
  - src/reductions/sum.lisp
  - src/reductions/tests.lisp
  - tests/reduce-tests.lisp