# ufunc

## Overview
Elementwise numeric operations with NumPy-style right‑aligned broadcasting. Provides a generic binary engine plus V1 numeric ufuncs add and mul over 0D/1D/2D float64 arrays. Use for broadcasted elementwise math; no dtype promotion.

## Public exports
- binary-ufunc — Generic broadcasted elementwise engine for custom binary ops; materializes a new array.
- add — Elementwise addition with broadcasting; float64 output.
- mul — Elementwise multiplication with broadcasting; float64 output.

## Functions and data structures
This module defines no new data structures; it operates on ndarray from the tensor module.

- binary-ufunc
  - Signature
    - (binary-ufunc a b op &key out-dtype name (allow-bool-input-p nil))
  - Arguments and expected types/shapes
    - a, b: real scalar or ndarray, rank ∈ {0,1,2}, row‑major contiguous. Default numeric dtype is :float64. Boolean-typed inputs are rejected unless allow-bool-input-p is T.
    - op: function of two values; for numeric paths returns a real number; for internal comparisons may return T/NIL.
    - out-dtype: required; one of :float64 or :bool.
    - name: optional symbol/string used in error messages.
    - allow-bool-input-p: when T, permits :bool-typed input arrays (default NIL). Not used for public add/mul.
    - Shapes must be broadcast-compatible by right‑aligned rules; ranks limited to 0/1/2.
  - Returns
    - New ndarray with shape broadcast-shape(shape-of a, shape-of b), contiguous row‑major storage, dtype = out-dtype.
  - Errors and edge cases
    - Invalid out-dtype (must be :float64 or :bool).
    - Boolean input arrays when allow-bool-input-p is NIL.
    - Broadcasting mismatch between a and b shapes.
    - Unsupported rank (>2) or zero-sized dims (delegated to dependencies).
    - Non-numeric payloads (rejected by asarray/dtype checks).
  - Notes
    - V1 numeric contract: coerce results to double-float; no dtype promotion.
    - :bool output exists for internal comparisons/masking; comparison helpers are not re-exported via :mini-array in V1.
    - Scalars normalize to 0D arrays (shape NIL).

- add
  - Signature
    - (add a b)
  - Arguments and expected types/shapes
    - a, b: real scalar or ndarray (0D/1D/2D), dtype :float64; shapes must be broadcastable.
  - Returns
    - New float64 ndarray with broadcasted shape.
  - Errors and edge cases
    - Shape mismatch under broadcasting.
    - Boolean-typed input arrays are rejected.
    - Ranks >2 and zero-sized dims are not supported.
  - Notes
    - Implements numeric path of binary-ufunc with out-dtype :float64.

- mul
  - Signature
    - (mul a b)
  - Arguments and expected types/shapes
    - a, b: real scalar or ndarray (0D/1D/2D), dtype :float64; shapes must be broadcastable.
  - Returns
    - New float64 ndarray with broadcasted shape.
  - Errors and edge cases
    - Shape mismatch under broadcasting.
    - Boolean-typed input arrays are rejected.
    - Ranks >2 and zero-sized dims are not supported.
  - Notes
    - Implements numeric path of binary-ufunc with out-dtype :float64.

## Examples
Runnables from repo root.

- Load once and enter package
```lisp
(in-package :cl)
(load "src/mini-array.lisp")
(in-package :mini-array)
```

- Scalar + vector add
```lisp
(to-list (add (asarray '(1 2 3)) 10))
;; => (11.0d0 12.0d0 13.0d0)
```

- Broadcast (1x3) + (2x1) -> (2x3)
```lisp
(let* ((row (asarray '((1 2 3))))    ; shape (1 3)
       (col (asarray '((10) (20))))) ; shape (2 1)
  (to-list (add row col)))
;; => ((11.0d0 12.0d0 13.0d0)
;;     (21.0d0 22.0d0 23.0d0))
```

- 0D + 0D -> scalar 5.0d0
```lisp
(to-list (add 2 3))
;; => 5.0d0
```

- Elementwise multiply
```lisp
(to-list (mul (asarray '(1 2 3)) (asarray '(10 20 30))))
;; => (10.0d0 40.0d0 90.0d0)
```

- Custom op via binary-ufunc (subtract)
```lisp
(to-list (binary-ufunc (asarray '((5 6))) (asarray '((2 3))) #'- :out-dtype :float64))
;; => ((3.0d0 3.0d0))
```

- Negative: shape mismatch errors
```lisp
(handler-case
    (add (asarray '(1 2 3)) (asarray '(1 2)))
  (error (e) (format t "Got expected error: ~a~%" e)))
```

- Negative: boolean inputs rejected for numeric ufuncs
```lisp
(handler-case
    (add (asarray '(1 2 3)) (asarray '(t nil t) :dtype :bool))
  (error (e) (format t "Got expected error: ~a~%" e)))
```

## See also
- Modules
  - ../modules/core.md — core shape/dtype utilities used by the ufunc engine.
  - ../modules/tensor.md — ndarray, asarray, to-list, and indexing used by ufuncs.
  - ../modules/broadcast.md — broadcasting shape/projection/iteration used by ufuncs.
- Source
  - src/ufunc/package.lisp
  - src/ufunc/engine.lisp
  - src/ufunc/numeric-ops.lisp
- Tests
  - tests/ufunc-tests.lisp — standalone ufunc tests for add/mul, broadcasting, and error cases.
