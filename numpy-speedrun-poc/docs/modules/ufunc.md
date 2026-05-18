# ufunc

## Overview
Elementwise numeric operations with NumPy-style right‑aligned broadcasting. This module provides a generic binary engine and the V1 numeric ufuncs add and mul over 0D/1D/2D float64 arrays. Scalars are treated as 0D (shape NIL). No dtype promotion; numeric outputs are :float64.

## Public exports
- binary-ufunc — Generic broadcasted elementwise engine for custom binary ops; materializes a new array.
- add — Elementwise addition with broadcasting; float64 output.
- mul — Elementwise multiplication with broadcasting; float64 output.

## Functions and data structures
This module defines no new data structures; it operates on ndarray from the tensor module.

- binary-ufunc
  - Signature: (binary-ufunc a b op &key out-dtype name (allow-bool-input-p nil))
  - Arguments
    - a, b: real scalar or ndarray, rank ∈ {0,1,2}, row‑major contiguous. Boolean-typed inputs are rejected unless allow-bool-input-p is T.
    - op: function of two values; numeric paths must return a real number. For internal comparisons, op may return T/NIL with :bool out.
    - out-dtype: required; one of :float64 or :bool.
    - name: optional symbol/string used in error messages.
    - allow-bool-input-p: when T, permits :bool-typed inputs (not used by public add/mul).
  - Returns: New ndarray with broadcasted shape and dtype = out-dtype.
  - Notes: V1 numeric contract coerces results to double-float; no promotion.
  - Example (custom subtract)
    ```lisp
    (load "numpy-speedrun-poc/src/mini-array.lisp")
    (in-package :mini-array)
    (to-list (binary-ufunc (asarray '((5 6))) (asarray '((2 3))) #'- :out-dtype :float64))
    ;; => ((3.0d0 3.0d0))
    ```

- add
  - Signature: (add a b)
  - Behavior: Elementwise addition of a and b with right‑aligned broadcasting; float64 output.
  - Returns: New ndarray of dtype :float64 and broadcasted shape.
  - Example (scalar + vector)
    ```lisp
    (load "numpy-speedrun-poc/src/mini-array.lisp")
    (in-package :mini-array)
    (to-list (add (asarray '(1 2 3)) 10))
    ;; => (11.0d0 12.0d0 13.0d0)
    ```
  - Edge cases: Boolean-typed inputs are not accepted; broadcasting incompatibility errors are signaled.

- mul
  - Signature: (mul a b)
  - Behavior: Elementwise multiplication of a and b with right‑aligned broadcasting; float64 output.
  - Returns: New ndarray of dtype :float64 and broadcasted shape.
  - Example (vector * vector)
    ```lisp
    (load "numpy-speedrun-poc/src/mini-array.lisp")
    (in-package :mini-array)
    (to-list (mul (asarray '(1 2 3)) (asarray '(10 20 30))))
    ;; => (10.0d0 40.0d0 90.0d0)
    ```
  - Edge cases: Boolean-typed inputs are not accepted; broadcasting incompatibility errors are signaled.

## Errors
Representative invalid inputs and the resulting errors. One-line forms shown with handler-case.

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Broadcasting mismatch
(handler-case
    (add (asarray '(1 2 3)) (asarray '(1 2)))
  (error (e) (format t "error: ~a~%" e)))

;; Boolean inputs rejected for numeric ufuncs
(handler-case
    (mul (asarray '(1 2 3)) (asarray '(t nil t) :dtype :bool))
  (error (e) (format t "error: ~a~%" e)))

;; Invalid out-dtype for binary-ufunc
(handler-case
    (binary-ufunc 1 2 #'+ :out-dtype :int32)
  (error (e) (format t "error: ~a~%" e)))
```

Notes
- Ranks beyond 2 and zero-sized dimensions are disallowed in V1; shape/dtype validation is enforced by dependencies (tensor/core/broadcast).
- Scalars normalize to 0D (shape NIL) before broadcasting.

## Examples
Compact end-to-end usage from repo root.

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Build a 2x3 matrix, scale by 10, then add 1 — all via broadcasting
(let* ((a (asarray '((1 2 3)
                     (4 5 6))))   ; shape (2 3)
       (scaled (mul a 10))          ; 0D*2D => (2 3)
       (shifted (add scaled 1)))    ; 0D+2D => (2 3)
  (to-list shifted))
;; => ((11.0d0 21.0d0 31.0d0)
;;     (41.0d0 51.0d0 61.0d0))

;; Broadcast (1x3) + (2x1) -> (2x3)
(let* ((row (asarray '((1 2 3))))    ; shape (1 3)
       (col (asarray '((10) (20))))) ; shape (2 1)
  (to-list (add row col)))
;; => ((11.0d0 12.0d0 13.0d0)
;;     (21.0d0 22.0d0 23.0d0))

;; Custom op via binary-ufunc
(to-list (binary-ufunc (asarray '(5 6 7)) 2 #'- :out-dtype :float64))
;; => (3.0d0 4.0d0 5.0d0)
```

## See also
- Modules
  - docs/modules/core.md — core shape/dtype utilities used by the ufunc engine.
  - docs/modules/tensor.md — ndarray, asarray, to-list, and indexing used by ufuncs.
  - docs/modules/broadcast.md — broadcasting shape/projection/iteration used by ufuncs.
  - docs/modules/reductions.md — reductions (sum-axis) that often consume ufunc results.
  - docs/modules/indexing.md — boolean-select; not used directly by numeric ufuncs.
- Source
  - src/ufunc/package.lisp
  - src/ufunc/engine.lisp
  - src/ufunc/numeric-ops.lisp
- Tests
  - tests/ufunc-tests.lisp — standalone ufunc tests for add/mul, broadcasting, and error cases.
