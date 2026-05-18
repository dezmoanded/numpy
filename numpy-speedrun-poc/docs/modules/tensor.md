# tensor

## Overview
The tensor module defines the ndarray data model and the primitives to build, inspect, and index arrays. Use it to construct arrays from scalars or lists, query shape/strides, convert to/from flat buffers, and perform safe element access. V1 constraints apply: numeric arrays are :float64 only; :bool is allowed only for 1D boolean masks; ranks limited to 0D/1D/2D; row‑major contiguous layout with positive, element‑based strides; zero‑sized dimensions are disallowed.

## Public exports
- ndarray — struct representing an array: shape, strides, dtype, data
- make-array-from-flat — construct an ndarray from a shape and flat data (optional dtype)
- shape-of — return the array shape (NIL, (n), or (r c))
- strides-of — return element strides (NIL, (1), or (c 1), row‑major)
- flat-offset — compute flat index from an ND index and strides (optional bounds via shape)
- aref-nd — element access by ND index with bounds checking
- asarray — construct an ndarray from scalars or (nested) lists; ndarray is idempotent
- to-list — convert ndarray back to number/list/list of lists

Note: dtype-of exists in :mini-array.tensor but is not re‑exported from :mini-array in V1.

## Functions and data structures

- ndarray (struct)
  - Slots: shape (list), strides (list), dtype (symbol), data (simple vector)
  - Rank/layout: 0D/1D/2D only; row‑major contiguous; positive strides; no zero‑sized dims
  - Dtypes: :FLOAT64 for numerics; :BOOL for 1D masks
  - Construction: prefer make-array-from-flat or asarray
  - Notes: dtype is immutable; V1 has no views/slicing/negative strides. Internal write helper %set-aref-nd exists but is not exported.

- make-array-from-flat
  - Signatures:
    - (make-array-from-flat shape flat-data)
    - (make-array-from-flat shape dtype flat-data)
  - Arguments:
    - shape: NIL | (n) | (r c); must be valid and dims > 0
    - dtype: :FLOAT64 (default) or :BOOL
    - flat-data: list or vector; length must equal (product shape)
  - Returns: new ndarray with row‑major default strides and coerced storage
  - Errors: unsupported dtype; length mismatch; bad element type for dtype; invalid shape
  - Examples:
```lisp
;; From repo root
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(to-list (make-array-from-flat '(2 2) '(1 2 3 4)))
;; => ((1.0d0 2.0d0) (3.0d0 4.0d0))

(to-list (make-array-from-flat '(3) :float64 #(10 20 30)))
;; => (10.0d0 20.0d0 30.0d0)
```

- shape-of
  - Signature: (shape-of a)
  - Returns: NIL, (n), or (r c)
  - Example:
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(shape-of (asarray '((1 2 3) (4 5 6))))
;; => (2 3)
```

- strides-of
  - Signature: (strides-of a)
  - Returns: NIL, (1), or (c 1) for row‑major contiguous
  - Example:
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(strides-of (asarray '((1 2 3) (4 5 6))))
;; => (3 1)
```

- flat-offset
  - Signature: (flat-offset index strides &optional shape)
  - Arguments: index = NIL|(i)|(i j); strides matches rank; optional shape enables bounds checks
  - Returns: non‑negative flat index (row‑major)
  - Notes: For 0D, index must be NIL
  - Examples:
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(let* ((a (asarray '((1 2 3) (4 5 6))))
       (s (strides-of a))
       (sh (shape-of a)))
  (flat-offset '(1 2) s sh))
;; => 5
```

- aref-nd
  - Signature: (aref-nd a index)
  - Returns: element at index (double-float for :FLOAT64; T/NIL for :BOOL)
  - Examples:
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(aref-nd (asarray '((1 2 3) (4 5 6))) '(1 0))
;; => 4.0d0
```

- asarray
  - Signature: (asarray obj &key dtype)
  - Behavior:
    - ndarray: returned as‑is; if a different dtype is requested, signals error (no coercion in V1)
    - Scalar number: 0D :FLOAT64 (dtype, if provided, must be :FLOAT64)
    - Scalar boolean T/NIL: requires :dtype :BOOL
    - 1D list: all numbers → :FLOAT64; all booleans → requires :dtype :BOOL; mixed → error
    - 2D list: non‑empty rectangular numeric rows only → :FLOAT64; :BOOL not supported
  - Examples:
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(to-list (asarray '(1 2 3)))
;; => (1.0d0 2.0d0 3.0d0)

(to-list (asarray '((1 2) (3 4))))
;; => ((1.0d0 2.0d0) (3.0d0 4.0d0))

(to-list (asarray '(t nil t) :dtype :bool))
;; => (T NIL T)
```

- to-list
  - Signature: (to-list a)
  - Returns: number (0D), list (1D), or list of lists (2D)
  - Example:
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(to-list (asarray 7))
;; => 7.0d0
```

## Errors
One‑line illustrations of invalid inputs that signal an error.

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; make-array-from-flat: length mismatch
(make-array-from-flat '(2 2) '(1 2 3))              ;; ERROR

;; make-array-from-flat: bad dtype element
(make-array-from-flat '(2) :float64 '(1 t))         ;; ERROR

;; asarray: boolean list requires :dtype :bool
(asarray '(t nil t))                                ;; ERROR

;; asarray: ragged 2D rows
(asarray '((1 2) (3)))                              ;; ERROR

;; asarray: 2D :bool unsupported
(asarray '((t) (nil)) :dtype :bool)                 ;; ERROR

;; asarray: dtype change on ndarray
(let ((a (asarray '(1 2 3))))
  (asarray a :dtype :bool))                         ;; ERROR

;; flat-offset / aref-nd: OOB
(let* ((a (asarray '(10 20 30)))
       (s (strides-of a))
       (sh (shape-of a)))
  (flat-offset '(3) s sh))                          ;; ERROR
(aref-nd (asarray '((1 2) (3 4))) '(2 0))           ;; ERROR

;; 0D index must be NIL
(let ((a (asarray 42)))
  (aref-nd a '(0)))                                 ;; ERROR
```

## Examples
Compact end‑to‑end usage.

```lisp
;; From repo root
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Build inputs
(defparameter *a* (asarray '((1 2 3) (4 5 6))))
(defparameter *mask* (asarray '(t nil t) :dtype :bool))

;; Inspect metadata
(list (shape-of *a*) (strides-of *a*))
;; => ((2 3) (3 1))

;; Element access (row-major)
(aref-nd *a* '(1 2))
;; => 6.0d0

;; Round-trip to list
(to-list *a*)
;; => ((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0))
```

## See also
- Core utilities (shapes/strides): docs/modules/core.md; tests: tests/core-tests.lisp
- Broadcasting helpers: docs/modules/broadcast.md; tests: tests/broadcast-tests.lisp
- Elementwise ufuncs (add, mul): docs/modules/ufunc.md; tests: tests/ufunc-tests.lisp
- Reductions (sum-axis): docs/modules/reductions.md; tests: tests/reduce-tests.lisp
- Indexing (boolean-select): docs/modules/indexing.md; tests: src/indexing/tests.lisp
- Source references: src/tensor/package.lisp, src/tensor/ndarray.lisp, src/tensor/indexing.lisp, src/tensor/conversion.lisp
