# tensor

## Overview
The tensor module defines the ndarray data model and basic array construction, inspection, and indexed access. Use it to create arrays from scalars or lists, query shape/strides/dtype, convert to/from flat buffers, and perform safe element access. Higher‑level broadcasting, ufuncs, reductions, and boolean selection live in their respective modules.

## Public exports
- ndarray — struct representing an array: shape, strides, dtype, data
- make-array-from-flat — construct an ndarray from a shape and flat data (optional dtype)
- shape-of — return the array shape as a list (NIL, (n), or (r c))
- strides-of — return element strides (row‑major; NIL, (1), or (c 1))
- dtype-of — return array dtype (:float64 or :bool)
- flat-offset — compute flat index from an ND index and strides (with optional bounds check)
- aref-nd — element access by ND index with bounds checking
- asarray — construct an ndarray from scalars or (nested) lists
- to-list — convert ndarray back to number/list/list of lists

## Functions and data structures
- ndarray (struct)
  - Slots: shape (list), strides (list), dtype (symbol; :float64 or :bool), data (simple vector of double-floats or T/NIL)
  - Rank constraints: 0D/1D/2D only; row‑major contiguous; positive strides; no zero‑sized dims
  - Construction: prefer make-array-from-flat or asarray
  - Notes: V1 has no views, slicing, negative strides, or dtype promotion

- make-array-from-flat
  - Signatures: (make-array-from-flat shape flat-data) | (make-array-from-flat shape dtype flat-data)
  - Arguments:
    - shape: NIL | (n) | (r c); must be valid and non‑empty dims
    - dtype: :float64 (default) or :bool
    - flat-data: list or vector; length must equal product(shape)
  - Returns: new ndarray with default row‑major strides and coerced storage
  - Errors and edge cases:
    - Unsupported dtype; length mismatch; non‑numeric element for :float64; non‑boolean element for :bool; non‑list/vector flat-data
  - Notes:
    - Elements are coerced to double-float for :float64; T/NIL for :bool
    - 2D with :bool is not supported (use 1D masks only in V1)

- shape-of
  - Signature: (shape-of a)
  - Arguments: a: ndarray
  - Returns: NIL, (n), or (r c)
  - Notes: Mirrors internal metadata; no copying

- strides-of
  - Signature: (strides-of a)
  - Arguments: a: ndarray
  - Returns: NIL, (1), or (c 1) for row‑major contiguous layout

- dtype-of
  - Signature: (dtype-of a)
  - Arguments: a: ndarray
  - Returns: :float64 or :bool

- flat-offset
  - Signature: (flat-offset index strides &optional shape)
  - Arguments:
    - index: NIL | (i) | (i j)
    - strides: list matching rank (NIL | (s0) | (s0 s1))
    - shape: optional list for bounds checking; when NIL, rank is treated as 0D
  - Returns: non‑negative integer flat index (row‑major)
  - Errors and edge cases:
    - Index out of bounds when shape provided; rank other than 0/1/2; non‑integer indices
  - Notes: Used by aref-nd; respects row‑major default strides

- aref-nd
  - Signature: (aref-nd a index)
  - Arguments: a: ndarray; index: as for flat-offset
  - Returns: element at index (double-float for :float64 arrays; T/NIL for :bool arrays)
  - Errors and edge cases: Out‑of‑bounds index; rank > 2

- asarray
  - Signature: (asarray obj &key dtype)
  - Accepted OBJ forms and behavior:
    - ndarray: returned as‑is; if dtype provided and different, signals error (no coercion in V1)
    - Scalar number: creates 0D :float64; dtype, if provided, must be :float64
    - Scalar boolean (T/NIL): requires :dtype :bool
    - 1D list: all numbers -> :float64; all booleans -> requires :dtype :bool; mixed types/error
    - 2D list of rows: must be non‑empty rectangular numeric rows -> :float64; :bool not supported
  - Returns: ndarray with default contiguous row‑major strides
  - Errors and edge cases:
    - Empty 1D list; ragged 2D rows; non‑numeric 2D entries; boolean inputs without required :dtype :bool; unsupported dtype requests; dtype change on ndarray

- to-list
  - Signature: (to-list a)
  - Arguments: a: ndarray
  - Returns: number for 0D; list for 1D; list of lists for 2D
  - Errors: rank > 2 (not supported in V1)

## Examples
All snippets assume repository root.

- Create 1D numeric array
  (load "src/mini-array.lisp")
  (in-package :mini-array)
  (to-list (asarray '(1 2 3)))
  ;; => (1.0d0 2.0d0 3.0d0)

- Create 2D numeric array and inspect metadata
  (load "src/mini-array.lisp")
  (in-package :mini-array)
  (let* ((a (asarray '((1 2 3) (4 5 6)))))
    (list (shape-of a) (strides-of a) (dtype-of a) (to-list a)))
  ;; => ((2 3) (3 1) :FLOAT64 ((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0)))

- Explicit construction from flat data
  (load "src/mini-array.lisp")
  (in-package :mini-array)
  (to-list (make-array-from-flat '(2 2) :float64 '(1 2 3 4)))
  ;; => ((1.0d0 2.0d0) (3.0d0 4.0d0))

- Indexed element access (row‑major)
  (load "src/mini-array.lisp")
  (in-package :mini-array)
  (let ((a (asarray '((1 2 3) (4 5 6)))))
    (aref-nd a '(1 2)))
  ;; => 6.0d0

- 1D boolean mask array (requires :dtype :bool)
  (load "src/mini-array.lisp")
  (in-package :mini-array)
  (to-list (asarray '(t nil t) :dtype :bool))
  ;; => (T NIL T)

- Negative example: boolean list without dtype signals error
  (load "src/mini-array.lisp")
  (in-package :mini-array)
  (asarray '(t nil t))
  ;; => ERROR (boolean 1D list requires :dtype :bool)

- Negative example: out‑of‑bounds index signals error
  (load "src/mini-array.lisp")
  (in-package :mini-array)
  (let* ((a (asarray '(10 20 30)))
         (s (strides-of a))
         (sh (shape-of a)))
    (flat-offset '(3) s sh))
  ;; => ERROR (index out of bounds)

## See also
- Modules:
  - ../modules/core.md — shape utilities and default-strides
  - ../modules/broadcast.md — broadcast-shape and iterator utilities
  - ../modules/ufunc.md — elementwise add/mul over broadcasted shapes
  - ../modules/reductions.md — sum-axis for 2D
  - ../modules/indexing.md — boolean-select masks and policy
- Source:
  - src/tensor/package.lisp — exports
  - src/tensor/ndarray.lisp — data model and constructors
  - src/tensor/indexing.lisp — flat-offset, aref-nd
  - src/tensor/conversion.lisp — asarray, to-list
- Tests:
  - tests/tensor-tests.lisp — tensor module tests
  - tests/core-tests.lisp — supporting core behavior