# Indexing

## Overview
Provides boolean-based selection for 1D numeric arrays. Use when you need to filter elements of a rank-1 ndarray by a same-length boolean mask. V1 is intentionally narrow: masks are lists of T/NIL, inputs are :float64 1D arrays, and zero-sized results are disallowed.

## Public exports
- mini-array.indexing:boolean-select — Select elements from a 1D ndarray using a same-length boolean mask (list of T/NIL). Returns a new 1D ndarray with order preserved.

Note: In V1 this symbol is exported by the :mini-array.indexing package (not re-exported from :mini-array).

## Functions and data structures

boolean-select
- Signature: (mini-array.indexing:boolean-select array mask)
- Arguments:
  - array: an ndarray (package :mini-array) of rank 1; dtype must be :float64; contiguous row-major with positive strides (as produced by asarray/make-array-from-flat).
  - mask: a proper list of booleans (T or NIL) of length equal to (first (shape-of array)).
- Returns: a new 1D ndarray (dtype :float64) containing the elements of array where the corresponding mask entry is T. Order is preserved.
- Errors and edge cases:
  - Type error if array is not an ndarray.
  - Rank error if array rank != 1.
  - Length error if mask is not a list or its length != array length.
  - Domain error if any mask entry is not T or NIL.
  - Policy error if all mask entries are NIL (zero-sized outputs are disallowed in V1).
- Notes:
  - No broadcasting; mask must be 1D and exactly match array length.
  - Implementation builds a fresh contiguous output via make-array-from-flat; no views.
  - Only numeric arrays (:float64) are supported in V1; mask is conceptually :bool.

## Examples
All examples load the integration loader and then qualify boolean-select from :mini-array.indexing.

Example 1 — basic selection

```
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(let* ((v (asarray '(10 20 30 40)))
       (mask '(t nil t nil))
       (sel (mini-array.indexing:boolean-select v mask)))
  (to-list sel))
;; => (10.0d0 30.0d0)
```

Example 2 — keep order with scattered truths

```
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(let* ((v (asarray '(1 2 3 4 5)))
       (mask '(nil t nil t t))
       (sel (mini-array.indexing:boolean-select v mask)))
  (to-list sel))
;; => (2.0d0 4.0d0 5.0d0)
```

Example 3 — error on all-false mask

```
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(handler-case
    (progn
      (mini-array.indexing:boolean-select (asarray '(1 2 3)) '(nil nil nil))
      :unexpected)
  (error (e) (format t "errored: ~a~%" e)))
;; => prints: errored: boolean-select: all-false MASK would yield empty selection, disallowed in V1
```

Example 4 — length mismatch

```
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(handler-case
    (mini-array.indexing:boolean-select (asarray '(10 20 30)) '(t nil))
  (error (e) (format t "errored: ~a~%" e)))
;; => prints an error about mask length
```

Example 5 — non-boolean entries

```
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(handler-case
    (mini-array.indexing:boolean-select (asarray '(1 2 3)) '(t 0 t))
  (error (e) (format t "errored: ~a~%" e)))
;; => prints an error about mask containing only T/NIL
```

Example 6 — rank check (2D not supported in V1)

```
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(handler-case
    (mini-array.indexing:boolean-select (asarray '((1 2) (3 4))) '(t t))
  (error (e) (format t "errored: ~a~%" e)))
;; => prints an error about rank-1 requirement
```

## See also
- Module pages:
  - ../modules/tensor.md — ndarray shape/strides, asarray, to-list
  - ../modules/core.md — shape validation, helpers used by construction
- Source and tests:
  - src/indexing/package.lisp — package and export
  - src/indexing/boolean-select.lisp — implementation
  - src/indexing/tests.lisp — module tests
  - examples/boolean-mask-demo.lisp — runnable example script