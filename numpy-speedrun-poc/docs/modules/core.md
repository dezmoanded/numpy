# core

## Overview
The core module provides small, composable shape and dtype primitives that underpin the rest of the system. Use it to validate shapes, compute ranks and default row‑major strides for 0D/1D/2D arrays, and (internally) enforce strict V1 dtype rules.

## Public exports
- product — Product of dimensions; product NIL → 1.
- rank — Rank (length) of a shape; NIL → 0.
- valid-shape-p — Predicate for V1‑valid shapes (proper list, rank ≤ 2, all dims > 0).
- default-strides — Row‑major element strides for ranks 0..2; errors if shape invalid.

Note: Dtype helpers (dtype-p, coerce-dtype-value, infer-scalar-dtype) live in :mini-array.core but are not re‑exported from :mini-array in V1.

## Functions and data structures

- product
  - Signature: (product dims)
  - Arguments: dims — proper list of positive integers; NIL allowed.
  - Returns: non‑negative integer; product NIL = 1.
  - Errors and edge cases: None (assumes dims already validated); use valid-shape-p when dims come from a shape.
  - Notes: Intended for size computations of contiguous arrays.

- rank
  - Signature: (rank shape)
  - Arguments: shape — proper list (NIL, (n), or (r c)).
  - Returns: 0 for NIL, 1 for (n), 2 for (r c).
  - Errors and edge cases: Passing a non‑list may signal a CL type error via length; call valid-shape-p first when unsure.
  - Notes: V1 supports ranks 0..2 only.

- valid-shape-p
  - Signature: (valid-shape-p shape)
  - Arguments: shape — expected NIL, (n), or (r c).
  - Returns: T iff shape is a proper list with rank ≤ 2 and every dim is a positive integer (> 0).
  - Errors and edge cases: Returns NIL for zero or negative dims, non‑lists, improper lists, or rank > 2.
  - Notes: Zero‑sized dimensions are disallowed in V1.

- default-strides
  - Signature: (default-strides shape)
  - Arguments: shape — NIL, (n), or (r c); must be valid per valid-shape-p.
  - Returns:
    - NIL   → NIL (0D scalar)
    - (n)   → (1)
    - (r c) → (c 1) (row‑major)
  - Errors and edge cases: Signals an error if shape invalid or rank > 2.
  - Notes: Strides are in element units, not bytes; row‑major contiguous only in V1.

## Examples
All snippets run from the repository root and use the integration loader.

```lisp
;; From repo root
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(rank nil)                     ;; => 0
(product '(2 3))               ;; => 6
(valid-shape-p '(2 3))         ;; => T
(default-strides '(2 3))       ;; => (3 1)
```

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(valid-shape-p '(0))           ;; => NIL (zero-sized dims disallowed)
(valid-shape-p '(2 -1))        ;; => NIL
```

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Negative example: invalid shape raises an error
(handler-case
    (default-strides '(2 0))   ;; invalid: zero dim
  (error (e)
    (format t "error: ~a~%" e)))
;; prints: error: default-strides: invalid shape (2 0) (rank<=2, positive integers only)
```

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Integration context: confirm row-major semantics via strides
;; (uses tensor module for construction/inspection)
(defparameter *a* (asarray '((1 2 3) (4 5 6))))
(shape-of *a*)                  ;; => (2 3)
(default-strides (shape-of *a*)) ;; => (3 1)
(to-list *a*)                   ;; => ((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0))
```

## See also
- Tensor construction and indexing: docs/modules/tensor.md; tests: tests/tensor-tests.lisp
- Broadcasting helpers: docs/modules/broadcast.md; tests: tests/broadcast-tests.lisp
- Elementwise ufuncs (add, mul): docs/modules/ufunc.md; tests: tests/ufunc-tests.lisp
- Reductions (sum-axis): docs/modules/reductions.md; tests: tests/reduce-tests.lisp
- Core module tests: tests/core-tests.lisp and src/core/tests.lisp
