# Broadcast

## Overview
Broadcast provides NumPy-style right‑aligned shape reconciliation and index utilities for ranks 0D/1D/2D. Use it to compute the output shape for elementwise operations, enumerate row‑major indices, and project an output index back to an input’s index under broadcasting.

## Public exports
- broadcast-shape — Compute the right‑aligned broadcasted output shape of two shapes (NIL counts as 0D).
- all-indices — Enumerate all row‑major indices for a shape as lists (NIL, (n), or (r c)).
- project-broadcast-index — Map an output index into an input index given input and output shapes under broadcasting.

(All three symbols are re‑exported from :mini-array.)

## Functions and data structures
- broadcast-shape (a-shape b-shape)
  - Arguments: a-shape, b-shape
    - Each is either NIL (0D) or a proper list of positive integers for 1D (n) or 2D (r c).
  - Returns: the broadcasted shape list (rank ≤ 2), using right‑aligned rules:
    - Per dimension: if equal → that dim; else if one is 1 → the other; else → error.
  - Errors and edge cases:
    - Signals an error if either shape is invalid (non‑positive dims, rank > 2) or shapes are incompatible to broadcast.
    - Zero‑sized dims are not allowed in V1.
  - Notes:
    - Shapes are validated via core:valid-shape-p.
    - Left‑padding with 1s is used internally to equalize ranks before comparison.

- all-indices (shape)
  - Arguments: shape — NIL, (n), or (r c); dims must be positive integers.
  - Returns: list of indices in row‑major order:
    - NIL → (NIL)
    - (n) → ((0) … (n‑1))
    - (r c) → ((0 0) (0 1) … (r‑1 c‑1))
  - Errors and edge cases:
    - Signals an error for invalid shapes or rank > 2.
  - Notes:
    - Intended for small shapes in V1; returns a concrete list for simplicity.

- project-broadcast-index (out-index in-shape out-shape)
  - Arguments:
    - out-index: list whose length equals the rank of out-shape (NIL for 0D).
    - in-shape: NIL, (n), or (r c); must be broadcastable to out-shape.
    - out-shape: the target broadcasted shape; must be valid (rank ≤ 2).
  - Returns: an index list into in-shape after dropping any left padding; NIL for 0D inputs.
  - Errors and edge cases:
    - Signals an error if shapes are invalid, out-index rank mismatches out-shape, or in-shape is not broadcastable to out-shape.
  - Notes:
    - For each padded dimension: if in-dim == 1 → 0; else copy the corresponding element from out-index.

## Examples
All examples use the integration loader and the :mini-array package.

; Load once per REPL session
(load "src/mini-array.lisp")
(in-package :mini-array)

; 1) Compute a broadcasted shape (2x1) ⊕ (1x3) → (2x3)
(broadcast-shape '(2 1) '(1 3))
; => (2 3)

; 2) 0D with 1D broadcasts to the 1D shape
(broadcast-shape nil '(3))
; => (3)

; 3) Enumerate all row‑major indices for a 2x3 shape
(all-indices '(2 3))
; => ((0 0) (0 1) (0 2) (1 0) (1 1) (1 2))

; 4) Project an output index back to each input
(project-broadcast-index '(1 2) '(1 3) '(2 3))  ; rows broadcast
; => (0 2)
(project-broadcast-index '(1 2) '(2 1) '(2 3))  ; cols broadcast
; => (1 0)

; 5) Negative example: incompatible shapes (2) and (3)
(handler-case (broadcast-shape '(2) '(3))
  (error (c) (format t "error: ~a~%" c)))
; Prints an error message; exact text not part of the API contract.

; 6) Practical: broadcasting in action via add (ufunc)
(let* ((a (asarray '((10 20 30))))   ; shape (1 3)
       (b (asarray '((100) (200))))  ; shape (2 1)
       (out (add a b)))              ; shape (2 3)
  (to-list out))
; => ((110.0d0 120.0d0 130.0d0)
;     (210.0d0 220.0d0 230.0d0))

## See also
- Core shape utilities: ../modules/core.md (product, rank, valid-shape-p, default-strides)
- Tensor basics and to-list: ../modules/tensor.md
- Ufuncs that rely on broadcasting: ../modules/ufunc.md
- Reductions (for context on outputs after elementwise ops): ../modules/reductions.md
- Source: src/broadcast/package.lisp, src/broadcast/shape.lisp, src/broadcast/projection.lisp, src/broadcast/iterator.lisp
- Tests: tests/broadcast-tests.lisp; integration usage also covered by tests/ufunc-tests.lisp and tests/test-mini-array.lisp