# indexing

## Overview
The indexing module provides boolean-based selection for 1D numeric arrays. Use it to filter elements of a rank-1 ndarray using a same-length boolean mask. V1 is intentionally narrow: inputs are :float64 ndarrays (row‑major, contiguous, positive strides), masks are proper lists of T/NIL (conceptual :bool), broadcasting is not supported, and zero-sized results are disallowed (all-false masks raise an error).

## Public exports
- boolean-select — Select elements from a 1D ndarray using a same-length boolean mask (list of T/NIL). Returns a new 1D ndarray with order preserved.

Note: In V1 this symbol is exported by the :mini-array.indexing package and is not re‑exported from :mini-array. Call it as mini-array.indexing:boolean-select.

## Functions and data structures

- boolean-select
  - Signature: (mini-array.indexing:boolean-select array mask)
  - Arguments:
    - array — an ndarray (package :mini-array) of rank 1; dtype must be :float64; contiguous row‑major with positive strides (as produced by asarray/make-array-from-flat).
    - mask — a proper list of booleans (T or NIL) of length equal to (first (shape-of array)).
  - Returns: a new 1D ndarray (dtype :float64) containing the elements of array where the corresponding mask entry is T. Order is preserved.
  - Notes:
    - No broadcasting; mask rank must be 1 and length must exactly match the array length.
    - A fresh contiguous output is allocated via make-array-from-flat; no views are produced in V1.

Examples (for this function)

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

(let* ((v (asarray '(10 20 30 40)))
       (mask '(t nil t nil))
       (sel (mini-array.indexing:boolean-select v mask)))
  (to-list sel))
;; => (10.0d0 30.0d0)
```

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Edge case: scattered truths preserve order
(let* ((v (asarray '(1 2 3 4 5)))
       (mask '(nil t nil t t))
       (sel (mini-array.indexing:boolean-select v mask)))
  (to-list sel))
;; => (2.0d0 4.0d0 5.0d0)
```

## Errors
The function signals errors in the following cases (one-line examples shown with handler-case):

- Rank not 1 (0D/2D unsupported in V1)
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(handler-case
    (mini-array.indexing:boolean-select (asarray '((1 2) (3 4))) '(t t))
  (error (e) (format t "error: ~a~%" e)))
```

- Length mismatch between mask and array
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(handler-case
    (mini-array.indexing:boolean-select (asarray '(10 20 30)) '(t nil))
  (error (e) (format t "error: ~a~%" e)))
```

- Non-boolean mask entries (must be T or NIL only)
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(handler-case
    (mini-array.indexing:boolean-select (asarray '(1 2 3)) '(t 0 t))
  (error (e) (format t "error: ~a~%" e)))
```

- All-false mask (zero-sized outputs disallowed in V1)
```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)
(handler-case
    (mini-array.indexing:boolean-select (asarray '(1 2 3)) '(nil nil nil))
  (error (e) (format t "error: ~a~%" e)))
;; typical message: boolean-select: all-false MASK would yield empty selection, disallowed in V1
```

## Examples
End-to-end snippet from the repository root using the integration loader:

```lisp
(load "numpy-speedrun-poc/src/mini-array.lisp")
(in-package :mini-array)

;; Start with a 1D array, select odds, then add 10 to each (demonstrates interop)
(let* ((v    (asarray '(1 2 3 4 5 6)))
       (mask '(t nil t nil t nil))
       (sel  (mini-array.indexing:boolean-select v mask))
       (out  (add sel 10)))
  (to-list out))
;; => (11.0d0 13.0d0 15.0d0)
```

## See also
- Tensor construction and indexing helpers: docs/modules/tensor.md; tests: tests/tensor-tests.lisp
- Core shape/dtype helpers: docs/modules/core.md; tests: tests/core-tests.lisp and src/core/tests.lisp
- Broadcasting helpers (not used by boolean-select): docs/modules/broadcast.md; tests: tests/broadcast-tests.lisp
- Elementwise ufuncs (add, mul): docs/modules/ufunc.md; tests: tests/ufunc-tests.lisp
- Reductions (sum-axis): docs/modules/reductions.md; tests: tests/reduce-tests.lisp
- Boolean mask demo script: examples/boolean-mask-demo.lisp
