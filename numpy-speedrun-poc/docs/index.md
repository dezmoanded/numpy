# Mini Array (numpy-speedrun-poc)

Tiny clean‑room numerical core in Common Lisp focused on clarity, portability, and testable NumPy‑style semantics.

Audience: developers building, extending, or embedding a minimal array runtime with broadcasting, elementwise ops, and simple reductions.

## Key properties (V1)
- Dtypes: :float64 for numbers; :bool for masks (T/NIL).
- Shapes: ranks 0D/1D/2D only; row‑major contiguous; positive, element‑based strides; zero‑sized dims disallowed.
- Semantics: NumPy‑style right‑aligned broadcasting; elementwise add/mul; sum along a single axis for 2D (axis ∈ {0,1}).

## Quick start
1) Requirements: SBCL (or another ANSI Common Lisp implementation)
2) Load the integration package from repo root:

   (load "numpy-speedrun-poc/src/mini-array.lisp")
   (use-package :mini-array)

3) First array and operation:

   (defparameter *a* (asarray '((1 2 3)
                                (4 5 6))))
   (defparameter *b* (asarray '(10 20 30)))
   (to-list (add *a* *b*))
   ;; => '((11.0d0 22.0d0 33.0d0)
   ;;      (14.0d0 25.0d0 36.0d0))

## Core concepts
- ndarray: a simple struct with shape, strides, dtype, and a contiguous double‑float data vector.
- Broadcasting (right‑aligned): a dim pair is compatible when equal or one is 1; otherwise signal an error.
- Ufunc iteration: iterate over the broadcasted output shape; project indices back to inputs.
- Reductions (V1): only sum along axis 0 or 1 for 2D inputs.

## API at a glance
- Array construction and metadata
  - asarray x → ndarray (scalar, 1D list, or 2D rectangular list)
  - make-array-from-flat shape flat → ndarray
  - shape-of a → shape | strides-of a → strides
  - to-list a → Lisp scalar/list form
- Core helpers
  - product shape → integer; rank shape → 0/1/2
  - valid-shape-p shape → boolean (no zero‑sized dims; rank ≤ 2)
  - default-strides shape → NIL | (1) | (c 1)
  - flat-offset index strides [shape] → integer (bounds‑checks if shape provided)
  - aref-nd a index → float64 | set via internal %set-aref-nd
- Broadcasting/ufuncs
  - broadcast-shape s1 s2 → shape
  - all-indices shape → list of indices
  - project-broadcast-index out-idx in-shape out-shape → in-idx
  - add a b → ndarray (elementwise)
  - mul a b → ndarray (elementwise)
  - binary-ufunc a b op → ndarray (for custom ops over two inputs)
- Reductions
  - sum-axis a axis → ndarray (1D result)
- Indexing (V1 policy)
  - boolean-select array mask → ndarray (mask rank=1; same length; all‑false disallowed)

## Module reference
- Core: modules/core.md
- Tensor: modules/tensor.md
- Broadcast: modules/broadcast.md
- Ufuncs: modules/ufunc.md
- Reductions: modules/reductions.md
- Indexing: modules/indexing.md

## Clean examples
- Scalars, vectors, matrices

  (to-list (asarray 7))
  ;; => 7.0d0

  (to-list (asarray '(1 2 3)))
  ;; => '(1.0d0 2.0d0 3.0d0)

  (to-list (asarray '((1 2) (3 4))))
  ;; => '((1.0d0 2.0d0)
  ;;      (3.0d0 4.0d0))

- Broadcasting add

  (to-list (add (asarray '((1 2 3)))    ; 1x3
                (asarray '((10) (20))))) ; 2x1
  ;; => '((11.0d0 12.0d0 13.0d0)
  ;;      (21.0d0 22.0d0 23.0d0))

- Elementwise multiply with scalar

  (to-list (mul (asarray '((1 2 3)
                           (4 5 6)))
                2.5d0))
  ;; => '((2.5d0 5.0d0 7.5d0)
  ;;      (10.0d0 12.5d0 15.0d0))

- Composition: totals with taxes, then row sums

  (let* ((prices (asarray '((10 20 30)
                            (40 50 60))))
         (tax    (asarray '(0.07 0.08 0.09)))
         (totals (mul prices (add 1.0d0 tax)))
         (row    (sum-axis totals 1)))
    (list (to-list totals)
          (to-list row)))
  ;; => '(((10.7d0 21.6d0 32.7d0)
  ;;       (42.8d0 54.0d0 65.4d0))
  ;;      (65.0d0 162.2d0))

- Boolean select (V1)

  (let* ((v (asarray '(10 20 30 40)))
         (m (asarray '(t nil t nil)))
         (sel (boolean-select v m)))
    (to-list sel))
  ;; => '(10.0d0 30.0d0)
  ;; All‑false masks signal an error in V1.

## Error behavior (selected)
- Broadcasting incompatibility → error
- sum-axis on rank≠2 or axis∉{0,1} → error
- Ragged 2D inputs to asarray → error
- Zero‑sized dims (anywhere) → error
- boolean-select with non‑rank‑1 mask, length mismatch, non‑booleans, or all‑false → error

## Project structure
- src/mini-array.lisp — integration loader (modules first; temporary fallback to legacy flat files)
- src/core/*, src/tensor/*, src/broadcast/*, src/ufunc/*, src/reductions/*, src/indexing/* — modular source of truth
- tests/util.lisp — shared test utilities (loaders, assertions)
- tests/* and src/*/tests.lisp — module and integration tests
- examples/* — small runnable demos

## Develop and test
- Run everything

  cd numpy-speedrun-poc
  ./run-all-tests.sh

- Run a specific test or demo

  sbcl --script tests/integration-tests.lisp
  sbcl --script src/reductions/tests.lisp
  sbcl --script examples/tax-totals-demo.lisp

## Roadmap (near‑term)
- Keep module tests green using the modular loader
- Remove legacy src/ops.lisp and src/reductions.lisp once covered by modular files
- Consider additional ufuncs and reductions while preserving the clean, minimal surface

## Clean‑room policy
Behavior is derived from NumPy docs and tests; implementation is original and intentionally small. No NumPy implementation code is copied.
