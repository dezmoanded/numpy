numpy-speedrun-poc — Tiny clean-room Lisp numerical core (V1)

What this POC demonstrates (agent-assisted "library speedrun")
- Agents can study a mature library (NumPy), extract a very small architectural slice, and stand up a faithful clean‑room port quickly.
- Output is a minimal, reviewable Common Lisp core with narrow, self-contained tests and a small demo. No upstream code is copied.

Current V1 scope (frozen)
- dtype: float64 only
- ranks: 0D, 1D, 2D
- layout: row‑major contiguous; element‑based positive strides
- semantics: NumPy‑style broadcasting (right‑aligned), elementwise add/mul, sum over a single axis for 2D (axis ∈ {0,1})
- exclusions: dtype promotion, non‑contiguous views, slicing/advanced indexing, masked/object arrays, BLAS/SIMD, negative/zero strides, >2D, keepdims, axis tuples

Module layout (finalized)
- src/package.lisp — package and exports (:mini-array)
- src/core.lisp — core helpers
  - %ensure-double, %make-double-vector
  - product, rank, valid-shape-p, default-strides
- src/tensor.lisp — ndarray model and indexing
  - defstruct ndarray(shape, strides, dtype :float64, data)
  - make-array-from-flat, asarray, shape-of, strides-of
  - flat-offset, aref-nd, %set-aref-nd, to-list
- src/ops.lisp — broadcasting + elementwise ufuncs
  - broadcast-shape, all-indices, project-broadcast-index
  - binary-ufunc, add, mul
  - self-loads core.lisp and tensor.lisp via dir‑relative eval‑when
- src/reductions.lisp — reductions (2D only)
  - sum-axis (axis ∈ {0,1}); compatibility alias sum_axis
  - self-loads core.lisp and tensor.lisp via dir‑relative eval‑when
- src/mini-array.lisp — thin integration loader only (loads: core → tensor → ops → reductions)

Stable public API (exports)
- ndarray, make-array-from-flat, asarray
- product, rank, valid-shape-p, default-strides
- shape-of, strides-of, flat-offset, aref-nd, to-list
- broadcast-shape, all-indices, project-broadcast-index
- binary-ufunc, add, mul
- sum-axis, sum_axis (temporary alias)

Testing policy (important)
- Subgroup tests must NOT load src/mini-array.lisp.
- Each subgroup test loads src/package.lisp plus its single src file:
  - tests/core-tests.lisp → load package.lisp, then core.lisp
  - tests/tensor-tests.lisp → load package.lisp, then tensor.lisp
  - tests/ops-tests.lisp → load package.lisp, then ops.lisp
  - tests/reduce-tests.lisp → load package.lisp, then reductions.lisp
- Integration test may load src/mini-array.lisp:
  - tests/test-mini-array.lisp

Quickstart: run tests (SBCL)
- From repo root:
  cd numpy-speedrun-poc
  sbcl --script tests/core-tests.lisp
  sbcl --script tests/tensor-tests.lisp
  sbcl --script tests/ops-tests.lisp
  sbcl --script tests/reduce-tests.lisp
  sbcl --script tests/test-mini-array.lisp   ; integration

Demo
- A tiny composition example (e.g., prices × (1+tax) → row totals):
  - examples/tax-totals-demo.lisp
  - Run: sbcl --script examples/tax-totals-demo.lisp

Process we used (Research → Plan → Implement)
1) Research with sub‑crawlers (agent_runs/*.md)
  - Narrow assignments: ndarray metadata; shape/strides; broadcasting; broadcasted iteration; add/mul; 2D sum(axis); parity tests; IR/port plan.
  - Strict exclusions and small surface area to keep the port clean and testable.
2) Synthesis plan (agent_runs/00-final-architecture-port-plan.md and central-interfaces-and-behavior.md)
  - Locked invariants: float64 only, rank ≤ 2, row‑major contiguous, positive strides, right‑aligned broadcasting.
  - Minimal API and pseudocode so implementation didn’t require re‑crawling.
3) Clean‑room implementation (src/*) + tests (tests/*) + demo (examples/*)
  - ops/reductions self‑load core/tensor; mini‑array is integration‑only.

Notes
- Clean‑room: behavior is derived from NumPy docs/tests; no NumPy implementation code is copied.
- For further slices (e.g., dtype promotion, >2D, additional ufuncs/reductions), repeat the Research → Plan → Implement loop and extend the core incrementally.
