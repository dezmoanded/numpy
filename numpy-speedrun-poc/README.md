numpy-speedrun-poc — Tiny clean-room Lisp numerical core

Purpose (agent-assisted "library speedrun")
- Demonstrate that agents can study a mature library (NumPy), extract a small architectural slice, and stand up a faithful clean‑room port quickly.
- Deliver a minimal, reviewable Common Lisp core with narrow, self-contained tests and small demos. No upstream code is copied.

Current scope and invariants (V1)
- Dtypes
  - :float64 for all numeric computation and storage
  - :bool for masks only (T/NIL); all-false selections are disallowed in V1
- Ranks: 0D/1D/2D only
- Layout: row‑major contiguous; positive, element‑based strides
- Semantics
  - NumPy‑style right‑aligned broadcasting
  - Elementwise add/mul via a tiny ufunc engine
  - sum over a single axis for 2D (axis ∈ {0,1})
- Exclusions: dtype promotion/casting, non‑contiguous views, slicing/advanced indexing (beyond provided boolean select), masked/object arrays, BLAS/SIMD, negative/zero strides, >2D, keepdims, axis tuples, NaN special handling

Architecture (modular, integration loader)
- Top-level integration loader: src/mini-array.lisp
  - Loads modular packages/implementations in dependency order
  - Maintains a legacy fallback while migration completes
- Modular packages and implementations (source-of-truth)
  - src/core/
    - package.lisp — package and exports for core helpers
    - shape.lisp — rank/product/valid-shape/default-strides
    - dtype.lisp — %ensure-double, %make-double-vector
  - src/tensor/
    - package.lisp — package and exports for ndarray/tensor helpers
    - ndarray.lisp — defstruct ndarray(shape strides dtype data)
    - indexing.lisp — flat-offset, aref-nd, %set-aref-nd
    - conversion.lisp — asarray, make-array-from-flat, to-list, shape-of, strides-of
  - src/broadcast/
    - package.lisp — package and exports for broadcasting
    - shape.lisp — broadcast-shape (right-aligned)
    - projection.lisp — project-broadcast-index
    - iterator.lisp — all-indices (over NIL/(n)/(r c))
  - src/ufunc/
    - package.lisp — package and exports for ufunc layer
    - engine.lisp — binary-ufunc (core iterator + projection)
    - numeric-ops.lisp — add, mul wrappers
  - src/reductions/
    - package.lisp — package and exports for reductions
    - sum.lisp — sum-axis for 2D, axis∈{0,1}
  - src/indexing/
    - package.lisp — package and exports for indexing
    - boolean-select.lisp — 1D boolean mask selection; errors if mask all-false or invalid
- Legacy flat files (kept temporarily for fallback during migration)
  - src/ops.lisp, src/reductions.lisp

Public API (via :mini-array integration package)
- Array and metadata: ndarray, asarray, make-array-from-flat, shape-of, strides-of, to-list
- Core helpers: product, rank, valid-shape-p, default-strides, flat-offset, aref-nd
- Broadcasting: broadcast-shape, all-indices, project-broadcast-index
- Ufuncs: binary-ufunc, add, mul
- Reductions: sum-axis (alias sum_axis retained temporarily)
- Indexing: boolean-select (mask rank=1; see policy above)

Development and test process
- Research → Plan → Implement
  - Research notes under agent_runs/*.md distilled NumPy behavior, scope, and invariants
  - Central interface/behavior guidance: agent_runs/central-interfaces-and-behavior.md
- CWD‑agnostic, modular test policy
  - Shared test utilities: tests/util.lisp
    - Path helpers: %here, %dir, %tests-root, %src-root
    - Loaders: load-modules (modular), load-integration (integration)
    - Assertions: assert-true, approx=, assert-approx=, expect-error
  - Module unit tests MUST NOT load src/mini-array.lisp. They load only their packages and files via load-modules
  - Integration tests and demos use load-integration
- Test suite
  - Orchestrator: run-all-tests.sh (runs module tests, then repo-level tests, then integration last)
  - Key tests
    - src/core/tests.lisp — core helpers
    - src/indexing/tests.lisp — boolean-select policy and errors
    - src/reductions/tests.lisp — sum-axis behavior
    - tests/core-tests.lisp — repo-level core checks
    - tests/tensor-tests.lisp — ndarray/asarray/indexing/flat-offset
    - tests/broadcast-tests.lisp — broadcast rules
    - tests/ufunc-tests.lisp — add/mul, broadcasting, scalar/vector cases
    - tests/reduce-tests.lisp — reductions from repo-level view
    - tests/integration-tests.lisp — composition checks (uses test util + integration loader)
    - tests/test-mini-array.lisp — legacy integration test (now uses test util)

Quickstart
- Run the full test suite
  cd numpy-speedrun-poc
  ./run-all-tests.sh
- Run a single test
  sbcl --script tests/integration-tests.lisp
  sbcl --script src/reductions/tests.lisp
- Run demos
  sbcl --script examples/tax-totals-demo.lisp
  sbcl --script examples/boolean-mask-demo.lisp

Behavioral policy highlights
- Broadcasting is right‑aligned; dimension pairs must match or one must be 1; otherwise signal an error
- sum-axis supports only 2D arrays with axis in {0,1}; errors on 0D/1D/invalid axis
- Indexing boolean-select requires a rank‑1 mask of the same length; all‑false masks are disallowed in V1 to avoid empty outputs
- Zero-sized dimensions are disallowed across V1 to keep implementations and tests small and predictable

Clean‑room principle
- We extract behavior, architecture, and invariants from NumPy tests/docs and re‑express them in our own code/pseudocode
- No NumPy implementation code is copied

Roadmap (near‑term)
- Keep module tests green under modular loader (tests/util.lisp:load-modules)
- Remove legacy src/ops.lisp and src/reductions.lisp once all narrow/module tests are green and covered by modular files
- Consider small additions: more ufuncs, column/row reductions on 2D, and improved error messaging/diagnostics
