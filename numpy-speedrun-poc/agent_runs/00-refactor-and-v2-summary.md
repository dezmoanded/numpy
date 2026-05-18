# integration-refactor-and-v2-summary

## 1. Summary

Implemented the V2 integration entrypoint and artifacts:
- Dependency-ordered, idempotent loader at src/mini-array.lisp with legacy fallback
- Integration test script that only loads the integration loader
- Two runnable examples demonstrating broadcasting arithmetic and boolean masking
- Top-level README describing usage, scope, and test/example commands

Package re-exports in src/package.lisp are left as-is (legacy) until all submodule packages are in place to avoid breaking current unit tests.

## 2. Files/docs inspected

- src/package.lisp — existing top-level exports; informs re-export surface
- src/core.lisp, src/tensor.lisp, src/ops.lisp, src/reductions.lisp — legacy flat modules used for fallback
- src/indexing/package.lisp — imports from :mini-array; informs loader ordering for indexing
- AGENT.md and central-interfaces-and-behavior.md — integration/testing policy and scope

## 3. Key architectural ideas

- Single integration entrypoint that can operate in two modes:
  1) Modular: load per-module packages/impls in a strict DAG order
  2) Legacy fallback: load flat files when modular packages are not yet present
- Always finish by loading the top-level package so :mini-array is available
- Defer modules that import from :mini-array (e.g., indexing) until after top-level package is present

## 4. Minimal behavior to port

- No functional logic added at integration; only load ordering and safe fallback
- Tests validate end-to-end add/mul, broadcasting, and sum-axis via a single entrypoint
- Examples exercise typical integration usage without touching submodule loaders directly

## 5. Implementation handoff

Data/Files created or updated
- src/mini-array.lisp — integration loader (DAG + legacy fallback + deferred indexing)
- tests/integration-tests.lisp — end-to-end checks: scalar+vector add, 1x3 + 2x1 broadcasting, sum-axis along both axes, composed pipeline
- examples/tax-totals-demo.lisp — scalar and per-item tax broadcasting
- examples/boolean-mask-demo.lisp — boolean-select with error on all-false
- README.md — how to run integration tests and examples; scope and notes

Loader algorithm
- Try modular files in this order: core → tensor → broadcast → ufunc → reductions
- Then load top-level src/package.lisp
- Then load indexing (imports from :mini-array)
- If :mini-array.tensor package is absent after step 1, load legacy flat files instead (core.lisp, tensor.lisp, ops.lisp, reductions.lisp) before top-level package

## 6. Explicit exclusions
- No changes to legacy functional code
- No re-export consolidation until all submodule packages exist
- No additional ops/features beyond those already provided by legacy modules

## 7. Suggested tests
- Keep existing unit tests per module (do not load src/mini-array.lisp)
- Integration script must only load src/mini-array.lisp and assert:
  - add(asarray '(1 2 3), 10) → '(11 12 13)
  - add '((1 2 3)) and '((10) (20)) → '((11 12 13) (21 22 23))
  - sum-axis '((1 2 3) (4 5 6)) axis 0 → '(5 7 9); axis 1 → '(6 15)
  - composition sum-axis(add(A,B),1) with A,B 2x2 → '(8 22)

## 8. Open questions
- When should src/package.lisp be converted to a re-export surface that imports from submodule packages? Proposed: after core/tensor/broadcast/ufunc/reductions packages are implemented and unit tests are green.
- Should integration also expose indexing via top-level re-exports in V2, or keep it namespaced under :mini-array.indexing until dtype/shape rules are stabilized?
