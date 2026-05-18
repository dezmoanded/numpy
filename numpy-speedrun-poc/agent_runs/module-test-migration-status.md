# Module test migration — status update

Date: 2026-05-18

- Per-crawler budget increased to 0.6
- Respawning module crawlers: core, tensor, broadcast, ufunc, reductions
- Objectives per module:
  - Reconcile legacy flat-file tests with modular sources
  - Update/create module-scoped tests using file‑relative loaders (no src/mini-array.lisp)
  - Make tests pass on SBCL
  - Remove the corresponding legacy flat file once green
  - Keep integration test (tests/test-mini-array.lisp) green

Acceptance reminders (V1): float64 only; ranks 0–2; row‑major contiguous; positive strides; broadcasting; add/mul; sum‑axis for 2D axis∈{0,1}.

Planned test files:
- tests/core-tests.lisp (module-only loader)
- tests/tensor-tests.lisp (module-only loader)
- tests/broadcast-tests.lisp (module-only loader)
- tests/ops-tests.lisp (UFUNC numeric ops; module-only loader)
- tests/reduce-tests.lisp (module-only loader)

Legacy files slated for removal once module tests are green:
- src/core.lisp
- src/tensor.lisp
- src/ops.lisp
- src/reductions.lisp

This file will be updated as crawlers land changes and confirm green runs.