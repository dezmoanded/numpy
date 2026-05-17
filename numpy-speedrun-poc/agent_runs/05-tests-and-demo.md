# 05-tests-and-demo

## Summary
Created self-contained Common Lisp tests (no external libs) and a small demo. Tests cover default strides, flat indexing, materialization, broadcasting, elementwise add/mul, composition, reductions, and error cases exactly as specified in the V1 scope.

## Files
- tests/test-mini-array.lisp — assertion helpers and scenarios; runnable via `sbcl --script tests/test-mini-array.lisp`
- examples/tax-totals-demo.lisp — prices/tax example; runnable via `sbcl --script examples/tax-totals-demo.lisp`
- README.md — usage and run instructions

## Notes
- Tests rely only on the public API exported by :mini-array.
- If no Lisp runtime is available (SBCL preferred), the files can be executed elsewhere with a Common Lisp implementation.
- Demo prints totals and row totals and echoes the expected values for visual parity.
