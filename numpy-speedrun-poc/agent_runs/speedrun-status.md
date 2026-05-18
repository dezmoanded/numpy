# Speedrun status — latest

- Date: 2026-05-18
- Environment: SBCL (non-interactive, noinform)

Results
- Tests: sbcl --noinform --non-interactive --script numpy-speedrun-poc/tests/test-mini-array.lisp → exit 0
- Demo: sbcl --noinform --non-interactive --script numpy-speedrun-poc/examples/tax-totals-demo.lisp → exit 0

Key patches in this pass
- src/broadcast/projection.lisp: rename local `pi` → `proj-elt` (avoid CL:PI collision)
- src/tensor/conversion.lisp: asarray now idempotent for ndarray; errors on dtype mismatch
- examples/tax-totals-demo.lisp: file-relative loader and asarray-based 2D input

Loader check
- src/mini-array.lisp loads ufunc/package.lisp, ufunc/engine.lisp, ufunc/numeric-ops.lisp; add/mul available via modular path

Notes
- Runner did not capture stdout text, but both scripts completed without errors
- If CI needs explicit success prints, add a final write-line in the test harness