# 00-implementation-summary

## What was implemented
A tiny clean-room Common Lisp numerical core aligned with the extracted NumPy architecture (V1 scope):
- ndarray struct: shape (NIL | (n) | (r c)), element-based strides (NIL | (1) | (c 1)), dtype :float64, data simple-array of double-float
- Array construction and validation (make-array-from-flat, asarray)
- Introspection/materialization (shape-of, strides-of, to-list)
- Indexing helpers (product, default-strides, flat-offset, aref-nd)
- Broadcasting + iteration (broadcast-shape, all-indices, project-broadcast-index)
- Elementwise ufunc engine and ops (binary-ufunc, add, mul)
- Reductions (sum-axis for 2D, axis 0 and 1)

## Mapping to NumPy architectural extraction
- Shapes/strides follow row-major contiguous rules; strides are element-based (Section 02, 04)
- Broadcasting uses right-aligned comparison; 1 expands; scalars broadcast to any shape (Section 03)
- Elementwise iteration is output-index driven with projection back to inputs (Section 04, 05)
- Reductions use nested loops over 2D arrays; no keepdims/axis tuples (Section 06)
- Errors: ValueError/simple-error on broadcast mismatch, invalid axis, rank>2, zero-sized dims (Section 00 plan)

## Files created
- src/package.lisp — package and exports (:mini-array)
- src/mini-array.lisp — implementation
- tests/test-mini-array.lisp — tests (no external deps)
- examples/tax-totals-demo.lisp — demo script
- README.md — run instructions and scope
- agent_runs/01-core-array-implementation.md
- agent_runs/02-broadcasting-implementation.md
- agent_runs/03-elementwise-ufunc-implementation.md
- agent_runs/04-reduction-implementation.md
- agent_runs/05-tests-and-demo.md

## How to run tests
If SBCL is available:
- cd numpy-speedrun-poc
- sbcl --script tests/test-mini-array.lisp

## Test results (local)
- No Common Lisp implementation was found in PATH in this environment (sbcl/clisp/ecl absent). Tests were not executed locally. The test file is designed to run under SBCL.

## How to run the demo
If SBCL is available:
- cd numpy-speedrun-poc
- sbcl --script examples/tax-totals-demo.lisp

## Demo output (expected)
Totals: ((10.7d0 21.6d0 32.7d0)
         (42.8d0 54.0d0 65.4d0))
Row totals: (65.0d0 162.2d0)

## Known limitations (intentional for V1)
- float64 only; ranks 0D/1D/2D only; contiguous row-major only
- No zero-sized dimensions; no slicing/views/advanced indexing
- No dtype promotion, out=, where=, or full ufunc dispatch
- No BLAS/SIMD optimizations

## Next steps (potential V2)
- Add 0-sized dims and general N-D shapes with contiguous views
- Implement basic slicing and negative strides
- Extend reductions (keepdims, other ops)
- Minimal dtype system and promotion rules
