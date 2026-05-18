# core

## 1. Summary
Implemented the CORE module providing shape/stride helpers and dtype primitives with no dependencies. Exposes small, stable API for downstream modules.

## 2. Files/docs inspected
- agent_runs/central-interfaces-and-behavior.md — V1 scope and testing policy
- User module spec (this task) — exact CORE API and semantics

## 3. Key architectural ideas
- Shapes are small lists with rank<=2 and positive dims only.
- Strides are element-based and row-major for contiguous arrays: NIL, (1), (c 1).
- Dtypes are a closed set: :float64 and :bool. Booleans are strict T/NIL; numeric values are coerced to double-float.

## 4. Minimal behavior to port
- product: multiply dims; NIL->1.
- rank: length; NIL->0.
- valid-shape-p: list, rank<=2, each dim integer>0.
- default-strides: NIL->NIL; (n)->(1); (r c)->(c 1); error on invalid.
- dtype-p: only :float64 or :bool.
- coerce-dtype-value:
  - :float64: accept real numbers only; return double-float.
  - :bool: accept only T or NIL; no numeric coercion.
- infer-scalar-dtype: T/NIL -> :bool; real number -> :float64; else error.

## 5. Implementation handoff
Data structures
- None (helpers only).

Functions
- product(shape-list) -> integer
- rank(shape-list) -> integer 0..2
- valid-shape-p(shape-list) -> boolean
- default-strides(shape-list) -> shape-list
- dtype-p(x) -> boolean
- coerce-dtype-value(value, dtype) -> coerced scalar
- infer-scalar-dtype(value) -> dtype

Algorithm notes
- default-strides computes element steps: for 2D, leading stride is columns.
- coerce-dtype-value uses CL realp check and COERCE to 'double-float; rejects complex and non-numeric.

Error cases
- invalid shapes and rank>2 in default-strides.
- dtype not in set; wrong value type for a dtype; inference on unsupported types.

Simplifications
- Only ranks 0/1/2; no zero-sized dims.
- Only dtypes :float64 and :bool.

## 6. Explicit exclusions
- Additional dtypes; complex numbers for :float64; negative/zero strides; zero-sized dims.

## 7. Suggested tests
- product nil->1; '(2 3)->6
- default-strides nil, '(5)->'(1), '(2 3)->'(3 1); error on '(0)
- dtype-p truth table; coerce-dtype-value for numbers and booleans; errors for wrong types; infer-scalar-dtype happy and error paths.

## 8. Open questions
- Keep valid-shape-p strictly list-based or allow vectors as shapes later? Current choice: list only.
