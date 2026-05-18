# CORE module contract

Responsibilities
- Provide minimal, dependency-free utilities for shapes/strides and dtypes.
- Establish global invariants used by downstream modules (tensor, broadcast, ufunc, reductions, indexing).

Public API (package :mini-array.core)
- product(shape-list) -> integer; NIL -> 1
- rank(shape-list) -> 0..2; NIL -> 0
- valid-shape-p(shape-list) -> T iff proper list, rank<=2, all dims positive integers (>0)
- default-strides(shape-list) -> row-major element strides; NIL, (1), (c 1) for ranks 0,1,2 respectively
- dtype-p(x) -> T iff x in {:float64, :bool}
- coerce-dtype-value(value, dtype) -> coerced value or error
  - :float64: accept any real number; return double-float
  - :bool: accept only T/NIL; return as-is; do not coerce numbers
- infer-scalar-dtype(value) -> :bool for T/NIL; :float64 for real numbers; error otherwise

Dependencies
- None. This module must not depend on any other project module.

Invariants
- Rank limited to {0,1,2} across the project.
- Zero-sized dimensions disallowed.
- Strides are element-based and row-major for contiguous arrays.
- Supported dtypes are exactly :float64 and :bool.
- Booleans are strict CL booleans: T or NIL.

Exclusions
- No support for higher ranks, negative/zero strides, zero-sized dims, or additional dtypes.
- No boolean coercion from numbers.
- No complex numbers in :float64 coercion (complex is rejected).

Errors
- default-strides: signals condition if SHAPE invalid or rank>2.
- coerce-dtype-value: signals when type/shape rules violated.
- infer-scalar-dtype: signals on unsupported scalars (e.g., strings, complex).

Notes
- Numbers are coerced to double-float via CL COERCE; ratios and integers are accepted.
- NaN/Inf are permitted as floating values as per host CL.
