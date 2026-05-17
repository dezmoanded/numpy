# 06-axis-reduction-model

## 1. Summary

Sum along a single axis is a ufunc-style reduction: fold add with identity 0.0 over the chosen axis, producing an array with that axis removed. For the POC: float64 only, 2D contiguous row-major arrays, axis in {0,1}. Implement with simple nested loops using flat indexing; no dtype promotion, keepdims, where, or out.

## 2. Files/docs inspected

- numpy/_core/_methods.py — shows _sum delegates to um.add.reduce (ufunc reduction), anchoring the conceptual model.
- doc/ufuncs.rst — documents ufunc.reduce semantics conceptually (reduction along an axis with an identity and associative op).
- numpy/core/fromnumeric.py — entry point layer for high-level reductions; indicates Python-level API dispatch (not needed to port but useful for provenance).

## 3. Key architectural ideas

- Reduction as ufunc.reduce: sum(a, axis=K) equals add.reduce(a, axis=K) conceptually.
- Identity and associativity: add has identity 0.0; fold order follows memory iteration but result is invariant under order for addition (ignoring FP roundoff).
- Shape rule: removing the reduced axis. For shape (M, N):
  - axis=0 → output shape (N)
  - axis=1 → output shape (M)
- Contiguity simplification: for row-major contiguous (strides = (N, 1)) flat indexing is index = i*N + j. No need for general strided iteration in the first POC.

## 4. Minimal behavior to port

Function: sum_axis(array, axis)
- Inputs:
  - array: float64, 2D, row-major contiguous, positive strides
  - axis: integer 0 or 1
- Output:
  - 1D float64 array with the reduced axis removed
- Errors:
  - raise on non-2D input, non-float64, non-contiguous, or axis not in {0,1}
- Semantics (pseudocode):

  - Let shape = (M, N); assume strides = (N, 1); flat data buffer length = M*N
  - If axis == 0:
    - out_shape = (N)
    - for j in 0..N-1:
      - acc = 0.0
      - for i in 0..M-1:
        - acc += data[i*N + j]
      - out[j] = acc
  - If axis == 1:
    - out_shape = (M)
    - for i in 0..M-1:
      - acc = 0.0
      - base = i*N
      - for j in 0..N-1:
        - acc += data[base + j]
      - out[i] = acc

Notes:
- Identity = 0.0. No special NaN handling beyond default IEEE add.
- For future generalization, this is equivalent to reduce(add, axis), but we do not implement the generic reducer in v1.

## 5. Explicit exclusions

- axis=None (full flatten reduction)
- multiple axes or tuples of axes
- keepdims parameter
- where/out/initial parameters
- dtype promotion and accumulator dtype choices
- non-contiguous arrays, negative strides, views
- dimensions other than exactly 2D
- integer/boolean/object dtypes
- empty dimensions and size-0 reductions

## 6. Suggested tests

Assume helper to make arrays: A(shape, data) yields row-major contiguous float64 arrays in the target runtime; NumPy shown for parity.

- Sum over axis=0 on 2x3
  - NumPy
    - a = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64)
    - np.sum(a, axis=0) -> array([5., 7., 9.])
  - Clean-room
    - sum_axis(A((2,3), [1,2,3,4,5,6]), axis=0) == A((3,), [5,7,9])

- Sum over axis=1 on 2x3
  - NumPy
    - a = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64)
    - np.sum(a, axis=1) -> array([6., 15.])
  - Clean-room
    - sum_axis(A((2,3), [1,2,3,4,5,6]), axis=1) == A((2,), [6,15])

- Composition sanity (optional parity spot-check)
  - NumPy
    - a = np.array([[1., 2., 3.], [4., 5., 6.]])
    - w = np.array([10., 1., -1.])
    - np.sum(a * w, axis=1) -> array([1*10+2*1+3*(-1), 4*10+5*1+6*(-1)]) = array([9., 39.])
  - Clean-room
    - sum_axis(mul(A((2,3), [1,2,3,4,5,6]), A((3,), [10,1,-1])), axis=1) == A((2,), [9,39])

- Error cases
  - axis not in {0,1} raises
  - non-2D input raises

## 7. Open questions

- Should we accept 1D input with axis=0 returning a scalar array in v1, or keep strictly 2D-only as above? Current plan: 2D-only.
- Do we want to explicitly check contiguity/strides in v1 or assume well-formed arrays from constructors?
- Empty-dimension behavior is excluded; if later needed, identity-only result semantics must be specified (e.g., sum over length-0 -> 0.0 per column/row).