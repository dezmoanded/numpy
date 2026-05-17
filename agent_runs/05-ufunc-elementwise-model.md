# 05-ufunc-elementwise-model

## 1. Summary

Goal: Model a minimal ufunc-style elementwise system: separate scalar kernels from array iteration, support add/multiply over float64, handle scalars and broadcasting, and allocate/stores results in flat row-major storage.

Conclusion: Implement a small binary_ufunc(op, a, b) that:
- Treats inputs as either scalars or ndarray-like records (float64 only).
- Computes the broadcasted output shape.
- Iterates over the output shape, projects each output index to each operand, applies a scalar op, and writes to a newly allocated contiguous float64 buffer.
- Exposes add(a, b) and mul(a, b) as thin wrappers over binary_ufunc.

## 2. Files/docs inspected

- doc/source/user/basics.ufuncs.rst — Conceptual definition: ufuncs are vectorized scalar loops with broadcasting; mentions one-dimensional strided inner loop and broadcasting behavior.
- doc/source/reference/ufuncs.rst — Reference: ufunc capabilities and options; confirms elementwise nature, broadcasting, and that many features (out, where, dtype/casting) exist but can be excluded for the POC.
- doc/source/user/basics.broadcasting.rst — Broadcasting rules relied upon by ufuncs; informs result-shape computation and how size-1 dimensions reuse values.

## 3. Key architectural ideas

- Vectorized scalar kernel: A ufunc is a thin array-iteration wrapper around a fixed-arity scalar function. For our POC, the scalar kernels are float64 add and multiply.
- Broadcasting drives iteration: Inputs are broadcast to a common output shape; any operand dimension equal to 1 (or missing) reuses the same scalar along that dimension.
- Strided view vs. index projection: NumPy implements a 1-D strided inner loop; broadcasting corresponds to stride 0 along broadcasted dimensions. For clarity/portability in the POC, we can instead compute per-operand indices from each output index (equivalent behavior without exposing stride-0 explicitly).
- Output allocation: The result shape is the broadcasted shape; allocate a new contiguous row-major float64 buffer. Scalars broadcast to any shape.
- Error on incompatibility: If shapes cannot broadcast, raise an error before iteration.

## 4. Minimal behavior to port

Data model assumptions (from overall POC):
- float64 only; arrays are row-major contiguous; positive strides; support scalars, 1D, 2D.

Required API:
- binary_ufunc(op, a, b)
- add(a, b) = binary_ufunc((x,y)->x+y, a, b)
- mul(a, b) = binary_ufunc((x,y)->x*y, a, b)

Helpers expected to exist (from sibling slices):
- broadcast_shape(sa, sb)
- all_indices(shape) — yields every multi-index in row-major order
- project_broadcast_index(out_idx, in_shape, out_shape)
- flat_offset(idx, strides) and default_strides(shape) for contiguous arrays

Clean-room pseudocode:

function as_array(x):
  if x is scalar(float64):
    return {shape: (), strides: (), data: [x]}  ; 0-d array
  else:
    return x  ; assume already ndarray record

function binary_ufunc(op, a, b):
  A = as_array(a)
  B = as_array(b)
  out_shape = broadcast_shape(A.shape, B.shape)  ; raises on incompatibility
  out_strides = default_strides(out_shape)
  out_data = new float64[product(out_shape)]

  for idx in all_indices(out_shape):
    a_idx = project_broadcast_index(idx, A.shape, out_shape)
    b_idx = project_broadcast_index(idx, B.shape, out_shape)
    av = A.data[flat_offset(a_idx, A.strides)]
    bv = B.data[flat_offset(b_idx, B.strides)]
    out_data[flat_offset(idx, out_strides)] = op(av, bv)

  return {shape: out_shape, strides: out_strides, dtype: float64, data: out_data}

function add(a, b):
  return binary_ufunc((x,y)->x+y, a, b)

function mul(a, b):
  return binary_ufunc((x,y)->x*y, a, b)

Edge/error cases to handle:
- Incompatible shapes → error before allocation/iteration.
- Either operand may be scalar () or array; both scalar → scalar result ().
- Empty dimensions (size 0) yield empty result with correct shape (no iteration).

## 5. Explicit exclusions

- Any dtype promotion/casting, mixed dtypes, integer/complex/bool.
- out=, where=, subok=, signature=, order=, casting=, and other ufunc kwargs.
- __array_ufunc__ overrides and dispatch to non-core types.
- Generalized ufuncs, reduce/accumulate/outer/at methods.
- Non-contiguous arrays, negative strides, views beyond simple contiguous buffers.
- Aliasing/overlap rules and in-place semantics.
- SIMD/backends/loop blocking/buffering.

## 6. Suggested tests

Assume helpers: array(shape, data) builds contiguous arrays; equality compares shape and elementwise values.

- 2D + 1D broadcasting
  - a = array((2,3), [1,2,3, 4,5,6])
  - b = array((3,),   [10,20,30])
  - add(a,b) → shape (2,3), data [11,22,33, 14,25,36]

- 2D * scalar
  - a = array((2,3), [1,2,3, 4,5,6])
  - s = 2.0
  - mul(a,s) → (2,3), [2,4,6, 8,10,12]

- scalar + 1D
  - s = 0.5; v = array((3,), [2,4,6])
  - add(s,v) → (3,), [2.5, 4.5, 6.5]

- 1D (length 1) with 2D (broadcasted along last dim)
  - a = array((2,3), [1,2,3, 4,5,6])
  - b = array((1,),   [10])
  - add(a,b) → (2,3), [11,12,13, 14,15,16]

- Incompatible shapes raise
  - a.shape=(2,3), b.shape=(2,) → error (cannot align trailing dims 3 vs 2)

- Empty dimension propagates
  - a.shape=(0,3), b.shape=(3,) → (0,3) with zero elements

Optional NumPy parity (illustrative):
- np.add(a,b) equals our add(a,b) for cases above.
- np.multiply(a,2.0) equals our mul(a,2.0).

## 7. Open questions

- Representing scalars internally as shape () 0-d arrays vs. a dedicated scalar tag: this note assumes shape () arrays; confirm consistency with the rest of the POC.
- Prefer stride-0 broadcasting vs. index projection: we chose index projection for clarity; if a stride-based inner loop is implemented later, ensure identical results and row-major iteration order.
- Handling size-0 dimensions: confirm that all_indices returns no indices and allocation still returns the correct empty buffer and shape.
