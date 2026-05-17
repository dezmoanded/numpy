# 03-elementwise-ufunc-implementation

## 1. Summary

Implement a tiny, NumPy-guided elementwise ufunc engine for add/mul over float64 with broadcasting for ranks 0–2. The engine separates the scalar kernel (x,y -> z) from array iteration. Iteration is output-shape–driven and uses right-aligned index projection to map each output index back to each input.

V1 stays intentionally small: float64 only; scalars, 1D, 2D; row-major contiguous; positive, element-based strides; broadcasting; add, mul; no zero-sized dimensions.

## 2. Files/docs inspected

- agent_runs/00-final-architecture-port-plan.md — Authoritative V1 scope, array model, allowed ops, and exclusions
- agent_runs/04-broadcasted-iteration.md — Output-driven iteration; project_broadcast_index contract
- agent_runs/05-ufunc-elementwise-model.md — Clean-room ufunc wrapper design and pseudocode
- agent_runs/07-numpy-parity-tests.md — Parity cases to target in tests/demo

Contextual references used by those notes (not re-crawled here):
- doc/source/user/basics.broadcasting.rst — Broadcasting rules
- doc/source/user/basics.ufuncs.rst — Elementwise ufunc behavior

## 3. Key architectural ideas

- Scalar kernel vs. array iteration
  - A ufunc is a thin iterator around a scalar function. Here, op ∈ {add, mul} over float64.
- Output-shape–driven broadcasting
  - Compute out_shape = broadcast_shape(a.shape, b.shape) by right-aligned comparison (dims equal or 1). Scalars () broadcast to any shape.
- Right-aligned index projection
  - For each output index oi, compute per-operand indices by project_broadcast_index(oi, in_shape, out_shape): if an input dim is 1 (or missing), clamp that index to 0; else copy oi’s component.
- Contiguous storage and element-based strides
  - V1 arrays are contiguous row-major, strides are element-based; default_strides(shape) = (n,1) for 2D and (1) for 1D; () for scalars.
- Errors early, simple inner loop
  - broadcast_shape raises ValueError on incompatibility; iteration assumes validity.

## 4. Minimal behavior to port

Assumptions (from 00 plan):
- dtype: float64 only
- ranks: 0D (), 1D (n), 2D (m,n)
- layout: row-major contiguous; positive element-based strides
- features: broadcasting; add; mul
- exclusions: zero-sized dims; >2D; out=/where=; mixed dtypes; non-contiguous/negative strides

Required API surface (this slice):
- as_array(x): float64 | ndarray -> ndarray (0D for scalars)
- binary_ufunc(op, a, b)
- add(a, b)
- mul(a, b)

Relied-on helpers (from sibling slices):
- product(shape) -> int
- default_strides(shape) -> list<int>
- flat_offset(idx, strides) -> int
- all_indices(shape) -> generator of tuples in row-major order (supports 0D/1D/2D)
- broadcast_shape(sa, sb) -> list<int> (raises ValueError on incompatibility)
- project_broadcast_index(oi, in_shape, out_shape) -> tuple for in_shape

Clean-room pseudocode

- as_array
  - Input: x (float64 or ndarray-record)
  - Output: ndarray-record
  - Behavior:
    - if type(x) is float64: return {shape: (), strides: (), dtype: 'float64, data: [x]}
    - else: return x (assumed ndarray-record per core array slice)

- binary_ufunc
  - Input: op: (float64,float64)->float64; a,b: float64|ndarray-record
  - Output: ndarray-record (new allocation)
  - Steps:
    1) A = as_array(a); B = as_array(b)
    2) out_shape = broadcast_shape(A.shape, B.shape)  // may raise ValueError
    3) out_strides = default_strides(out_shape)
    4) out_data = new float64[product(out_shape)]
    5) for oi in all_indices(out_shape):
         ai = project_broadcast_index(oi, A.shape, out_shape)
         bi = project_broadcast_index(oi, B.shape, out_shape)
         av = A.data[flat_offset(ai, A.strides)]
         bv = B.data[flat_offset(bi, B.strides)]
         out_data[flat_offset(oi, out_strides)] = op(av, bv)
    6) return {shape: out_shape, strides: out_strides, dtype: 'float64, data: out_data}

- add / mul
  - add(a,b) = binary_ufunc((x,y)->x+y, a, b)
  - mul(a,b) = binary_ufunc((x,y)->x*y, a, b)

Notes:
- 0D case: all_indices(()) yields one index (), so scalar-scalar returns a 0D array.
- For inputs with fewer dims, project_broadcast_index treats missing leading dims as 1 and clamps index to 0 as needed.

## 5. Implementation handoff

Data structures
- ndarray-record:
  - shape: list<int>  ; len in {0,1,2}
  - strides: list<int>; element-based; default_strides(shape)
  - dtype: 'float64
  - data: list<float64> of length product(shape)

Functions to implement in this slice
- as_array(x)
- binary_ufunc(op, a, b)
- add(a, b)
- mul(a, b)

Functions this slice depends on (must exist per other slices)
- product(shape), default_strides(shape), flat_offset(idx, strides)
- all_indices(shape)
- broadcast_shape(sa, sb)
- project_broadcast_index(oi, in_shape, out_shape)

Algorithmic details and contracts
- Input normalization: Always call as_array at ufunc boundary so scalars participate in broadcasting as shape ().
- Allocation: Always allocate a fresh output buffer sized product(out_shape). V1 has no in-place or out=.
- Iteration order: Row-major over out_shape to ease parity checks and future generalization.
- Projection: project_broadcast_index must right-align shapes with out_shape and clamp 1-sized dimensions to 0; assumes out_shape came from broadcast_shape.

Error handling
- broadcast_shape raises ValueError("operands could not be broadcast together with shapes {sa} {sb}") on first incompatible axis.
- V1 excludes zero-sized dimensions; array constructors should reject them so ufuncs never see them.
- Invalid types are not handled here; rely on asarray/construction layer to accept only float64 scalars or ndarray-records.

Simplifications allowed
- No out=, where=, signature, casting rules, or dtype promotion.
- Only float64; strides are element-based.
- Only ranks 0–2; non-contiguous/negative strides not supported.
- Use index projection; do not implement stride-0 optimization in V1.

Expected inputs/outputs
- Inputs: float64 scalars or ndarray-records obeying V1 invariants.
- Output: ndarray-record with shape = broadcast_shape(a.shape, b.shape), strides = default_strides(shape), dtype='float64.

## 6. Explicit exclusions
- Zero-sized dimensions
- Dtype variations/promotions; NaN-special rules
- out=/where=/kwargs; gufuncs; reduce/accumulate/outer/at
- >2D arrays; advanced indexing; slicing; views; negative strides; non-contiguous layouts
- In-place or aliasing semantics; SIMD/backends/loop buffering

## 7. Suggested tests

Assume helpers: array(shape, data), tolist, and NumPy for parity checks.

1) 2D + 1D (broadcast along last axis)
- a = array((2,3), [1,2,3, 4,5,6])
- b = array((3,),   [10,20,30])
- tolist(add(a,b)) == [[11,22,33],[14,25,36]]

2) 2D * scalar
- a = array((2,3), [1,2,3, 4,5,6])
- s = 2.0
- tolist(mul(a,s)) == [[2,4,6],[8,10,12]]

3) scalar + 1D
- s = 0.5; v = array((3,), [2,4,6])
- tolist(add(s,v)) == [2.5, 4.5, 6.5]

4) 2D + 1D of length 1 (broadcasted reuse)
- a = array((2,3), [1,2,3, 4,5,6])
- b = array((1,),   [10])
- tolist(add(a,b)) == [[11,12,13],[14,15,16]]

5) 2D + 2D with broadcasted second dim
- a = array((2,3), [1,2,3, 4,5,6])
- c = array((2,1), [10,20])
- tolist(add(a,c)) == [[11,12,13],[24,25,26]]

6) Incompatible shapes raise
- a.shape=(2,3), b.shape=(2,) -> ValueError

NumPy parity spot-checks
- np.allclose(np.array(add(a,b).tolist()), a_np + b_np) for the above where applicable.

## 8. Open questions
- None blocking V1. For V2+: consider stride-0 broadcasted inner loop for speed, and extend to >2D and zero-sized dims once core semantics stabilize.
