# 00-final-architecture-port-plan

## 1) Scope and guardrails (V1)
- Dtype: float64 only
- Ranks: scalars (0D), 1D, 2D only
- Layout: row-major contiguous; positive element-based strides
- Features: broadcasting; elementwise add, multiply; sum(axis=0), sum(axis=1)
- Exclusions in V1: zero-sized dimensions; non-contiguous/negative strides; slicing/advanced indexing; dtype promotion; out/where/kwargs; >2D; keepdims; multiple axes
- Errors: use ValueError for broadcast incompatibility and invalid axis
- Scalars: represented internally as 0-D arrays with shape ()

## 2) Data model and invariants
Array record
- shape: list<int> (len 0, 1, or 2)
- strides: list<int> (element-based, same length as shape)
- dtype: symbol 'float64 (constant in V1)
- data: flat list<float64> of length product(shape)

Invariants
- len(shape) == len(strides)
- strides == default_strides(shape) for all constructed arrays in V1
- product(shape) == len(data)
- For 2D with shape (R,C): strides == (C, 1)

Helpers
- product(shape): multiply dims (empty shape () -> 1)
- default_strides(shape): C-order element-based cumulative products from the right
- flat_offset(indices, strides): sum(indices[i] * strides[i]) with bounds checks
- get(A, idx): A.data[flat_offset(idx, A.strides)]
- set(A, idx, v): A.data[flat_offset(idx, A.strides)] = v

## 3) Broadcasting semantics (binary)
Rules
- Compare dimensions from the right; pad the shorter rank with leading 1s
- At each axis: sizes are compatible if equal or one is 1; result size is max(a_i, b_i)
- Scalars shape () broadcast to any shape
- On first incompatible pair, raise ValueError("operands could not be broadcast together with shapes {sa} {sb}")

Pseudocode: broadcast_shape(sa, sb)
- See agent_runs/03-broadcasting-rules.md; adopt that algorithm verbatim in clean-room code

Examples
- (2,3) with (3,) -> (2,3)
- (2,3) with () -> (2,3)
- (3,1) with (1,4) -> (3,4)
- (2,3) with (4,) -> ValueError

## 4) Broadcasted iteration
Concept
- Drive iteration by the output shape; map each output index back to per-operand indices via right-aligned projection: if an input axis is 1 while the output axis >1, use index 0 for that axis; otherwise copy the output index

Primitives
- all_indices(shape): row-major index enumerator for 0D/1D/2D
- project_broadcast_index(out_idx, in_shape, out_shape): right-align and clamp 1-sized axes to 0

Pseudocode skeleton
- See agent_runs/04-broadcasted-iteration.md; implement index-projection variant (no stride-0 machinery in V1)

## 5) Elementwise ufuncs: add, mul
API
- binary_ufunc(op, a, b)
- add(a, b) = binary_ufunc((x,y)->x+y, a, b)
- mul(a, b) = binary_ufunc((x,y)->x*y, a, b)

Behavior
- as_array(x): if scalar float64, treat as 0-D array with shape () and strides ()
- out_shape = broadcast_shape(A.shape, B.shape) (raises on incompatibility)
- Allocate out with out_shape, strides = default_strides(out_shape), flat buffer of size product(out_shape)
- For each idx in all_indices(out_shape):
  - ai = project_broadcast_index(idx, A.shape, out_shape)
  - bi = project_broadcast_index(idx, B.shape, out_shape)
  - out[idx] = op(A[ai], B[bi])

Reference pseudocode
- See agent_runs/05-ufunc-elementwise-model.md; implement as written

## 6) Reductions: sum along axis
API
- sum_axis(A, axis) where axis in {0,1} and A is 2D

Behavior (2D only)
- shape (R,C), strides (C,1)
- axis=0: output shape (C); out[j] = sum_{i=0..R-1} A[i,j]
- axis=1: output shape (R); out[i] = sum_{j=0..C-1} A[i,j]
- Raise ValueError for invalid axis or non-2D in V1

Reference pseudocode
- See agent_runs/06-axis-reduction-model.md; implement nested-loop variant

## 7) Constructor and utilities for tests/demo
- array(shape, data): build contiguous array; validate product(shape) == len(data); set strides = default_strides(shape)
- asarray(py):
  - float -> {shape: (), strides: (), data: [float]}
  - 1D list -> shape (n), strides (1), data flatten as float64
  - 2D nested list -> shape (m,n), strides (n,1), row-major flatten
  - otherwise: ValueError
- shape(A) -> tuple(shape)
- strides(A) -> tuple(strides)
- tolist(A) -> nested lists per shape for parity checks

## 8) Test plan (parity against NumPy)
Implement pytest cases from agent_runs/07-numpy-parity-tests.md:
- add: 2D + 1D broadcasting; 1D + scalar; 2D + scalar
- mul: 2D * scalar; 2D * 1D
- compose: mul then sum_axis(axis=1)
- errors: incompatible broadcasting raises ValueError; invalid axis raises
- reductions: sum(axis=0) -> (C); sum(axis=1) -> (R)

## 9) Implementation order
1) Core structs and constructors: array, asarray, default_strides, product, validations
2) Indexing: flat_offset, get, set, all_indices
3) Broadcasting: broadcast_shape, project_broadcast_index
4) Elementwise engine: binary_ufunc; add, mul wrappers
5) Reduction: sum_axis for 2D
6) Utilities: shape, strides, tolist
7) Test harness and parity tests

## 10) Explicit exclusions to keep V1 tiny
- Zero-sized dimensions (defer); any dimension must be >0
- Non-contiguous views, negative strides, slicing/advanced indexing
- Dtype variations and promotion; NaN-special rules
- out=, where=, signature, casting, keepdims, axis tuples, axis=None
- >2D arrays; reductions beyond sum over a single axis
- Performance optimizations (stride-0, blocking, SIMD, threading)

## 11) Key references consulted (repo-relative)
- doc/source/user/basics.broadcasting.rst — authoritative broadcasting rules and examples
- doc/source/user/basics.ufuncs.rst — elementwise ufunc semantics
- doc/source/reference/ufuncs.rst — reference overview
- doc/source/reference/arrays.ndarray.rst — ndarray shape/strides/memory layout
- numpy/_core/_methods.py — reduction via ufunc.add.reduce (conceptual anchor)

## 12) Final decisions and open follow-ups
Decisions for V1
- Error type: ValueError for broadcast incompatibility and invalid axis
- Scalars: Represent as 0-D arrays with shape () and strides ()
- Strides: element-based (not bytes) since dtype is fixed
- Zero-sized dimensions: excluded in V1 for simplicity (iteration and allocation remain well-defined when added later)
- Broadcasting: pairwise helper broadcast_shape only (N-ary can be layered later)

Open follow-ups (V2+)
- Add zero-sized dimensions semantics and tests (empty iteration, identity-only reductions)
- Consider stride-0 inner-loop optimization while preserving semantics
- Extend to >2D by generalizing all_indices and projection
- Introduce additional ops (sub, div) and reductions (mean) as thin wrappers
- Evaluate switching strides to bytes if/when multiple dtypes land
