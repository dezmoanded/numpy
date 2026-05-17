# 04-broadcasted-iteration

## 1. Summary

Goal: Describe how to iterate elementwise over two operands using NumPy-style broadcasting, mapping each output index back to each input. Core insight: align shapes from the right, and for any input dimension equal to 1, reuse the same element along that axis (conceptually zero stride). This iteration is reusable for any scalar binary op.

## 2. Files/docs inspected

- doc/source/user/basics.broadcasting.rst — User-facing definition of broadcasting rules (align-from-right, 1 or equal, scalar broadcasting).
- doc/neps/nep-0010-new-iterator-ufunc.rst — Iterator design and “broadcast strides = 0 for size-1 dims” examples; conceptual basis for mapping indices/strides.
- numpy/__init__.cython-30.pxd — Mentions PyArrayMultiIter* and PyArray_Broadcast; evidence of a dedicated broadcast multi-iterator abstraction (conceptual reference only).

## 3. Key architectural ideas

- Output-driven iteration
  - Compute result_shape via broadcasting rules (right alignment; dims compatible if equal or 1; missing treated as 1).
  - Iterate over all indices of result_shape; map each output index to each input index.
- Right-aligned projection
  - Let out_nd = len(result_shape). For an input with shape in_shape, align from the right. If a given aligned input dim is 1 while the output dim is k>1, the input index for that dim is 0 (reused value).
- Zero-stride view (conceptual)
  - Equivalent to computing per-input broadcasted strides: stride=0 where in_dim==1 and out_dim>1; otherwise use the input’s actual stride. Then flat offsets are a dot-product of output indices with these broadcasted strides. For the POC’s contiguous, positive-stride 1D/2D float64 arrays, this reduces to simple index projection.
- Reusable across elementwise ops
  - The iteration skeleton is independent of the scalar op; plug in add/mul or any scalar kernel.

## 4. Minimal behavior to port

Assumptions: float64 only; scalars, 1D, 2D; row-major contiguous; positive strides.

- all_indices(shape)
  - Generate all N-dimensional indices in row-major order.
  - Pseudocode (handle 0D, 1D, 2D only):
    - if shape == () yield ()
    - if shape == (n,) yield (i) for i in [0..n-1]
    - if shape == (m,n) nested loops over i in [0..m-1], j in [0..n-1] yield (i,j)

- project_broadcast_index(output_index, in_shape, out_shape)
  - Right-align in_shape with out_shape. For each dim from the back:
    - if in_dim == out_dim: use output_index at that dim
    - elif in_dim == 1 and out_dim > 1: use 0
    - else: incompatible (shouldn’t happen if out_shape came from broadcast)
  - For fewer input dims, treat missing leading dims as 1.
  - Pseudocode:
    - let out_nd = len(out_shape); in_nd = len(in_shape)
    - for k in 0..out_nd-1 from the right:
      - o_k = output_index[out_nd-1-k]
      - i_dim = in_shape[in_nd-1-k] if k < in_nd else 1
      - if i_dim == out_shape[out_nd-1-k]: idx_k = o_k
      - elif i_dim == 1: idx_k = 0
      - else: error
    - return tuple of idx_k reversed back to input’s ndim length (drop any leading broadcasted 1s beyond in_nd)

- Optional stride-based flat offset (contiguous 1D/2D only)
  - default_strides(shape): for (n,) -> (1); for (m,n) -> (n,1); for scalar -> ()
  - broadcasted_strides(in_shape, out_shape): align from right; if in_dim==1 and out_dim>1 use 0; else copy in stride component.
  - flat_offset(idx, strides) = sum(idx[d] * strides[d])

- Elementwise iteration skeleton
  - Pseudocode:
    - out_shape = broadcast_shape(a.shape, b.shape)
    - allocate out with out_shape
    - for oi in all_indices(out_shape):
      - ai = project_broadcast_index(oi, a.shape, out_shape)
      - bi = project_broadcast_index(oi, b.shape, out_shape)
      - out[oi] = op(a[ai], b[bi])

Note: For 1D/2D contiguous arrays, a simpler flattened loop using broadcasted strides (with 0s on broadcasted axes) is equivalent but not strictly required here.

## 5. Explicit exclusions

- >2D arrays beyond noting the algorithm generalizes trivially by extending loops and right-alignment.
- Non-contiguous layouts, negative strides, transposed views, or memory aliasing concerns.
- Dtype promotion/dispatch; only float64 supported.
- ufunc C-level iterator APIs and advanced iterator flags (masking, buffering, external loops, etc.).

## 6. Suggested tests

Use tiny arrays; compare against NumPy for parity.

- 2D + 1D broadcast along last axis
  - a.shape=(2,3), b.shape=(3)
  - Expect out[i,j] = a[i,j] + b[j]
- 2D * scalar
  - a.shape=(2,3), s is scalar
  - Expect out = a * s
- 1D (n,)->(n) with 1D (1,)->scalar-like broadcast
  - a.shape=(3), b.shape=(1)
  - Expect out[j] = a[j] op b[0]
- Check repeated reads along broadcasted axis
  - a = [[1,2,3],[4,5,6]], b = [[10],[20]] (shapes (2,3) and (2,1))
  - out = a + b ⇒ [[11,12,13],[24,25,26]]

Sketch parity snippets (Python/NumPy):

```
import numpy as np

a = np.array([[1.,2.,3.],[4.,5.,6.]])
b = np.array([10.,20.,30.])
assert np.allclose(a + b, np.array([[11.,22.,33.],[14.,25.,36.]]))

s = 2.0
assert np.allclose(a * s, np.array([[2.,4.,6.],[8.,10.,12.]]))

b1 = np.array([100.])
assert np.allclose(a + b1, np.array([[101.,102.,103.],[104.,105.,106.]]))

c = np.array([[10.],[20.]])
assert np.allclose(a + c, np.array([[11.,12.,13.],[24.,25.,26.]]))
```

These validate both projection-by-index and the conceptual zero-stride reuse.

## 7. Open questions

- Scalars as shape () vs shape (1,): For the POC, treat Python/NumPy scalars as shape (), and arrays with shape (1,) as true 1D; ensure projection handles both consistently.
- Error propagation: Should project_broadcast_index raise, or should broadcast_shape catch incompatibility earlier? Proposed: broadcast_shape raises; projection assumes valid out_shape.
- For future >2D generalization: prefer index-projection or zero-stride flat iteration? Both are equivalent; zero-stride is faster but requires careful stride handling.
