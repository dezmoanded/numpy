# 02-shape-strides-and-indexing

## 1. Summary

Extract the minimal row-major indexing model: compute default strides for a shape, map an integer index-tuple to a flat buffer offset using strides, and provide get/set via that mapping. Strides are element-based in this POC (NumPy uses byte strides). Scope: float64, 1D/2D, positive strides, contiguous row-major only.

## 2. Files/docs inspected

- numpy/doc/source/reference/arrays.ndarray.rst — ndarray basics; shape, strides, memory layout overview
- numpy/doc/source/user/basics.indexing.rst — conceptual rules of indexing; bounds and integer indexing semantics
- numpy/core/include/numpy/ndarraytypes.h — PyArrayObject fields; dimensions and strides stored as arrays (in bytes in NumPy)
- numpy/core/src/multiarray/ctors.c — array creation; computes default C-order (row-major) strides
- numpy/core/src/multiarray/mapping.c — index-to-offset logic for item access (general case; confirms byte-stride sum)

## 3. Key architectural ideas

- Strides define how to convert a logical multi-index into a byte offset in the underlying contiguous buffer.
  - In NumPy, strides are in bytes; default C-order strides are computed from the trailing dimension upward as cumulative products times itemsize.
  - Flat byte offset = sum_k (index[k] * byte_stride[k]).
- For a row-major contiguous 2D array with shape (R, C) and itemsize s:
  - byte_strides = (C * s, 1 * s). Example: shape (2,3), float64 (s=8) -> byte_strides = (24, 8).
- Bounds and rank checks are enforced before offset computation.
- General NumPy extends far beyond this POC (negative/zero strides, views, non-contiguous, Fortran-order, suboffsets, advanced/fancy/boolean indexing). We explicitly ignore these in the first pass.
- Clean-room simplification for the POC:
  - Store strides in element units (not bytes) because dtype is fixed (float64); itemsize is a constant 8 internally.
  - default_strides(shape) uses cumulative products of subsequent dimensions; flat_offset uses sum(i_k * stride_k).

## 4. Minimal behavior to port

- default_strides(shape)

```
# shape: list<int> of length n (n in {1,2})
# returns element-based strides (not bytes), C-order (row-major)
default_strides(shape):
    n = len(shape)
    if n == 0:
        return []  # scalar (not used in this POC)
    strides = [0] * n
    stride = 1
    for k in reversed(range(n)):
        strides[k] = stride
        stride *= shape[k]
    return strides
```

- flat_offset(indices, shape, strides)

```
# indices: list<int> length n
# shape: list<int> length n
# strides: list<int> length n (element-based)
flat_offset(indices, shape, strides):
    n = len(shape)
    assert len(indices) == n
    off = 0
    for k in range(n):
        i = indices[k]
        # POC: only non-negative indices
        if i < 0 or i >= shape[k]:
            raise IndexError
        off += i * strides[k]
    return off
```

- get(array, indices)

```
# array has fields: shape, strides (element-based), data (flat float64 list)
get(array, indices):
    off = flat_offset(indices, array.shape, array.strides)
    return array.data[off]
```

- set(array, indices, value)

```
set(array, indices, value):
    off = flat_offset(indices, array.shape, array.strides)
    array.data[off] = value  # float64
```

Notes:
- For 2D, the above reduces to: off = i * strides[0] + j * strides[1] with strides = (cols, 1).
- Because strides are element-based, the byte offset would be off * 8 if needed.

## 5. Explicit exclusions

- Negative indices, slices, Ellipsis, None/newaxis, boolean/fancy indexing, multi-field dtypes.
- Non-contiguous views, broadcasting via zero strides, subviews with offsets, negative or zero strides, Fortran-order arrays.
- Dtype variability and itemsize handling (fixed to float64, 8 bytes).
- Overlapping memory and aliasing concerns.

## 6. Suggested tests

- Default strides
  - shape (3,) -> strides (1,)
  - shape (2,3) -> strides (3,1)
- Flat offset and get/set
  - a = array(shape=(2,3), strides=(3,1), data=[0,1,2,3,4,5])
    - flat_offset([0,0]) == 0; get(a,[0,0]) == 0
    - flat_offset([0,2]) == 2; get(a,[0,2]) == 2
    - flat_offset([1,0]) == 3; get(a,[1,0]) == 3
    - flat_offset([1,2]) == 5; get(a,[1,2]) == 5
    - set(a,[0,1], 9.0); then get(a,[0,1]) == 9.0 and a.data == [0,9,2,3,4,5]
- 1D convenience
  - b = array(shape=(4,), strides=(1,), data=[10,11,12,13])
    - get(b,[2]) == 12; set(b,[1], 7.0) -> b.data == [10,7,12,13]
- Errors
  - get(a,[2,0]) raises IndexError
  - get(a,[-1,0]) raises IndexError (negative indices excluded in POC)
  - get(a,[0]) or get(a,[0,0,0]) raises error (rank mismatch)

## 7. Open questions

- Stride units: Keep element-based for simplicity now; if/when multiple dtypes are added, should we switch to byte-based strides to mirror NumPy? Migration path seems straightforward (multiply/divide by itemsize).
- Zero-sized dimensions: Support constructing arrays with a zero in shape? Safe for strides, but indexable elements do not exist; okay to allow but all indexing should fail. Worth allowing now or defer?
- Scalar (ndim=0) handling: Out of immediate scope; confirm later whether to represent as shape=() with empty strides or a distinct scalar type.
