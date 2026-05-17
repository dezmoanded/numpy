# 01-ndarray-metadata

## 1. Summary

A NumPy ndarray is a view over a flat data buffer described by minimal metadata: shape (per-dimension lengths), strides (per-dimension step to move one index along that dimension), dtype (defines itemsize and interpretation), and a data pointer. For the POC we fix dtype=float64, require row-major contiguous layout, and use positive strides only. This yields a tiny, portable model where logical indices map deterministically to flat storage.

Proposed clean-room IR stores strides in element units (not bytes) to keep math small and readable, since itemsize is constant (8 for float64).

## 2. Files/docs inspected

- doc/source/reference/arrays.ndarray.rst — ndarray attributes (shape, strides, dtype, itemsize) and memory layout overview.
- doc/source/user/basics.types.rst — dtype concept; itemsize derives from dtype.
- doc/source/user/basics.creation.rst — array creation defaults; C-order (row-major) contiguous by default.
- numpy/core/include/numpy/ndarraytypes.h — C-API struct fields (ndim, shape pointer, strides pointer in bytes, descr, data), source of truth for metadata.
- numpy/lib/stride_tricks.py — docs and comments referencing stride semantics (for context only; we exclude tricks).

## 3. Key architectural ideas

- ndarray = (data buffer, dtype, shape, strides)
  - shape: tuple of non-negative ints; len(shape) = ndim
  - strides: tuple of ints giving byte steps to move 1 index along each dimension in NumPy; for our IR we normalize to element-steps
  - dtype: defines itemsize; we fix to float64 → itemsize = 8 bytes
  - data: flat storage of size prod(shape) elements for contiguous arrays
- Row-major contiguous definition (C-order):
  - For dtype itemsize B, strides_bytes[i] = B * prod(shape[i+1:])
  - Last dimension stride_bytes = B
  - Example shape (2,3): strides_elements = (3,1); strides_bytes = (24,8)
- Logical index to storage:
  - Offset_elements = Σ indices[i] * strides_elements[i]
  - Offset_bytes = Offset_elements * itemsize
  - Address = data_ptr + Offset_bytes
- Invariants useful for the POC:
  - len(shape) == len(strides)
  - All dims ≥ 0; for the POC we assume dims > 0
  - For contiguous arrays, strides match default_strides(shape)
  - Total elements = prod(shape)

## 4. Minimal behavior to port

Representation (Lisp-like IR):

  (array :shape '(D0 ... Dn-1)
         :strides '(S0 ... Sn-1)   ; element-steps (not bytes)
         :dtype 'float64
         :data '(... flat float64 ...))

Functions:
- default-strides(shape): row-major element strides
  - Pseudocode:
    
    function default_strides(shape):
        n = len(shape)
        strides = [0]*n
        if n == 0:
            return []
        strides[n-1] = 1
        for i in range(n-2, -1, -1):
            strides[i] = strides[i+1] * shape[i+1]
        return strides

- flat-offset(indices, strides): index → flat element offset
  - Pseudocode:
    
    function flat_offset(indices, strides):
        assert len(indices) == len(strides)
        off = 0
        for i in range(len(indices)):
            off += indices[i] * strides[i]
        return off

- get(array, indices):
  - Pseudocode:
    
    function get(A, indices):
        off = flat_offset(indices, A.strides)
        return A.data[off]

Construction rules for POC:
- new_array(shape, data):
  - assert dtype is float64
  - assert len(data) == prod(shape)
  - strides = default_strides(shape)
  - return array(shape, strides, 'float64, data)

Note: General indexing/broadcasting are covered in subsequent assignments; here we establish metadata and contiguous layout only.

## 5. Explicit exclusions

- Non-contiguous views; arbitrary/negative strides; subarray views/slices
- Fortran (column-major) order and memory-order flags
- Dtype system beyond float64; byteorder, alignment, structured dtypes
- Ownership/base-object, writeable/readonly flags, memory-mapped buffers
- Object and masked arrays
- Zero-sized dimensions (defer or reject in POC)

## 6. Suggested tests

- Default strides (elements) for simple shapes:
  - shape (3,) → strides (1,)
  - shape (2,3) → strides (3,1)
- Offset mapping sanity (elements):
  - shape (2,3), strides (3,1):
    - (0,0) → 0; (0,2) → 2; (1,0) → 3; (1,2) → 5
- Round-trip get with flat row-major data:
  - A = (array :shape '(2 3)
                :strides '(3 1)
                :dtype 'float64
                :data '(1 2 3 4 5 6))
  - get(A, '(0 0)) == 1; get(A, '(0 2)) == 3; get(A, '(1 0)) == 4; get(A, '(1 2)) == 6
- Size/consistency checks:
  - prod(shape) equals length(data); len(shape) == len(strides)

## 7. Open questions

- Strides unit: NumPy uses bytes; POC IR uses element-steps for simplicity. Acceptable mismatch? Implementation must multiply by 8 only if interfacing with raw bytes.
- Zero-sized dimensions: should we allow shape components of 0? Simpler to forbid initially; confirm with top crawler.
- Do we store ndim explicitly or derive as len(shape)? Plan: derive to reduce redundancy.
- Any need to expose itemsize separately given fixed float64? Likely constant 8; omit from IR for now.
