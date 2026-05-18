# INDEXING module — CONTRACT

Module: src/indexing
Package: :mini-array.indexing
Exports: boolean-select
Depends on: core (via top-level :mini-array), tensor (via top-level :mini-array)

Scope (V1/V2 slice)
- Rank support: 1D only
- Dtypes: input numeric (:float64) for now; output preserves input dtype (:float64)
- Mask type: list of booleans (T/NIL); dtype enforcement is by validation here (no separate tensor bool arrays yet)
- Zero-sized dims: globally disallowed — all-false masks are rejected with an error

API
- boolean-select(array mask) -> ndarray
  - array: :mini-array ndarray, rank=1, dtype :float64
  - mask: proper list of booleans (T/NIL), length == length(array)
  - returns: new 1D ndarray with selected elements, in original order, contiguous, dtype matches input

Algorithm
- Validate array is ndarray and rank==1
- Validate mask is a list of length n and all elements are booleans (T or NIL)
- Count trues; if zero, signal error (zero-sized excluded)
- Iterate i=0..n-1; when mask[i] is T, push aref-nd(array, (i)) into an output list
- Nreverse the list to restore order and allocate via make-array-from-flat with shape (count)

Invariants and error cases
- Shape/rank: only (n); NIL or (r c) -> error
- Length mismatch: error
- Non-boolean mask entry: error
- All-false mask: error (documented policy)
- Output is contiguous row-major with default-strides '(1)

Complexity
- Time: O(n)
- Space: O(k) to store selected values, O(n) worst-case when all true

Simplifications / Exclusions (deferred)
- No 0D or 2D selection
- No integer indexing, slices, or advanced indexing forms
- No boolean tensor mask input yet — mask is a plain Lisp list
- No support for :bool arrays or mixed dtypes yet; when tensor grows :bool dtype, revisit validation to accept a 1D :bool ndarray as mask
- No support for empty outputs (global zero-sized exclusion); policy may change later

Testing policy
- tests.lisp must load only: src/package.lisp, src/core.lisp, src/tensor.lisp, then src/indexing/package.lisp and src/indexing/boolean-select.lisp
- Cover: basic selection, all-true, wrong rank, length mismatch, non-boolean mask, all-false -> error

Notes / Future
- When bool dtype arrays are available, add an overload: mask may be a 1D ndarray with dtype :bool, same shape
- When zero-sized arrays are allowed, remove the all-false error and return shape (0)
