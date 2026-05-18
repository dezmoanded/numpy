# indexing

## 1. Summary

Implemented rank-1 boolean selection (boolean-select) for the indexing module. Supports selecting elements from a 1D ndarray using a same-length list of booleans (T/NIL). Preserves order and dtype. All-false masks are rejected to honor the V1 no–zero-sized-arrays policy.

## 2. Files/docs inspected

- src/package.lisp — top-level exports used by legacy helpers (asarray, make-array-from-flat, aref-nd)
- src/core.lisp — shape/strides helpers and invariants in the legacy flat layout (referenced by tests)
- src/tensor.lisp — ndarray struct, asarray, aref-nd, to-list behavior in the legacy flat layout (referenced by tests)
- src/indexing/package.lisp — defines :mini-array.indexing package and exports boolean-select
- src/indexing/boolean-select.lisp — implementation of boolean-select
- src/indexing/tests.lisp — module tests (happy paths + error cases)
- src/indexing/CONTRACT.md — API, scope, invariants, exclusions

## 3. Key architectural ideas

- Keep indexing orthogonal to broadcasting/ufuncs; depend only on array metadata and element accessors.
- Boolean masks are plain Lisp lists in V1 (not arrays) to minimize dependencies; each entry must be exactly T or NIL.
- Only rank-1 arrays are supported to keep scope minimal; output is a new contiguous ndarray.
- All-false masks are errors to avoid constructing zero-sized arrays (disallowed globally in V1).

## 4. Minimal behavior to port

Function: boolean-select(array, mask)
- Inputs:
  - array: ndarray, rank=1; dtype in {:float64, :bool} (practically :float64 for current data paths)
  - mask: proper list of booleans (T/NIL), length == length(array)
- Behavior:
  - Produce a new 1D ndarray of the same dtype containing elements where mask[i] is T, preserving order.
- Errors:
  - Non-ndarray input; array rank != 1
  - Mask not a list, wrong length, or contains non-boolean entries
  - All-false mask (would yield zero-sized output)

Pseudocode:
- assert array is ndarray and rank(array) == 1
- assert listp(mask) and length(mask) == n = shape[0]
- assert every(booleanp, mask)
- let out = empty list
- for i in 0..n-1:
  - when mask[i] == T: push aref-nd(array, (i)) to out
- if length(out) == 0: error("all-false mask not allowed in V1")
- return make-array-from-flat((length(out)), nreverse(out))

## 5. Implementation handoff

Data structures
- Reuse ndarray from tensor (shape, strides, dtype, data). No views or negative strides.

Functions to implement/export
- :mini-array.indexing:boolean-select (done)

Algorithm outline
- Single pass over mask; collect selected values by aref-nd; build a contiguous 1D result via make-array-from-flat; preserve dtype.

Inputs/outputs
- In: ndarray rank-1, list of T/NIL
- Out: ndarray rank-1 with shape (k), where k is number of true entries (k>0 enforced)

Error handling
- Clear error messages for rank mismatch, mask length mismatch, non-boolean mask entries, and all-false masks.

Simplifications (V1)
- Rank limited to 1 for array argument
- Mask is a Lisp list only (not an ndarray)
- Zero-sized output disallowed

## 6. Explicit exclusions
- 0D/2D inputs; advanced indexing; slicing; integer/fancy indexing; ndarray-valued masks; strided/non-contiguous views; negative strides; zero-sized results.

## 7. Suggested tests

- Happy path: select alternating elements from '(10 20 30 40 50) with mask '(t nil t nil t) ⇒ '(10 30 50)
- Errors:
  - all-false mask '(nil nil nil)
  - rank != 1 input (2D array)
  - mask length mismatch
  - non-boolean entries '(t 1 nil 0)

## 8. Open questions
- When :bool dtype ndarrays are broadly available, should we also accept a 1D bool ndarray as mask (same shape), in addition to a list?
- When the global policy allows zero-sized arrays, should boolean-select return an empty array instead of erroring on all-false masks?
