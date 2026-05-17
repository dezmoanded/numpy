# 03-broadcasting-rules

## 1. Summary

Extracts NumPy’s semantic broadcasting rules and provides clean-room pseudocode for computing a binary broadcasted shape within a 0D/1D/2D scope. Core rule: compare shapes from the trailing (rightmost) dimensions; dimensions are compatible if equal or one is 1; missing leading dimensions are treated as 1. Result shape takes the maximum at each compared position. Scalars broadcast to any shape. Incompatible pairs raise ValueError.

## 2. Files/docs inspected
- doc/source/user/basics.broadcasting.rst — Authoritative description of broadcasting semantics, examples, and failure cases.
- doc/source/user/theory.broadcasting.rst — Legacy pointer confirming basics.broadcasting.rst is the updated reference.
- numpy/_core/_add_newdocs.py — Docstrings for numpy.broadcast, broadcast_to, broadcast_shapes (for phrasing and conceptual parity).
- numpy/_core/tests/test_numeric.py — Tests around np.broadcast for error behavior and argument handling phrasing.

## 3. Key architectural ideas
- Right-aligned comparison: Compare dimensions from the last (trailing/right) moving left.
- Compatibility relation per-axis: Two sizes are compatible if they are equal or one of them is 1.
- Implicit leading 1s: If ranks differ, conceptually pad the shorter shape on the left with 1s for comparison.
- Result shape rule: At each axis, the resulting size is max(a_i, b_i) after padding. If neither is 1 and they are not equal, broadcasting fails.
- Scalars: The empty shape () is treated as if all missing axes were 1; thus it broadcasts to any shape.
- Error on first mismatch (for our binary function): If any compared pair is incompatible, raise a ValueError that reports both input shapes.
- Scope constraints for POC: Limit to float64 arrays, 0D/1D/2D, contiguous row-major, positive strides. Broadcasting only affects shape logic; no data copies implied by shape computation.

## 4. Minimal behavior to port
- Inputs
  - shape_a, shape_b: tuples of non-negative integers representing 0D/1D/2D shapes. Prefer positive sizes for the first POC; zero-sized axes may be deferred (see Open questions).
- Output
  - A tuple representing the broadcasted shape, with rank equal to max(len(shape_a), len(shape_b)), up to 2 for this POC.
- Errors
  - Raise ValueError with a concise message (e.g., "operands could not be broadcast together with shapes {shape_a} {shape_b}") when any axis pair is incompatible.

- Pseudocode: broadcast_shape(shape_a, shape_b)

```
function broadcast_shape(shape_a, shape_b):
    # Accept 0D/1D/2D shapes ((), (n,), (m,n))
    ra = length(shape_a)
    rb = length(shape_b)
    r = max(ra, rb)

    # Pad on the left with 1s to align trailing axes
    pa = list(shape_a)
    pb = list(shape_b)
    while length(pa) < r: prepend 1 to pa
    while length(pb) < r: prepend 1 to pb

    result = new list of length r
    for i in range(r-1, -1, -1):  # from last to first
        a = pa[i]
        b = pb[i]
        if a == b:
            result[i] = a
        else if a == 1:
            result[i] = b
        else if b == 1:
            result[i] = a
        else:
            raise ValueError("operands could not be broadcast together with shapes "
                             + str(tuple(shape_a)) + " " + str(tuple(shape_b)))

    return tuple(result)
```

- Worked examples (shapes only)
  - (2, 3) with (3,) -> pad (3,) to (1,3); per-axis max => (2,3)
  - (2, 3) with ()   -> pad () to (1,1); per-axis max => (2,3)
  - (3, 1) with (1, 4) -> already aligned; per-axis max => (3,4)
  - (2, 3) with (4,) -> pad (4,) to (1,4); last axis 3 vs 4 incompatible -> ValueError

- Notes for implementation
  - Keep broadcast_shape purely about shapes (no allocation, no dtype logic).
  - Later iteration can use the result to drive index arithmetic without materializing tiled data.

## 5. Explicit exclusions
- N-ary broadcasting helper (accepting more than two input shapes) — out of scope for first pass; compose via pairwise application if needed later.
- Dimensions beyond 2D — defer; algorithm generalizes but implementation target is 0D/1D/2D.
- Any dtype promotion or ufunc dispatch behavior — explicitly excluded.
- Strides for non-contiguous or negative-stride arrays — excluded in this POC.
- Subclassing, masked arrays, and object dtype broadcasting nuances — excluded.

## 6. Suggested tests
- Shape-only unit checks for the helper:

```
# happy paths
assert broadcast_shape((2,3), (3,)) == (2,3)
assert broadcast_shape((2,3), ()) == (2,3)
assert broadcast_shape((3,1), (1,4)) == (3,4)
assert broadcast_shape((1,5), (5,)) == (1,5)
assert broadcast_shape((5,), (1,5)) == (1,5)

# failures
with pytest.raises(ValueError):
    broadcast_shape((2,3), (4,))
with pytest.raises(ValueError):
    broadcast_shape((2,1), (3,))
```

- Optional parity checks against NumPy during development (not for the clean-room runtime):

```
import numpy as np

def to_np_shape(t):
    return () if len(t) == 0 else t

cases = [((2,3),(3,)), ((2,3),()), ((3,1),(1,4))]
for a,b in cases:
    # Build dummy arrays with ones to ask NumPy for the broadcasted shape
    A = np.empty(to_np_shape(a))
    B = np.empty(to_np_shape(b))
    expect = np.broadcast(A, B).shape
    assert broadcast_shape(a, b) == expect
```

## 7. Open questions
- Should zero-sized dimensions be supported in the first POC? NumPy allows them and the rule still holds; implementation simplicity may argue to defer.
- Exact error message wording: mirror NumPy’s phrasing closely or keep a simpler message? The above uses a concise, NumPy-like format.
- Do we want a convenience function for N-ary shapes (broadcast_shapes) now, or compose pairwise later when wiring ufuncs?
