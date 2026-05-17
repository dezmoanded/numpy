# 07-numpy-parity-tests

## 1. Summary

Design a tiny pytest suite that treats NumPy as the behavioral oracle for the clean-room core. The tests target only the POC scope: float64 scalars/1D/2D, row-major contiguous layout, positive strides, broadcasting, add, multiply, and sum over axis 0 or 1. Each test asserts both shape and values, and verifies that incompatible broadcasting raises an error.

## 2. Files/docs inspected
- doc/source/user/basics.broadcasting.rst — Authoritative user doc for broadcasting rules and examples.
- doc/source/user/theory.broadcasting.rst — Conceptual description of matching from trailing dimensions.
- doc/source/user/basics.ufuncs.rst — How ufuncs apply elementwise with broadcasting (context for add/multiply parity).
- doc/source/reference/ufuncs.rst — Reference overview of ufunc behavior (kept at semantic level).
- doc/source/reference/arrays.ndarray.rst — ndarray.sum and axis semantics.
- doc/source/reference/routines.math.rst — numpy.sum reference and reduction notes.
- doc/source/reference/routines.testing.rst — Guidance on numerical comparisons (assert_allclose semantics).

## 3. Key architectural ideas
- Parity-by-behavior: Use NumPy computations as ground truth; verify our results match in shape and values.
- Broadcasting semantics: Compare shapes from the trailing dimensions; sizes must match or be 1; scalars broadcast to any shape; incompatible sizes raise an error.
- Axis reductions: sum over a single axis for 2D arrays; axis=0 reduces rows (column-wise); axis=1 reduces columns (row-wise); output shape drops the reduced axis.
- Float64 only: Avoid dtype promotion/dispatch; construct test inputs as float64 to keep comparisons simple.
- Numerical comparison: Prefer exact equality for small integers but use allclose-equivalent tolerance for float64 (tests below rely on NumPy to compute expected, then compare values).

## 4. Minimal behavior to port
Expose a tiny API used by the tests:
- asarray(py): construct a float64 array from scalar, 1D list, or 2D nested list.
- shape(x): return a Python tuple of ints.
- add(a, b): elementwise addition with broadcasting.
- mul(a, b): elementwise multiplication with broadcasting.
- sum_axis(a, axis): sum along axis 0 or 1 for 2D; returns 1D.
- tolist(x): materialize as nested Python lists matching shape (for test comparison).
- Error: raise ValueError on incompatible broadcast or invalid axis.

## 5. Explicit exclusions
- Dtype promotion, mixed dtypes, object/complex/NaN handling beyond float64.
- Non-contiguous views, negative strides, advanced indexing, slicing.
- keepdims, axis=None, multiple axes, empty shapes, >2D shapes.
- Full ufunc dispatch, threading/SIMD/BLAS optimizations.

## 6. Suggested tests
```python
import numpy as np
import pytest

# Clean-room API under test
from tiny import asarray, shape, add, mul, sum_axis, tolist

def as_np(x):
    """Convert clean-room array to a NumPy float64 array for comparison."""
    return np.array(tolist(x), dtype=np.float64)


def test_broadcast_add_2d_1d():
    a = asarray([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])
    b = asarray([10.0, 20.0, 30.0])

    out = add(a, b)

    assert shape(out) == (2, 3)
    expected = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]], dtype=np.float64) + np.array([10.0, 20.0, 30.0], dtype=np.float64)
    np.testing.assert_allclose(as_np(out), expected)


def test_mul_scalar_2d():
    a = asarray([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])

    out = mul(a, 2.5)  # scalar broadcasting

    assert shape(out) == (2, 3)
    expected = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]], dtype=np.float64) * 2.5
    np.testing.assert_allclose(as_np(out), expected)


def test_mul_2d_1d_then_sum_axis1():
    a = asarray([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])
    w = asarray([1.0, 0.5, 0.25])

    prod = mul(a, w)            # broadcast (2,3) * (3,)
    out = sum_axis(prod, axis=1)  # sums per row -> shape (2,)

    assert shape(out) == (2,)
    expected = (np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=np.float64)
                * np.array([1.0, 0.5, 0.25], dtype=np.float64)).sum(axis=1)
    np.testing.assert_allclose(as_np(out), expected)


def test_incompatible_broadcast_shape_raises():
    a = asarray([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])  # shape (2,3)
    b = asarray([1.0, 2.0])         # shape (2,)

    with pytest.raises(ValueError):
        add(a, b)


def test_sum_axis0():
    a = asarray([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])

    out = sum_axis(a, axis=0)

    assert shape(out) == (3,)
    expected = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]], dtype=np.float64).sum(axis=0)
    np.testing.assert_allclose(as_np(out), expected)


def test_sum_axis1():
    a = asarray([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])

    out = sum_axis(a, axis=1)

    assert shape(out) == (2,)
    expected = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]], dtype=np.float64).sum(axis=1)
    np.testing.assert_allclose(as_np(out), expected)
```

## 7. Open questions
- Exception type: Should incompatible broadcasting raise ValueError or a custom BroadcastError? Tests assume ValueError.
- Input coercion: Should scalar inputs be wrapped via asarray by the caller, or auto-coerced inside add/mul? Tests rely on auto-coercion for scalars.
- tolist vs to_numpy: Tests assume tolist(x) exists; alternatively, we could expose to_numpy(x) for direct ndarray comparison.
