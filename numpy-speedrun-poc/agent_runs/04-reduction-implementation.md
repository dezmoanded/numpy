# 04-reduction-implementation

## Summary
Implemented sum-axis for 2D arrays along axis 0 and 1. Nested loops with double-float accumulator; returns 1D output vector.

## Files touched
- src/mini-array.lisp — sum-axis

## Behavior
- axis=0: out shape (C), out[j] = sum_i A[i,j]
- axis=1: out shape (R), out[i] = sum_j A[i,j]
- Errors for non-2D inputs or invalid axes.

## Notes
- No keepdims/axis tuples/dtype promotion. Identity is 0.0d0.
