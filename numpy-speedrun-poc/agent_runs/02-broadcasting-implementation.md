# 02-broadcasting-implementation

## Summary
Implemented broadcast-shape, all-indices, and project-broadcast-index per V1: right-aligned comparison, size 1 expands, scalar NIL broadcasts to any shape, iteration is row-major.

## Files touched
- src/mini-array.lisp — broadcast-shape, all-indices, project-broadcast-index

## Behavior
- broadcast-shape compares from trailing dims, pads missing with 1, returns max or errors.
- all-indices:
  - NIL → (NIL)
  - (n) → ((0) .. (n-1))
  - (r c) → ((0 0) .. (r-1 c-1))
- project-broadcast-index clamps axes of size 1 to 0 when mapping output index back to operand index.

## Examples
- SA=(2 3), SB=(3) → (2 3)
- SA=NIL, SB=(2 3) → (2 3)
- in (3), out (2 3), out-idx (1 2) → in-idx (2)
- in (1 4), out (3 4), out-idx (2 3) → in-idx (0 3)
