# 01-core-array-implementation

## 1) Summary
V1 core array for the Lisp mini-runtime. Implements contiguous row-major float64 arrays of rank 0–2 with shape, element-based strides, and flat data buffer. Provides constructors, validators, indexing (get/set), iteration helpers, and materialization (tolist). All behavior is constrained to the V1 scope from agent_runs/00-final-architecture-port-plan.md.

## 2) Scope confirmation and guardrails (V1)
- dtype: float64 only; store as simple-array of double-float
- ranks: 0D (scalar), 1D, 2D only
- layout: contiguous row-major; positive element-based strides
- dims: strictly > 0 in V1 (no zero-sized dims)
- errors: signal error for invalid axis or broadcast shape checks done elsewhere
- disallow: non-contiguous views, negative strides, slicing/advanced indexing

Add assertions in constructors and setters to enforce invariants above.

## 3) Data model and invariants
ndarray record (class/struct)
- shape: list<int> (NIL for 0D, (n) for 1D, (r c) for 2D)
- strides: list<int> (same length as shape; element-based)
- dtype: keyword :float64 (constant in V1)
- data: simple-array double-float (*), length = product(shape)

Invariants
- length(shape) == length(strides)
- strides == default-strides(shape) for all constructed arrays in V1
- product(shape) == length(data)
- For 2D (R,C): strides == (C 1)

## 4) API to implement (package :mini-array)
Exported symbols
- ndarray (struct/class) and constructor array
- product, default-strides, valid-shape-p
- flat-offset, aref-nd, set-aref-nd
- shape-of, strides-of
- all-indices
- asarray, to-list

Function behaviors
- (product shape) -> positive integer; NIL => 1; (n) => n; (r c) => (* r c)
- (valid-shape-p shape) -> boolean; ranks 0–2; all dims positive; integers
- (default-strides shape) -> list<int>; NIL => NIL; (n) => (1); (r c) => (c 1)
- (array shape data) -> ndarray; validates product(shape) == length(data); sets strides = default-strides(shape); coerces data to double-float
- (shape-of A) -> shape list; (strides-of A) -> strides list
- (flat-offset idx strides) -> integer; sum(idx[i] * strides[i]); with bounds checks when paired with shape
- (aref-nd A idx) -> double-float; idx is list<int> with length == rank(A) (NIL for scalar); performs bounds check
- (set-aref-nd A idx val) -> double-float; same idx rules; stores double-float
- (all-indices shape) -> iterator/sequence of index lists in row-major order for ranks 0–2 (returns a single NIL for 0D)
- (asarray x) -> ndarray; float -> 0D; 1D list<number> -> (n); 2D list<list<number>> rectangular -> (m n)
- (to-list A) -> nested lists; 0D -> number; 1D -> (list ...); 2D -> ((row1) (row2) ...)

## 5) Algorithms (CL-like pseudocode)
- product
  - (cond ((null shape) 1)
         ((and (consp shape) (every #'positive-integer-p shape)) (reduce #'* shape))
         (t (error "Invalid shape ~a" shape)))

- default-strides
  - (case (length shape)
      (0 NIL)
      (1 (list 1))
      (2 (destructuring-bind (r c) shape (list c 1)))
      (t (error "Rank > 2 not supported in V1")))

- array (constructor)
  - (assert (valid-shape-p shape))
  - (let* ((n (product shape))
          (buf (make-array n :element-type 'double-float)))
      (assert (= n (length data)))
      (loop for i from 0 below n
            for v in data do (setf (aref buf i) (coerce v 'double-float)))
      (make-ndarray :shape shape :strides (default-strides shape)
                    :dtype :float64 :data buf))

- flat-offset (with shape for bounds)
  - (assert (= (length idx) (length strides)))
  - (loop for k from 0
         for i in idx
         for s in strides
         sum (* i s) into off
         finally (return off))

- aref-nd / set-aref-nd
  - (let* ((rank (length (shape-of A)))
          (idx (or idx NIL))) ; NIL allowed for 0D
      (assert (= (length idx) rank))
      (bounds-check idx (shape-of A))
      (let ((o (flat-offset idx (strides-of A))))
        (if get (aref (data A) o)
                (setf (aref (data A) o) (coerce val 'double-float)))))

- bounds-check
  - for each axis k: 0 <= idx[k] < shape[k]; else (error "index out of bounds")

- all-indices
  - rank 0: return list of one element: (NIL)
  - rank 1: for i=0..n-1 -> (list i)
  - rank 2: nested loops i=0..r-1, j=0..c-1 -> (list i j)

- asarray
  - number -> shape NIL; strides NIL; data (list (coerce x 'double-float))
  - list<number> -> ensure all numbers; shape (n); data row-major copy as double-float
  - list<list<number>> -> rectangular check; shape (m n); data row-major flatten as double-float
  - otherwise error

- to-list
  - rank 0 -> return the single number (aref-nd A NIL)
  - rank 1 -> build list via (aref-nd A (list i))
  - rank 2 -> outer loop i, inner loop j -> rows of numbers

## 6) Error handling
- invalid shape (rank > 2, non-positive dims, non-integers) -> error "Invalid shape"
- length(data) mismatch -> error "Data length mismatch"
- aref/set index length mismatch -> error "Rank/index length mismatch"
- index out of bounds -> error with axis/index/size details

Notes
- Use a dedicated condition type (e.g., mini-array-error) if desired, but simple-error with clear messages is sufficient for V1.

## 7) Simplifications and explicit exclusions
- No views or slicing; every ndarray owns a contiguous buffer
- Strides always equal default-strides; no negative or zero-byte strides
- No dtype promotion/conversion beyond coercion to double-float
- No zero-sized dims in V1

## 8) Tiny tests/examples (parity-targeted)
- (default-strides '(2 3)) => '(3 1)
- (product '()) => 1; (product '(4)) => 4; (product '(2 3)) => 6
- (to-list (asarray '((1 2) (3 4)))) => '((1.0d0 2.0d0) (3.0d0 4.0d0))
- (let* ((A (asarray '(1 2 3))) (v (aref-nd A '(2)))) (= v 3.0d0))
- (let* ((A (asarray '((1 2) (3 4)))) (v (aref-nd A '(1 0)))) (= v 3.0d0))
- (signals error (asarray '())) ; empty list invalid in V1 due to zero-sized dims

## 9) Implementation checklist
- Define package :mini-array exporting the symbols above
- Define ndarray struct/class with shape, strides, dtype, data
- Implement product, valid-shape-p, default-strides
- Implement array (constructor) with validations
- Implement flat-offset, bounds-check, aref-nd, set-aref-nd
- Implement all-indices for ranks 0–2
- Implement asarray and to-list with coercion to double-float
- Add unit tests mirroring Tiny tests above

## 10) Notes for integration
- Broadcasting, elementwise ops, and reductions build on these primitives:
  - broadcasting uses shape-of and default row-major iteration via all-indices
  - ufunc engine uses aref-nd/set-aref-nd and project indices (in its module)
  - reduction uses nested loops with aref-nd over 2D shapes
- Keep this module free of higher-level concerns; it should not depend on broadcasting or ufunc modules.
