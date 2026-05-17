# 08-lisp-ir-port-plan

## 1. Summary

Design a tiny, clean-room, Lisp-like IR and API to express NumPy-like array programs sufficient for: float64 scalars/arrays; 1D/2D; row-major contiguous storage; positive strides; broadcasting; add; multiply; sum along axis 0 or 1. The IR separates array metadata from iteration, mirrors NumPy’s broadcasting semantics, and keeps a minimal set of primitives that an interpreter can implement directly.

## 2. Files/docs inspected

- doc/source/user/basics.broadcasting.rst — semantic rules for broadcasting, compatibility, and result shape.
- doc/source/reference/ufuncs.rst — conceptual model of ufuncs: scalar kernels applied over broadcasted iteration.
- doc/source/reference/arrays.ndarray.rst — ndarray core concepts: shape, strides, dtype, row-major layout.
- numpy/core/include/numpy/ndarraytypes.h — confirms presence/role of shape/strides/dtype in the C struct (for vocabulary only; no code copied).

## 3. Key architectural ideas

- Data model: An array is a tuple of (shape, strides, dtype, data). For this POC: dtype is fixed to float64; storage is flat row-major; strides are positive and represent elements, not bytes.
- Separation of concerns:
  - Metadata and indexing compute flat offsets.
  - Broadcasting computes an output shape and per-operand index projection.
  - Elementwise ops reuse one broadcasted iteration skeleton and swap scalar kernels (add, mul).
  - Reduction iterates systematically over one axis while accumulating.
- Deterministic errors: Broadcasting incompatibility and invalid axis raise explicit errors. No dtype promotion or casting.

## 4. Minimal behavior to port

Proposed surface syntax (S-expressions):

- Array literal with optional strides (default row-major if omitted):
  (array :shape '(R C)
         :dtype 'float64
         :data '(...))
  (array :shape '(R C) :strides '(C 1) :dtype 'float64 :data '(...))

- Introspection:
  (shape A) -> list of ints
  (strides A) -> list of ints

- Core ops:
  (broadcast-shape a-shape b-shape) -> list of ints or error
  (binary-ufunc op A B) -> new array
  (add A B) ; sugar for (binary-ufunc + A B)
  (mul A B) ; sugar for (binary-ufunc * A B)
  (sum-axis A axis) -> new array

Minimal internal representation (host record shown schematically):

- Array = { shape: int[], strides: int[], dtype: :float64, data: float64[] }
- Invariants:
  - length(strides) == length(shape)
  - contiguous row-major default-strides(shape) = compute once if not provided
  - size(shape) == product(shape)
  - length(data) == size(shape)

Pseudocode (Lisp-like, schematic):

; Utilities
(def default-strides (shape)
  (let ((n (len shape))
        (s (make-list n 1)))
    (when (> n 0)
      (set! (nth s (- n 1)) 1)
      (for i from (- n 2) downto 0
        (set! (nth s i) (* (nth s (+ i 1)) (nth shape (+ i 1))))))
    s))

(def flat-offset (indices strides)
  (reduce + 0 (map * indices strides)))

(def get (A indices)
  (nth (A.data) (flat-offset indices (A.strides))))

; Broadcasting
(def broadcast-shape (a b)
  (let ((ra (reverse a)) (rb (reverse b)) (r '()))
    (for k from 0 to (max (len ra) (len rb))
      (let ((da (if (< k (len ra)) (nth ra k) 1))
            (db (if (< k (len rb)) (nth rb k) 1)))
        (cond
          ((or (= da db) (= da 1)) (push! r (max da db)))
          ((= db 1) (push! r da))
          (else (error 'broadcast-incompatible a b)))))
    (reverse r)))

(def project-index (out-idx in-shape out-shape)
  ; Align right; if in dim == 1, clamp to 0, else copy
  (let ((oi (align-right out-idx out-shape))
        (is (align-right in-shape out-shape))
        (r '()))
    (for i from 0 to (len oi)-1
      (let ((id (nth is i)) (od (nth oi i)) (sd (nth in-shape (- (len in-shape) 1 i))))
        (push! r (if (= sd 1) 0 od))))
    (right-trim r (len in-shape))))

; Elementwise binary ufunc
(def binary-ufunc (op A B)
  (let* ((oshape (broadcast-shape (A.shape) (B.shape)))
         (out (array :shape oshape :dtype 'float64
                     :data (make-vector (product oshape) 0.0))))
    (for-each-index oshape (lambda (oi)
      (let* ((ai (project-index oi (A.shape) oshape))
             (bi (project-index oi (B.shape) oshape))
             (av (get A ai))
             (bv (get B bi))
             (ov (op av bv))
             (off (flat-offset oi (default-strides oshape))))
        (set! (nth (out.data) off) ov)))
    out))

(def add (A B) (binary-ufunc + A B))
(def mul (A B) (binary-ufunc * A B))

; Reduction along axis (2D only in first cut)
(def sum-axis (A axis)
  (let* ((s (A.shape))
         (rows (nth s 0)) (cols (nth s 1)))
    (cond
      ((= axis 0)
       (let ((out (array :shape (list cols) :dtype 'float64
                          :data (make-vector cols 0.0))))
         (for j from 0 to cols-1
           (let ((acc 0.0))
             (for i from 0 to rows-1
               (set! acc (+ acc (get A (list i j)))))
             (set! (nth (out.data) j) acc)))
         out))
      ((= axis 1)
       (let ((out (array :shape (list rows) :dtype 'float64
                          :data (make-vector rows 0.0))))
         (for i from 0 to rows-1
           (let ((acc 0.0))
             (for j from 0 to cols-1
               (set! acc (+ acc (get A (list i j)))))
             (set! (nth (out.data) i) acc)))
         out))
      (else (error 'invalid-axis axis)))))

Small demo program and expected outputs:

; Construct arrays
(def A (array :shape '(2 3) :dtype 'float64 :data '(1 2 3 4 5 6)))
(def b (array :shape '(3)   :dtype 'float64 :data '(10 20 30)))

; Elementwise add with broadcasting: (2,3) + (3) -> (2,3)
(def C (add A b))
; Expected C.data = (11 22 33 14 25 36)

; Multiply by scalar: (2,3) * scalar -> (2,3)
(def D (mul A 2.0))
; Expected D.data = (2 4 6 8 10 12)

; Row-wise sums: sum over axis=1 -> shape (2)
(def r1 (sum-axis A 1))
; Expected r1.data = (6 15)

## 5. Explicit exclusions

- Any dtype other than float64; dtype promotion; casting; NaN/Inf special casing beyond IEEE behavior.
- Non-contiguous layouts; negative or zero strides; views/slices; advanced indexing.
- Higher-rank reductions; keepdims; multiple axes; accumulator dtype selection.
- Full ufunc dispatch, type resolution, or generalized ufuncs.
- Memory aliasing checks; writeback semantics; threading/SIMD/BLAS.

## 6. Suggested tests

- Broadcasting add (2D + 1D):
  - Input: A = reshape(range(1,7),(2,3)); b = [10,20,30]
  - Expect: add(A,b).shape == (2,3); data == [11,22,33,14,25,36]
- Scalar multiply:
  - Input: A; s = 2.0
  - Expect: mul(A,s).data == [2,4,6,8,10,12]
- Chained op + reduction:
  - Input: mul(A,b) then sum-axis(axis=1)
  - Expect: [(1*10+2*20+3*30)=140, (4*10+5*20+6*30)=320]
- Incompatible broadcasting raises error:
  - Shapes (2,3) and (2,2) -> broadcast-incompatible
- sum-axis axis=0 and axis=1 produce shapes (3) and (2) respectively with expected totals.

Express these as small evaluator runs over the S-expressions above and compare to NumPy for parity.

## 7. Open questions

- Exact error classes/messages: use simple symbolic tags now; refine later for parity.
- Scalar handling syntax: treat host numbers as 0-d arrays or special-case in evaluator? For MVP, special-case host float64 as scalar broadcasting.
- Should strides be stored as elements or bytes? For this IR, elements are simpler; later typed IR may switch to bytes.
- Zero-sized dimensions: exclude in MVP or allow? MVP can exclude; later extension can admit them with clear rules.
- How to surface iteration order guarantees? Assume row-major for now; later IR pass could parameterize order.
