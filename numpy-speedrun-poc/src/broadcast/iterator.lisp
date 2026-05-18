;;;; mini-array.broadcast iterator

(in-package :mini-array.broadcast)

(defun all-indices (shape)
  "Enumerate all indices for SHAPE in row-major order.

Returns a list of index lists:
- NIL -> (NIL)
- (n) -> ((0) ... (n-1))
- (r c) -> ((0 0) (0 1) ... (r-1 c-1))
Errors if SHAPE invalid or rank>2."
  (unless (mini-array.core:valid-shape-p shape)
    (error "all-indices: invalid shape ~S" shape))
  (case (mini-array.core:rank shape)
    (0 (list nil))
    (1 (let* ((n (first shape))
              (out '()))
         (dotimes (i n (nreverse out))
           (push (list i) out)))
    )
    (2 (destructuring-bind (r c) shape
         (let ((out '()))
           (dotimes (i r (nreverse out))
             (dotimes (j c)
               (push (list i j) out))))))
    (t (error "all-indices: rank>2 not supported: ~S" shape))))
