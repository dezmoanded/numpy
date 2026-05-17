;;;; numpy-speedrun-poc/src/core.lisp
;;;; Core utilities: numeric coercion, shape/stride helpers, buffers (V1)

(in-package :mini-array)

(defun %ensure-double (x)
  (coerce x 'double-float))

(defun product (shape)
  "Product of dimensions for SHAPE; nil -> 1."
  (if (null shape)
      1
      (reduce #'* shape :initial-value 1)))

(defun rank (shape)
  "Public: return the rank (length) of SHAPE."
  (length shape))

(defun valid-shape-p (shape)
  "V1: SHAPE must be a list with rank<=2 and all positive integer dims (>0). nil is allowed (0D)."
  (and (listp shape)
       (<= (length shape) 2)
       (every (lambda (d) (and (integerp d) (> d 0))) shape)))

(defun default-strides (shape)
  "Row-major element-based strides for ranks 0/1/2. Error for rank>2."
  (cond
    ((null shape) nil)
    ((= (length shape) 1) (list 1))
    ((= (length shape) 2)
     (let ((r (first shape))
           (c (second shape)))
       (declare (ignore r))
       (list c 1)))
    (t (error "default-strides: rank > 2 not supported in V1: ~S" shape))))

(defun %make-double-vector (len &optional (init 0.0d0))
  "Allocate a simple double-float vector of length LEN, initialized to INIT."
  (let ((v (make-array len :element-type 'double-float)))
    (loop for i from 0 below len do
      (setf (aref v i) (coerce init 'double-float)))
    v))
