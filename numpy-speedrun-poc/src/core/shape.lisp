;;;; mini-array.core shape utilities

(in-package :mini-array.core)

(defun product (dims)
  "Return the product of a list of positive integer dimensions. NIL -> 1."
  (let ((acc 1))
    (dolist (d dims acc)
      (setf acc (* acc d)))))

(defun rank (shape)
  "Return the rank (length) of SHAPE. NIL -> 0."
  (length shape))

(defun %positive-integer-list-p (xs)
  (and (listp xs)
       (every (lambda (x) (and (integerp x) (> x 0))) xs)))

(defun valid-shape-p (shape)
  "V1 validity: proper list of length <= 2, each dim a positive integer (>0)."
  (and (or (null shape) (listp shape))
       (<= (rank shape) 2)
       (%positive-integer-list-p shape)))

(defun default-strides (shape)
  "Row-major element strides for ranks 0..2.
  Errors if SHAPE invalid or rank > 2."
  (unless (valid-shape-p shape)
    (error "default-strides: invalid shape ~S (rank<=2, positive integers only)" shape))
  (case (rank shape)
    (0 nil)
    (1 (list 1))
    (2 (destructuring-bind (r c) shape
         (declare (ignore r))
         (list c 1)))
    (t (error "default-strides: rank>2 not supported: ~S" shape))))
