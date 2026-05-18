;;;; numpy-speedrun-poc/src/tensor/indexing.lisp
;; Indexing helpers: flat-offset, aref-nd

(in-package :mini-array.tensor)

(defun flat-offset (index strides &optional shape)
  "Compute flat offset (row-major) from INDEX and STRIDES.
If SHAPE is provided, perform bounds checking against it.
INDEX shapes: NIL (0D), (i) (1D), (i j) (2D)."
  (labels ((%check-bounds-1 (i n)
             (unless (and (integerp i) (<= 0 i) (< i n))
               (error "flat-offset: index ~S out of bounds for dim ~A" i n))))
    (cond
      ;; 0D
      ((or (null shape) (and shape (null shape)))
       (when (and shape (not (null index)))
         (error "flat-offset: 0D expects NIL index, got ~S" index))
       0)
      ;; 1D
      ((= (length (or shape strides)) 1)
       (destructuring-bind (s0) strides
         (destructuring-bind (i) index
           (when shape
             (destructuring-bind (n) shape
               (%check-bounds-1 i n)))
           (* i s0))))
      ;; 2D
      ((= (length (or shape strides)) 2)
       (destructuring-bind (s0 s1) strides
         (destructuring-bind (i j) index
           (when shape
             (destructuring-bind (r c) shape
               (%check-bounds-1 i r)
               (%check-bounds-1 j c)))
           (+ (* i s0) (* j s1)))))
      (t (error "flat-offset: unsupported rank from SHAPE/STRIDES")))))

(defun aref-nd (a index)
  "Fetch element from ndarray A at INDEX using its own shape/strides.
Performs bounds checking."
  (let* ((shape (shape-of a))
         (strides (strides-of a))
         (off (flat-offset index strides shape)))
    (aref (ndarray-data a) off)))
