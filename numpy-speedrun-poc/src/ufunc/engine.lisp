;;;; UFUNC engine — binary ufunc core loop

(in-package :mini-array.ufunc)

;; Dependencies (packages expected to be loaded before this file):
;;  - :mini-array.core      (nick :ma.core)
;;  - :mini-array.tensor    (nick :ma.tensor)
;;  - :mini-array.broadcast (nick :ma.broadcast)

(defun %bool-vector-to-nested-list (vec shape)
  "Convert a vector of T/NIL in row-major order into a nested list
   shaped by SHAPE (NIL | (n) | (r c)). For 0D, return the single element."
  (cond
    ((null shape) (aref vec 0))
    ((= (length shape) 1)
     (let* ((n (first shape))
            (out '()))
       (dotimes (i n (nreverse out))
         (push (aref vec i) out))))
    ((= (length shape) 2)
     (let* ((r (first shape))
            (c (second shape))
            (rows '()))
       (dotimes (i r (nreverse rows))
         (let ((row '()))
           (dotimes (j c)
             (push (aref vec (+ (* i c) j)) row))
           (push (nreverse row) rows)))))
    (t (error "bool nesting: rank>2 not supported: ~S" shape))))

(defun binary-ufunc (a b op &key out-dtype name (allow-bool-input-p nil))
  "Generic binary ufunc engine.

  Inputs A and B may be scalars or ndarrays. OP is a function of two
  arguments returning either a double-float (numeric ops) or T/NIL (comparisons).

  Keyword args:
    - :out-dtype — one of :float64 or :bool (required)
    - :name — optional symbol/name used in error messages
    - :allow-bool-input-p — when NIL, reject boolean-typed input arrays.

  Behavior:
    1) Normalize inputs via tensor:asarray
    2) Compute broadcasted output shape via broadcast:broadcast-shape
    3) Iterate broadcast indices; project to each input; fetch; apply OP
    4) Materialize an output ndarray of OUT-DTYPE with row-major storage.

  Errors:
    - Unsupported ranks (>2) (delegated to deps)
    - Broadcasting mismatch (delegated to deps)
    - Disallowed input dtype (:bool when not allowed)
    - OUT-DTYPE not in {:float64, :bool}.
  "
  (declare (type (function (t t) t) op))
  (unless (member out-dtype '(:float64 :bool))
    (error "~A: invalid out-dtype ~S (expected :float64 or :bool)"
           (or name 'binary-ufunc) out-dtype))
  (let* ((A (ma.tensor:asarray a))
         (B (ma.tensor:asarray b))
         (ad (ma.tensor:dtype-of A))
         (bd (ma.tensor:dtype-of B)))
    (when (and (not allow-bool-input-p)
               (or (eql ad :bool) (eql bd :bool)))
      (error "~A: boolean inputs are not allowed for this operation (got ~S and ~S)"
             (or name 'binary-ufunc) ad bd))
    (unless (and (ma.core:dtype-p ad) (ma.core:dtype-p bd))
      (error "~A: invalid operand dtype(s) ~S, ~S" (or name 'binary-ufunc) ad bd))
    (let* ((ash (ma.tensor:shape-of A))
           (bsh (ma.tensor:shape-of B))
           (osh (ma.broadcast:broadcast-shape ash bsh))
           (n   (ma.core:product osh)))
      (cond
        ((eql out-dtype :float64)
         ;; Accumulate double-floats row-major, then make ndarray via tensor:make-array-from-flat
         (let ((acc (make-array (max 1 n) :element-type 'double-float)))
           (let ((k 0))
             (dolist (idx (ma.broadcast:all-indices osh))
               (let* ((ia (ma.broadcast:project-broadcast-index idx ash osh))
                      (ib (ma.broadcast:project-broadcast-index idx bsh osh))
                      (va (ma.tensor:aref-nd A ia))
                      (vb (ma.tensor:aref-nd B ib))
                      (vx (funcall op va vb))
                      (df (ma.core:coerce-dtype-value vx :float64)))
                 (setf (aref acc k) df)
                 (incf k))))
           (ma.tensor:make-array-from-flat osh (loop for i from 0 below (max 1 n) collect (aref acc i)))))
        ((eql out-dtype :bool)
         ;; Accumulate T/NIL row-major; build nested list, then asarray with :dtype :bool
         (let ((acc (make-array (max 1 n))))
           (let ((k 0))
             (dolist (idx (ma.broadcast:all-indices osh))
               (let* ((ia (ma.broadcast:project-broadcast-index idx ash osh))
                      (ib (ma.broadcast:project-broadcast-index idx bsh osh))
                      (va (ma.tensor:aref-nd A ia))
                      (vb (ma.tensor:aref-nd B ib))
                      (bx (funcall op va vb)))
                 (setf (aref acc k) (if bx t nil))
                 (incf k))))
           (let ((payload (%bool-vector-to-nested-list acc osh)))
             (ma.tensor:asarray payload :dtype :bool))))
        (t (error "~A: unsupported out-dtype ~S" (or name 'binary-ufunc) out-dtype))))))
