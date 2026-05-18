;;;; numpy-speedrun-poc/src/indexing/boolean-select.lisp
(in-package :mini-array.indexing)

(defun %booleanp (x)
  (or (eq x t) (null x)))

(defun boolean-select (array mask)
  "Select 1D elements of ARRAY by a 1D boolean MASK (list of T/NIL).

Constraints (V1):
- ARRAY must be an :mini-array ndarray of rank 1 and dtype :float64 (numeric only for now).
- MASK must be a proper list of booleans (T/NIL) with length equal to ARRAY length.
- Result preserves order and dtype (numeric float64 for now).
- All-false MASK is an error (zero-sized outputs are globally excluded in V1).

Returns a new 1D ndarray."
  (declare (type list mask))
  ;; Validate array rank 1
  (unless (typep array 'mini-array:ndarray)
    (error "boolean-select: ARRAY must be an ndarray, got ~S" (type-of array)))
  (let* ((shape (mini-array:shape-of array))
         (r (length shape)))
    (unless (= r 1)
      (error "boolean-select: only rank-1 arrays supported, got rank ~A with shape ~S" r shape))
    (let* ((n (first shape)))
      ;; Validate mask
      (unless (and (listp mask) (= (length mask) n))
        (error "boolean-select: MASK must be a list of length ~A (got ~S)" n (length mask)))
      (unless (every #'%booleanp mask)
        (error "boolean-select: MASK must contain only T/NIL"))
      ;; Count trues and collect selected values
      (let ((count 0)
            (out-values '()))
        (dotimes (i n)
          (when (nth i mask)
            (incf count)
            (push (mini-array:aref-nd array (list i)) out-values)))
        (when (= count 0)
          (error "boolean-select: all-false MASK would yield empty selection, disallowed in V1"))
        (setf out-values (nreverse out-values))
        (mini-array:make-array-from-flat (list count) out-values)))))
