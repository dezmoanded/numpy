;;;; mini-array.core dtype utilities

(in-package :mini-array.core)

(defun dtype-p (x)
  "Return T iff X is a supported dtype keyword (:float64 or :bool)."
  (or (eql x :float64)
      (eql x :bool)))

(defun %ensure-supported-dtype (dtype)
  (unless (dtype-p dtype)
    (error "Unsupported dtype: ~S (allowed: :float64, :bool)" dtype)))

(defun coerce-dtype-value (value dtype)
  "Coerce VALUE according to DTYPE.
  :float64 => double-float; accept real numbers; reject complex/non-numeric.
  :bool    => return T or NIL only; do not coerce numbers or other objects."
  (%ensure-supported-dtype dtype)
  (case dtype
    (:float64
     (cond
       ((realp value) (coerce value 'double-float))
       (t (error "coerce-dtype-value: not a real number for :float64: ~S" value))))
    (:bool
     (cond
       ((eq value t) t)
       ((null value) nil)
       (t (error "coerce-dtype-value: only T or NIL allowed for :bool, got ~S" value))))
    (t (error "Internal error: unexpected dtype ~S" dtype))))

(defun infer-scalar-dtype (value)
  "Infer dtype for a scalar VALUE. T/NIL => :bool; real number => :float64; else error."
  (cond
    ((typep value 'boolean) :bool)
    ((realp value) :float64)
    (t (error "infer-scalar-dtype: unsupported scalar ~S" value))))
