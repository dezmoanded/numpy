;;;; numpy-speedrun-poc/src/tensor/ndarray.lisp
;; Data model and constructors for ndarray

(in-package :mini-array.tensor)

;; Local dtype helpers (core/dtype.lisp not present yet)
(defun %dtype-p (x)
  (or (eql x :float64) (eql x :bool)))

(defun %boolp (x) (or (null x) (eql x t)))

(defun %ensure-double (x)
  (coerce x 'double-float))

(defstruct ndarray
  (shape nil :type list)
  (strides nil :type list)
  (dtype :float64 :type symbol)
  (data #() :type simple-vector))

(defun shape-of (a) (ndarray-shape a))
(defun strides-of (a) (ndarray-strides a))
(defun dtype-of (a) (ndarray-dtype a))

(defun %validate-nd-args (shape dtype data)
  (unless (valid-shape-p shape)
    (error "ndarray: invalid shape ~S (rank<=2, positive)" shape))
  (unless (%dtype-p dtype)
    (error "ndarray: invalid dtype ~S (only :float64 or :bool)" dtype))
  (let ((n (mini-array.core:product shape)))
    (unless (= n (length data))
      (error "ndarray: product(shape)=~A does not equal data length ~A" n (length data))))
  (let ((exp (default-strides shape)))
    (declare (ignore exp))
    ;; Strides will be set to default by make-array-from-flat
    t))

;; Accept 2-arg or 3-arg forms:
;;  - (make-array-from-flat shape flat-data)           ; dtype defaults to :float64
;;  - (make-array-from-flat shape dtype flat-data)
(defun make-array-from-flat (shape arg2 &optional arg3)
  "Create an ndarray from SHAPE and FLAT-DATA, optionally specifying DTYPE.
  - Validates rank<=2, product(shape)=len(data), dtype in {:float64,:bool}.
  - Enforces default row-major contiguous strides.
  - Coerces elements per dtype: double-float for :float64; T/NIL for :bool."
  (let* ((shape (or shape nil))
         (dtype (if arg3 (or arg2 :float64) :float64))
         (flat-data (if arg3 arg3 arg2))
         (flat-list (cond
                      ((vectorp flat-data) (coerce flat-data 'list))
                      ((listp flat-data) flat-data)
                      (t (error "make-array-from-flat: FLAT-DATA must be list or vector, got ~S" (type-of flat-data)))))
         (n (mini-array.core:product shape)))
    (%validate-nd-args shape dtype flat-list)
    (let ((vec (make-array n :element-type t)))
      (cond
        ((eql dtype :float64)
         (loop for i from 0 below n
               for v in flat-list do
                 (unless (realp v)
                   (error "make-array-from-flat: non-numeric element ~S for :float64" v))
                 (setf (aref vec i) (%ensure-double v))))
        ((eql dtype :bool)
          (loop for i from 0 below n
                for v in flat-list do
                  (unless (%boolp v)
                    (error "make-array-from-flat: non-boolean element ~S for :bool" v))
                  (setf (aref vec i) (if v t nil))))
        (t (error "unsupported dtype ~S" dtype)))
      (make-ndarray :shape shape
                    :strides (default-strides shape)
                    :dtype dtype
                    :data vec))))
