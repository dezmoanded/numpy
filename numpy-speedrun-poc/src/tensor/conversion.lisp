;;;; numpy-speedrun-poc/src/tensor/conversion.lisp
;; Conversions: asarray, to-list

(in-package :mini-array.tensor)

(defun %list-all-p (xs pred)
  (and (listp xs) (every pred xs)))

(defun %numericp (x) (realp x))

(defun %boolp (x) (or (null x) (eql x t)))

(defun %rectangular-2d-shape (rows)
  "If ROWS is a non-empty list of equal-length non-empty lists, return (r c);
else return NIL."
  (when (and (listp rows) rows (every #'listp rows))
    (let* ((r (length rows))
           (c (length (first rows))))
      (when (and (> c 0) (every (lambda (row) (= (length row) c)) rows))
        (list r c)))))

(defun %flatten-2d (rows)
  (apply #'append rows))

(defun asarray (obj &key dtype)
  "Create an ndarray from OBJ.
Rules:
- If OBJ is already an ndarray, return it unchanged; if DTYPE is given and does not match, error.
- Scalars: real number -> :float64 (default). With :dtype :bool accept T/NIL only; mismatch errors.
- 1D: list of numbers -> :float64; with :dtype :bool accept list of T/NIL; boolean list without dtype errors.
- 2D: rectangular list of numeric rows only; dtype forced to :float64 (error if :bool requested).
- Zero-sized dims are not allowed."
  (cond
    ;; Already an ndarray: idempotent
    ((ndarray-p obj)
     (when dtype
       (let ((cur (ndarray-dtype obj)))
         (unless (eql dtype cur)
           (error "asarray: cannot coerce ndarray dtype ~S to ~S in V1" cur dtype))))
     obj)
    ;; Scalar number
    ((realp obj)
     (when (and dtype (not (eql dtype :float64)))
       (error "asarray: numeric scalar with non-:float64 dtype ~S" dtype))
     (make-array-from-flat nil :float64 (list obj)))
    ;; Scalar boolean requires explicit :bool
    ((or (null obj) (eql obj t))
      (unless (eql dtype :bool)
        (error "asarray: boolean scalar requires :dtype :bool"))
      (make-array-from-flat nil :bool (list (if obj t nil))))
    ;; 1D list
    ((and (listp obj) (or (null (first obj)) (realp (first obj)) (eql (first obj) t)))
     (when (null obj)
       (error "asarray: empty 1D list not allowed (zero-sized dims disallowed)"))
     (let ((all-bool (every #'%boolp obj))
           (all-num  (every #'%numericp obj)))
       (cond
         (all-num
          (when (and dtype (not (eql dtype :float64)))
            (error "asarray: 1D numeric with non-:float64 dtype ~S" dtype))
          (make-array-from-flat (list (length obj)) :float64 obj))
         (all-bool
          (unless (eql dtype :bool)
            (error "asarray: boolean 1D list requires :dtype :bool"))
          (make-array-from-flat (list (length obj)) :bool obj))
         (t (error "asarray: 1D list must be all numbers or all booleans")))))
    ;; 2D list
    ((and (listp obj) (every #'listp obj))
     (let ((shape (%rectangular-2d-shape obj)))
       (unless shape
         (error "asarray: 2D input must be non-empty rectangular rows"))
       (let* ((r (first shape)) (c (second shape))
              (rows obj))
         (declare (ignore r c))
         (unless (every (lambda (row) (every #'%numericp row)) rows)
           (error "asarray: 2D input must be numeric only"))
         (when dtype
           (unless (eql dtype :float64)
             (error "asarray: 2D only supports :float64 dtype in V1")))
         (make-array-from-flat shape :float64 (%flatten-2d rows)))))
    (t (error "asarray: unsupported input ~S" obj))))

(defun to-list (a)
  "Return OBJ representation mirroring rank: number for 0D; list for 1D; list of lists for 2D."
  (let* ((shape (shape-of a))
         (data (ndarray-data a)))
    (case (length shape)
      (0 (aref data 0))
      (1 (let ((n (first shape)))
           (loop for i from 0 below n collect (aref data i))))
      (2 (destructuring-bind (r c) shape
           (loop for i from 0 below r collect
                 (loop for j from 0 below c collect
                       (aref data (+ (* i c) j))))))
      (t (error "to-list: rank>2 not supported")))))
