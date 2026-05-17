;;;; numpy-speedrun-poc/src/tensor.lisp
;;;; Tensor: ndarray struct, constructors, indexing, materialization (V1)

(in-package :mini-array)

;;; Self-load minimal dependency: core.lisp (dir-relative)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (let* ((here (or *load-truename* *compile-file-truename*))
         (dir  (and here (make-pathname :name nil :type nil :version nil :defaults here))))
    (when dir
      (load (merge-pathnames "core.lisp" dir)))))

;;; Data model
(defstruct ndarray
  shape    ;; NIL | (n) | (r c)
  strides  ;; NIL | (1) | (c 1) (element-based)
  dtype    ;; :float64 (symbol)
  data)    ;; simple-array of double-float

(defun %flatten-2d-list (rows)
  (let* ((r (length rows))
         (c (length (first rows)))
         (out (make-array (* r c) :element-type 'double-float)))
    (loop for i from 0 below r do
      (let ((row (nth i rows)))
        (loop for j from 0 below c do
          (let ((x (nth j row)))
            (setf (aref out (+ (* i c) j)) (%ensure-double x))))))
    out))

(defun make-array-from-flat (shape data)
  "Create an ndarray from SHAPE and flat DATA (list of numbers).
Validates rank <= 2, dims > 0, product(shape) == length(data). Dtype fixed :float64."
  (unless (valid-shape-p shape)
    (error "make-array-from-flat: invalid shape (rank>2 or non-positive dims): ~S" shape))
  (let* ((n (product shape))
         (data-list (if (listp data) data (error "DATA must be a list for V1")))
         (len (length data-list)))
    (unless (= n len)
      (error "make-array-from-flat: data length ~A does not match product(shape)=~A for shape ~S" len n shape))
    (let ((vec (%make-double-vector len)))
      (loop for i from 0 below len
            for x in data-list do
              (setf (aref vec i) (%ensure-double x)))
      (make-ndarray :shape (copy-list shape)
                    :strides (default-strides shape)
                    :dtype :float64
                    :data vec))))

(defun asarray (obj)
  "Normalize OBJ to an ndarray.
- number -> 0D array (shape NIL)
- existing ndarray -> returned as-is
- 1D list of numbers -> shape (n)
- 2D list of numbers (rectangular) -> shape (r c)
Errors on empty lists, ragged lists, non-numeric elements, rank>2."
  (cond
    ((typep obj 'ndarray) obj)
    ((numberp obj)
      (let ((v (%make-double-vector 1)))
        (setf (aref v 0) (%ensure-double obj))
        (make-ndarray :shape nil :strides nil :dtype :float64 :data v)))
    ((listp obj)
      (when (null obj) (error "asarray: empty list not allowed in V1"))
      (let* ((has-list (some #'listp obj))
             (has-nonlist (some (lambda (x) (not (listp x))) obj)))
        (when (and has-list has-nonlist)
          (error "asarray: mixed list/non-list elements not allowed (ragged)"))
        (if has-list
            ;; 2D
            (let* ((rows obj)
                   (r (length rows))
                   (c (length (first rows))))
              (when (or (= r 0) (= c 0))
                (error "asarray: zero-sized dimensions excluded in V1"))
              (unless (every #'listp rows)
                (error "asarray: all rows must be lists"))
              (unless (every (lambda (row) (= (length row) c)) rows)
                (error "asarray: ragged 2D list (rows have differing lengths)"))
              (unless (every (lambda (row) (every #'numberp row)) rows)
                (error "asarray: non-numeric element in 2D list"))
              (let ((vec (%flatten-2d-list rows)))
                (make-ndarray :shape (list r c)
                              :strides (default-strides (list r c))
                              :dtype :float64
                              :data vec)))
            ;; 1D
            (let* ((xs obj)
                   (n (length xs)))
              (when (= n 0) (error "asarray: zero-sized dimensions excluded in V1"))
              (unless (every #'numberp xs)
                (error "asarray: non-numeric element in 1D list"))
              (make-array-from-flat (list n) xs)))))
    (t (error "asarray: unsupported type ~S" (type-of obj)))))

(defun shape-of (a)
  (copy-list (ndarray-shape a)))

(defun strides-of (a)
  (copy-list (ndarray-strides a)))

(defun flat-offset (index strides &optional shape)
  "Compute flat offset from INDEX and STRIDES; if SHAPE provided, check bounds. INDEX and SHAPE are list or NIL for 0D."
  (cond
    ((and (null strides) (null index)) 0)
    ((/= (length index) (length strides))
      (error "flat-offset: index rank ~A does not match strides rank ~A" (length index) (length strides)))
    (t
      (when shape
        (unless (= (length index) (length shape))
          (error "flat-offset: index rank ~A does not match shape rank ~A" (length index) (length shape)))
        (loop for k from 0 below (length shape) do
          (let ((i (nth k index))
                (n (nth k shape)))
            (when (or (minusp i) (>= i n))
              (error "flat-offset: index ~S out of bounds for shape ~S" index shape)))))
      (let ((off 0))
        (loop for k from 0 below (length strides) do
          (incf off (* (nth k index) (nth k strides))))
        off))))

(defun aref-nd (a index)
  (let* ((shape (ndarray-shape a))
         (strides (ndarray-strides a))
         (off (flat-offset index strides shape)))
    (aref (ndarray-data a) off)))

(defun %set-aref-nd (a index value)
  (let* ((shape (ndarray-shape a))
         (strides (ndarray-strides a))
         (off (flat-offset index strides shape)))
    (setf (aref (ndarray-data a) off) (%ensure-double value))))

(defun to-list (a)
  (let ((shape (ndarray-shape a))
        (data (ndarray-data a)))
    (cond
      ((null shape) (aref data 0))
      ((= (length shape) 1)
        (let* ((n (first shape))
               (out '()))
          (dotimes (i n (nreverse out))
            (push (aref data i) out))))
      ((= (length shape) 2)
        (let* ((r (first shape))
               (c (second shape))
               (rows '()))
          (dotimes (i r (nreverse rows))
            (let ((row '()))
              (dotimes (j c)
                (push (aref data (+ (* i c) j)) row))
              (push (nreverse row) rows)))))
      (t (error "to-list: rank > 2 not supported")))))
