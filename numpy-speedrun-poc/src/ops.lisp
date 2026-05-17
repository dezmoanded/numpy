;;;; numpy-speedrun-poc/src/ops.lisp
;;;; OPS: broadcasting helpers + elementwise add/mul with broadcasting (V1)
;;;; V1: float64 only; ranks 0/1/2; row-major contiguous; positive strides; no views.

(in-package :mini-array)

;;; Self-load minimal dependencies (dir-relative): core.lisp, tensor.lisp
(eval-when (:compile-toplevel :load-toplevel :execute)
  (let* ((here (or *load-truename* *compile-file-truename*))
         (dir  (and here (make-pathname :name nil :type nil :version nil :defaults here))))
    (when dir
      (load (merge-pathnames "core.lisp" dir))
      (load (merge-pathnames "tensor.lisp" dir)))))

;;; Broadcasting helpers (right-aligned, NumPy-like)
(defun %pad-left-with-ones (s r)
  (let ((pad (- r (length s))))
    (if (<= pad 0)
        (copy-list s)
        (nconc (make-list pad :initial-element 1)
               (copy-list s)))))

(defun broadcast-shape (sa sb)
  (let* ((ra (length sa))
         (rb (length sb))
         (r (max ra rb))
         (pa (%pad-left-with-ones sa r))
         (pb (%pad-left-with-ones sb r))
         (out '()))
    (dotimes (i r (nreverse out))
      (let ((a (nth i pa))
            (b (nth i pb)))
        (cond
          ((= a b) (push a out))
          ((= a 1) (push b out))
          ((= b 1) (push a out))
          (t (error "broadcast-shape: incompatible shapes ~S and ~S" sa sb)))))))

(defun all-indices (shape)
  (cond
    ((null shape) (list nil))
    ((= (length shape) 1)
      (let* ((n (first shape))
             (out '()))
        (dotimes (i n (nreverse out))
          (push (list i) out))))
    ((= (length shape) 2)
      (let* ((r (first shape))
             (c (second shape))
             (out '()))
        (dotimes (i r)
          (dotimes (j c)
            (push (list i j) out)))
        (nreverse out)))
    (t (error "all-indices: rank > 2 not supported in V1"))))

(defun project-broadcast-index (out-idx in-shape out-shape)
  (let* ((r-out (length out-shape))
         (r-in (length in-shape))
         (pad (- r-out r-in))
         (in-pad (%pad-left-with-ones in-shape r-out))
         (proj '()))
    (dotimes (i r-out)
      (let ((in-d (nth i in-pad))
            (oi (nth i out-idx)))
        (push (if (= in-d 1) 0 oi) proj)))
    ;; drop the padded leading components
    (subseq (nreverse proj) pad)))

;;; Ufuncs
(defun %ensure-ndarray (x)
  (if (typep x 'ndarray) x (asarray x)))

(defun binary-ufunc (a b op)
  "Apply elementwise OP over A and B with NumPy-style broadcasting (V1: rank<=2).
A and B can be ndarrays or scalars. Returns a new contiguous ndarray."
  (let* ((aa (%ensure-ndarray a))
         (bb (%ensure-ndarray b))
         (sa (ndarray-shape aa))
         (sb (ndarray-shape bb))
         (out-shape (broadcast-shape sa sb))
         (n (product out-shape))
         (out-data (%make-double-vector (if (listp out-shape) n 1) 0.0d0))
         (out (make-ndarray :shape (and (consp out-shape) (copy-list out-shape))
                            :strides (default-strides out-shape)
                            :dtype :float64
                            :data out-data)))
    (dolist (idx (all-indices out-shape) out)
      (let* ((ia (project-broadcast-index idx sa out-shape))
             (ib (project-broadcast-index idx sb out-shape))
             (va (aref-nd aa ia))
             (vb (aref-nd bb ib))
             (vz (funcall op va vb))
             (off (flat-offset idx (ndarray-strides out) (ndarray-shape out))))
        (setf (aref out-data off) (%ensure-double vz))))))

(defun add (a b)
  (binary-ufunc a b #'+))

(defun mul (a b)
  (binary-ufunc a b #'*))
