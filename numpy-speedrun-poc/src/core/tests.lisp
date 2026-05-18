;;;; CORE module tests (run with: sbcl --script src/core/tests.lisp)

(defun %dir-of (path)
  (make-pathname :name nil :type nil :defaults path))

(let* ((here (or *load-pathname* *compile-file-pathname*))
       (dir (%dir-of here)))
  (load (merge-pathnames "package.lisp" dir))
  (load (merge-pathnames "shape.lisp" dir))
  (load (merge-pathnames "dtype.lisp" dir)))

(use-package :mini-array.core)

(defun check (cond &optional (msg "check failed"))
  (unless cond (error msg)))

(defun check-error (thunk)
  (let ((ok nil))
    (handler-case (progn (funcall thunk) (setf ok t))
      (error () (setf ok nil)))
    (when ok (error "Expected error, but call succeeded"))
    t))

;; shape tests
(check (= 1 (product nil)) "product NIL => 1")
(check (= 6 (product '(2 3))) "product (2 3) => 6")
(check (= 0 (rank nil)) "rank NIL => 0")
(check (= 2 (rank '(2 3))) "rank (2 3) => 2")
(check (valid-shape-p nil) "valid-shape-p NIL")
(check (valid-shape-p '(2)) "valid-shape-p (2)")
(check (not (valid-shape-p '(0))) "invalid zero-sized dim")
(check (not (valid-shape-p '(2 0))) "invalid zero-sized dim (rank 2)")
(check (not (valid-shape-p '(2 3 4))) "invalid rank > 2")
(check (equal nil (default-strides nil)) "default-strides NIL => NIL")
(check (equal '(1) (default-strides '(5))) "default-strides (5) => (1)")
(check (equal '(3 1) (default-strides '(2 3))) "default-strides (2 3) => (3 1)")
(check-error (lambda () (default-strides '(0))))

;; dtype tests
(check (dtype-p :float64))
(check (dtype-p :bool))
(check (not (dtype-p :int32)))

(let ((x (coerce-dtype-value 3 :float64)))
  (check (typep x 'double-float))
  (check (= x 3.0d0)))
(let ((x (coerce-dtype-value 1/2 :float64)))
  (check (typep x 'double-float))
  (check (= x 0.5d0)))
(check-error (lambda () (coerce-dtype-value #c(1 2) :float64)))
(check-error (lambda () (coerce-dtype-value "3" :float64)))

(check (eq t (coerce-dtype-value t :bool)))
(check (eq nil (coerce-dtype-value nil :bool)))
(check-error (lambda () (coerce-dtype-value 1 :bool)))
(check-error (lambda () (coerce-dtype-value :x :bool)))

(check (eql :bool (infer-scalar-dtype t)))
(check (eql :bool (infer-scalar-dtype nil)))
(check (eql :float64 (infer-scalar-dtype 1)))
(check (eql :float64 (infer-scalar-dtype 1.0)))
(check-error (lambda () (infer-scalar-dtype "x")))

(format t "CORE tests PASSED~%")
