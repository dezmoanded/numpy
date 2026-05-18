;; numpy-speedrun-poc/tests/ufunc-tests.lisp
;; Standalone module-scoped tests for UFUNC (numeric path: add/mul)
;; Run from repo root: sbcl --noinform --script tests/ufunc-tests.lisp

(in-package :cl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %here () (or *load-truename* *compile-file-truename* (truename ".")))
  (defun %dir (p) (make-pathname :name nil :type nil :version nil :defaults p))
  (let* ((here (%here))
         (root (merge-pathnames "../" (%dir here)))
         (src  (merge-pathnames "src/" root)))
    (flet ((load-rel (rel)
             (let ((p (merge-pathnames rel src)))
               (unless (probe-file p)
                 (format t "Could not find ~A~%" p)
                 #+sbcl (sb-ext:exit :code 2))
               (load p))))
      ;; Load CORE
      (load-rel "core/package.lisp")
      (load-rel "core/shape.lisp")
      (load-rel "core/dtype.lisp")
      ;; Load TENSOR
      (load-rel "tensor/package.lisp")
      (load-rel "tensor/ndarray.lisp")
      (load-rel "tensor/indexing.lisp")
      (load-rel "tensor/conversion.lisp")
      ;; Load BROADCAST
      (load-rel "broadcast/package.lisp")
      (load-rel "broadcast/shape.lisp")
      (load-rel "broadcast/projection.lisp")
      (load-rel "broadcast/iterator.lisp")
      ;; Load UFUNC
      (load-rel "ufunc/package.lisp")
      (load-rel "ufunc/engine.lisp")
      (load-rel "ufunc/numeric-ops.lisp")
      ;; Do NOT load src/mini-array.lisp here
      )))

;; Small test helpers
(defun fail (fmt &rest args)
  (apply #'format t fmt args)
  (terpri)
  #+sbcl (sb-ext:exit :code 1))

(defmacro expect-error (&body body)
  `(handler-case (progn ,@body :no-error)
     (error (e) (declare (ignore e)) :errored)))

(defun assert-true (cond &optional (msg ""))
  (unless cond (fail "ASSERT-TRUE failed: ~A" msg)))

(defun assert-equal (got expect &optional (msg ""))
  (unless (equal got expect)
    (fail "ASSERT-EQUAL ~A Expected ~S, got ~S" msg expect got)))

;; Begin tests
(let ((*print-pretty* t))
  ;; Helpers
  (flet ((vec (xs) (mini-array.tensor:asarray xs))
         (to-list (a) (mini-array.tensor:to-list a)))
    ;; 1) Scalar + vector add
    (let* ((a (vec '(1 2 3)))
           (b 10)
           (c (mini-array.ufunc:add a b)))
      (assert-equal (mini-array.tensor:shape-of c) '(3) "shape 1D")
      (assert-equal (mini-array.tensor:dtype-of c) :float64 "dtype float64")
      (assert-equal (to-list c) '(11.0d0 12.0d0 13.0d0)))

    ;; 2) 2D broadcasting add: (1x3) + (2x1) -> (2x3)
    (let* ((row (vec '(1 2 3)))            ; shape (3)
           (col (vec '((10) (20))))         ; shape (2 1)
           (row2d (mini-array.tensor:asarray '((1 2 3)))) ; shape (1 3)
           (out (mini-array.ufunc:add row2d col)))
      (assert-equal (mini-array.tensor:shape-of out) '(2 3))
      (assert-equal (to-list out) '((11.0d0 12.0d0 13.0d0)
                                    (21.0d0 22.0d0 23.0d0))))

    ;; 3) Incompatible shapes should error
    (let* ((x (vec '(1 2 3)))
           (y (vec '(1 2))))
      (assert-true (eq (expect-error (mini-array.ufunc:add x y)) :errored) "shape mismatch errors"))

    ;; 4) Bool-input rejection for numeric ufuncs
    (let* ((nums (vec '(1 2 3)))
           (bools (mini-array.tensor:asarray '(t nil t) :dtype :bool)))
      (assert-true (eq (expect-error (mini-array.ufunc:add nums bools)) :errored) "bool input rejected"))

    ;; 5) 0D + 0D add -> scalar 5.0d0
    (let* ((a 2)
           (b 3)
           (c (mini-array.ufunc:add a b)))
      (assert-equal (mini-array.tensor:shape-of c) nil)
      (assert-equal (mini-array.tensor:to-list c) 5.0d0))

    ;; 6) mul smoke
    (let* ((a (vec '(1 2 3)))
           (b (vec '(10 20 30)))
           (c (mini-array.ufunc:mul a b)))
      (assert-equal (to-list c) '(10.0d0 40.0d0 90.0d0))))

  (format t "OK: UFUNC tests passed~%")
  #+sbcl (sb-ext:exit :code 0))
