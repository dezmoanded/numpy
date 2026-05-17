;;; SBCL --script runnable tests for TENSOR subgroup
;;; Run from project root: sbcl --script tests/tensor-tests.lisp

(load "src/package.lisp")
(load "src/tensor.lisp")

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defmacro expect-error (&body body)
    `(handler-case (progn ,@body nil)
       (error (e) (declare (ignore e)) t)))

  (defun double-list-p (xs)
    (every (lambda (x) (typep x 'double-float)) xs))

  (defun deep-double-list-p (xss)
    (every (lambda (row) (every (lambda (x) (typep x 'double-float)) row)) xss)))

(let ((*print-pretty* t))
  ;; 0D scalar
  (let ((a (mini-array:asarray 7)))
    (assert (equal nil (mini-array:shape-of a)))
    (assert (equal nil (mini-array:strides-of a)))
    (assert (typep (mini-array:to-list a) 'double-float)))

  ;; 1D happy path
  (let ((a (mini-array:asarray '(1 2 3))))
    (assert (equal '(3) (mini-array:shape-of a)))
    (assert (equal '(1) (mini-array:strides-of a)))
    (let ((lst (mini-array:to-list a)))
      (assert (equal '(1.0d0 2.0d0 3.0d0) lst))
      (assert (double-list-p lst))))

  ;; 2D rectangular happy path
  (let ((a (mini-array:asarray '((1 2) (3 4)))))
    (assert (equal '(2 2) (mini-array:shape-of a)))
    (assert (equal '(2 1) (mini-array:strides-of a)))
    (let ((mat (mini-array:to-list a)))
      (assert (equal '((1.0d0 2.0d0) (3.0d0 4.0d0)) mat))
      (assert (deep-double-list-p mat))))

  ;; asarray errors: empty, ragged, non-numeric, rank>2
  (assert (expect-error (mini-array:asarray '())))
  (assert (expect-error (mini-array:asarray '((1 2) (3)))))
  (assert (expect-error (mini-array:asarray '(1 a 3))))
  (assert (expect-error (mini-array:asarray '(((1))))))

  ;; flat-offset happy path and bounds errors; aref-nd wraps it
  (let* ((a (mini-array:asarray '((1 2) (3 4))))
         (shape (mini-array:shape-of a))
         (strides (mini-array:strides-of a)))
    (assert (= 0 (mini-array:flat-offset '(0 0) strides shape)))
    (assert (= 3 (mini-array:flat-offset '(1 1) strides shape)))
    (assert (expect-error (mini-array:flat-offset '(2 0) strides shape)))
    (assert (= 1.0d0 (mini-array:aref-nd a '(0 0))))
    (assert (= 4.0d0 (mini-array:aref-nd a '(1 1))))
    (assert (expect-error (mini-array:aref-nd a '(2 0)))))

  ;; make-array-from-flat happy and error
  (let ((a (mini-array:make-array-from-flat '(2 3) '(1 2 3 4 5 6))))
    (assert (equal '(2 3) (mini-array:shape-of a)))
    (assert (equal '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0)) (mini-array:to-list a))))
  (assert (expect-error (mini-array:make-array-from-flat '(2 2) '(1 2 3))))

  (format t "TENSOR TESTS PASS~%")
  #+sbcl (sb-ext:quit :unix-status 0)
  )
