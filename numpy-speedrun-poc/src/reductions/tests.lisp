;;;; Run with: sbcl --script src/reductions/tests.lisp

(in-package :cl)

;; Load minimal upstream modules (no src/mini-array.lisp)
(load "src/core/package.lisp")
(load "src/core/shape.lisp")
(load "src/core/dtype.lisp")

(load "src/tensor/package.lisp")
(load "src/tensor/ndarray.lisp")
(load "src/tensor/indexing.lisp")
(load "src/tensor/conversion.lisp")

(load "src/reductions/package.lisp")
(load "src/reductions/sum.lisp")

(defun assert-equal (x y)
  (unless (equal x y)
    (error "Assertion failed: ~S =/= ~S" x y)))

(defun expect-error (thunk)
  (let ((ok t))
    (handler-case
        (progn (funcall thunk))
      (error () (setf ok nil)))
    (when ok
      (error "Expected error but none was signaled"))))

;; Happy-path numeric 2x3
(let* ((a (mini-array.tensor:asarray '((1 2 3) (4 5 6))))
       (s0 (mini-array.reductions:sum-axis a 0))
       (s1 (mini-array.reductions:sum-axis a 1)))
  (assert-equal '(5.0d0 7.0d0 9.0d0) (mini-array.tensor:to-list s0))
  (assert-equal '(6.0d0 15.0d0) (mini-array.tensor:to-list s1)))

;; Error: non-2D input
(expect-error (lambda () (mini-array.reductions:sum-axis (mini-array.tensor:asarray '(1 2 3)) 0)))

;; Error: invalid axis
(expect-error (lambda () (mini-array.reductions:sum-axis (mini-array.tensor:asarray '((1 2) (3 4))) 2)))

(format t "reductions/tests.lisp OK~%")
