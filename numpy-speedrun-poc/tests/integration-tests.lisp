;;;; numpy-speedrun-poc/tests/integration-tests.lisp
;;;; Run with: sbcl --script numpy-speedrun-poc/tests/integration-tests.lisp

(in-package :cl)

;; Bootstrap: load shared test utilities relative to this file
(defun %here () (or *load-truename* *compile-file-truename* (truename ".")))
(defun %dir (p) (make-pathname :name nil :type nil :version nil :defaults p))
(let* ((here (%here))
       (tests-dir (%dir here)))
  (load (merge-pathnames "util.lisp" tests-dir)))

;; Use the shared integration loader
(mini-array.test-util:load-integration)

;; Test package uses :mini-array after it exists, plus test-utils
(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :mini-array.int-tests)
    (defpackage :mini-array.int-tests (:use :cl :mini-array :mini-array.test-util))))
(in-package :mini-array.int-tests)

(defun main ()
  ;; 1) Scalar + vector add
  (let* ((a (asarray '(1 2 3)))
         (c (add a 10)))
    (assert-approx= '(11 12 13) (to-list c) 1.0d-12))

  ;; 2) Broadcasting 1x3 + 2x1 -> 2x3
  (let* ((a (asarray '((1 2 3))))
         (b (asarray '((10) (20))))
         (c (add a b)))
    (assert-approx= '((11 12 13) (21 22 23)) (to-list c) 1.0d-12))

  ;; 3) sum-axis along axis 0 and 1
  (let* ((m (asarray '((1 2 3) (4 5 6))))
         (s0 (sum-axis m 0))
         (s1 (sum-axis m 1)))
    (assert-approx= '(5 7 9) (to-list s0) 1.0d-12)
    (assert-approx= '(6 15) (to-list s1) 1.0d-12))

  ;; 4) Composition: sum-axis(add(A,B), 1)
  (let* ((a (asarray '((1 2) (3 4))))
         (b (asarray '((5 6) (7 8))))
         (s (add a b))
         (row-sum (sum-axis s 1)))
    (assert-approx= '(14 22) (to-list row-sum) 1.0d-12))

  (format t "Integration tests passed.~%")
  t)

(main)
