;;; numpy-speedrun-poc/tests/test-mini-array.lisp
;;; Run with: sbcl --script tests/test-mini-array.lisp

;; Ensure the library package exists, then enter a test package that :USEs it
(load "src/package.lisp")
(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :mini-array-tests)
    (defpackage :mini-array-tests (:use :cl :mini-array))))
(in-package :mini-array-tests)

;; Load implementation after packages are set up
(load "src/mini-array.lisp")

(defun main ()
  (labels ((assert-true (cond &optional (msg "assert-true failed"))
             (unless cond (error msg)))
           (approx= (a b &optional (eps 1.0d-6))
             (cond
               ((and (numberp a) (numberp b)) (<= (abs (- (coerce a 'double-float)
                                                          (coerce b 'double-float))) eps))
               ((and (null a) (null b)) t)
               ((and (listp a) (listp b))
                (and (= (length a) (length b))
                     (every #'identity (mapcar (lambda (x y) (approx= x y eps)) a b))))
               (t nil)))
           (assert-approx= (x y &optional (eps 1.0d-6))
             (unless (approx= x y eps)
               (error "Approx equal failed: ~S vs ~S" x y)))
           (expect-error (thunk)
             (handler-case (progn (funcall thunk)
                                  (error "Expected an error, but none was signaled"))
               (error (c) (declare (ignore c)) t))))

    ;; default-strides tests
    (assert-true (equal (default-strides nil) nil) "default-strides scalar")
    (assert-true (equal (default-strides '(3)) '(1)) "default-strides 1D")
    (assert-true (equal (default-strides '(2 3)) '(3 1)) "default-strides 2D")

    ;; flat-offset for 2D
    (let* ((shape '(2 3))
           (strides (default-strides shape)))
      (assert-true (= (flat-offset '(0 0) strides shape) 0))
      (assert-true (= (flat-offset '(0 1) strides shape) 1))
      (assert-true (= (flat-offset '(1 2) strides shape) 5)))

    ;; to-list for scalar, 1D, 2D
    (assert-approx= (to-list (asarray 1.25d0)) 1.25d0)
    (assert-approx= (to-list (asarray '(1.0 2.0 3.0))) '(1.0d0 2.0d0 3.0d0))
    (assert-approx=
     (to-list (asarray '((1.0 2.0 3.0)
                         (4.0 5.0 6.0))))
     '((1.0d0 2.0d0 3.0d0)
       (4.0d0 5.0d0 6.0d0)))

    ;; broadcast-shape tests
    (assert-true (equal (broadcast-shape '(2 3) '(3)) '(2 3)))
    (assert-true (equal (broadcast-shape '(2 3) nil) '(2 3)))
    (assert-true (equal (broadcast-shape '(3 1) '(1 4)) '(3 4)))
    (assert-true (expect-error (lambda () (broadcast-shape '(2 3) '(4)))))

    ;; add tests
    (let* ((a (asarray '((1.0 2.0 3.0)
                         (4.0 5.0 6.0))))
           (b (asarray '(10.0 20.0 30.0)))
           (out (add a b)))
      (assert-true (equal (shape-of out) '(2 3)))
      (assert-approx= (to-list out)
                      '((11.0d0 22.0d0 33.0d0)
                        (14.0d0 25.0d0 36.0d0))))

    (let* ((v (asarray '(1.0 2.0 3.0)))
           (out (add v 2.5d0)))
      (assert-true (equal (shape-of out) '(3)))
      (assert-approx= (to-list out) '(3.5d0 4.5d0 5.5d0)))

    (let* ((m (asarray '((1.0 -2.0 3.5)
                         (4.0 5.25 -6.0))))
           (out (add m 2.5d0)))
      (assert-true (equal (shape-of out) '(2 3)))
      (assert-approx= (to-list out)
                      '((3.5d0 0.5d0 6.0d0)
                        (6.5d0 7.75d0 -3.5d0))))

    ;; mul tests
    (let* ((a (asarray '((1.0 2.0 3.0)
                         (4.0 5.0 6.0))))
           (out (mul a 2.5d0)))
      (assert-true (equal (shape-of out) '(2 3)))
      (assert-approx= (to-list out)
                      '((2.5d0 5.0d0 7.5d0)
                        (10.0d0 12.5d0 15.0d0))))

    (let* ((a (asarray '((1.0 2.0 3.0)
                         (4.0 5.0 6.0))))
           (w (asarray '(1.0 0.5 0.25)))
           (out (mul a w)))
      (assert-true (equal (shape-of out) '(2 3)))
      (assert-approx= (to-list out)
                      '((1.0d0 1.0d0 0.75d0)
                        (4.0d0 2.5d0 1.5d0))))

    ;; composition: prices * (1 + tax), then sum-axis axis 1
    (let* ((prices (asarray '((10.0 20.0 30.0)
                               (40.0 50.0 60.0))))
           (tax    (asarray '(0.07 0.08 0.09)))
           (totals (mul prices (add 1.0d0 tax)))
           (row-totals (sum-axis totals 1)))
      (assert-true (equal (shape-of totals) '(2 3)))
      (assert-approx= (to-list totals)
                      '((10.7d0 21.6d0 32.7d0)
                        (42.8d0 54.0d0 65.4d0)))
      (assert-true (equal (shape-of row-totals) '(2)))
      (assert-approx= (to-list row-totals)
                      '(65.0d0 162.2d0)))

    ;; reductions
    (let* ((a (asarray '((1.0 2.0 3.0)
                         (4.0 5.0 6.0))))
           (out0 (sum-axis a 0))
           (out1 (sum-axis a 1)))
      (assert-true (equal (shape-of out0) '(3)))
      (assert-approx= (to-list out0) '(5.0d0 7.0d0 9.0d0))
      (assert-true (equal (shape-of out1) '(2)))
      (assert-approx= (to-list out1) '(6.0d0 15.0d0)))

    ;; errors
    (let* ((a (asarray '((1.0 2.0 3.0)
                         (4.0 5.0 6.0))))
           (b (asarray '(1.0 2.0))))
      (assert-true (expect-error (lambda () (add a b)))))
    (let ((a (asarray '((1.0 2.0 3.0)
                         (4.0 5.0 6.0)))))
      (assert-true (expect-error (lambda () (sum-axis a -1))))
      (assert-true (expect-error (lambda () (sum-axis a 2)))))
    (let ((v (asarray '(1.0 2.0 3.0))))
      (assert-true (expect-error (lambda () (sum-axis v 0)))))

    (format t "All tests passed.~%"))
  t)

(main)