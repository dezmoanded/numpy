;;; numpy-speedrun-poc/tests/broadcast-tests.lisp
;;; Run with: sbcl --noinform --script tests/broadcast-tests.lisp

(in-package :cl)

(defun %here () (or *load-truename* *compile-file-truename* (truename ".")))
(defun %dir (p) (make-pathname :name nil :type nil :version nil :defaults p))

(let* ((here (%here))
       (root (merge-pathnames "../" (%dir here)))
       (src  (merge-pathnames "src/" root)))
  ;; core (package + impls used by broadcast)
  (load (merge-pathnames "core/package.lisp" src))
  (load (merge-pathnames "core/shape.lisp" src))
  (load (merge-pathnames "core/dtype.lisp" src))
  ;; broadcast module
  (load (merge-pathnames "broadcast/package.lisp" src))
  (load (merge-pathnames "broadcast/shape.lisp" src))
  (load (merge-pathnames "broadcast/projection.lisp" src))
  (load (merge-pathnames "broadcast/iterator.lisp" src)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :broadcast-tests)
    (defpackage :broadcast-tests
      (:use :cl :mini-array.core :mini-array.broadcast))))
(in-package :broadcast-tests)

(defun expect-error (thunk)
  (handler-case (progn (funcall thunk)
                       (error "Expected an error, but none was signaled"))
    (error (c) (declare (ignore c)) t)) )

(defun approx= (a b &optional (eps 1.0d-6))
  (cond
    ((and (numberp a) (numberp b)) (<= (abs (- (coerce a 'double-float)
                                              (coerce b 'double-float))) eps))
    ((and (null a) (null b)) t)
    ((and (listp a) (listp b))
     (and (= (length a) (length b))
          (every #'identity (mapcar (lambda (x y) (approx= x y eps)) a b))))
    (t nil)))

(defun main ()
  ;; broadcast-shape basics
  (assert (equal (broadcast-shape nil '(3)) '(3)))
  (assert (equal (broadcast-shape '(2 1) '(1 3)) '(2 3)))
  (assert (equal (broadcast-shape '(3) '(2 3)) '(2 3)))
  (assert (expect-error (lambda () (broadcast-shape '(2) '(3)))))
  ;; invalid shapes
  (assert (expect-error (lambda () (broadcast-shape '(0) '(3)))))
  (assert (expect-error (lambda () (broadcast-shape '(1 1 1) '(3)))))

  ;; all-indices
  (assert (equal (all-indices nil) (list nil)))
  (assert (equal (all-indices '(3)) '((0) (1) (2))))
  (assert (equal (all-indices '(2 3))
                 '((0 0) (0 1) (0 2)
                   (1 0) (1 1) (1 2))))
  (assert (expect-error (lambda () (all-indices '(2 0)))))
  (assert (expect-error (lambda () (all-indices '(1 1 1)))))

  ;; project-broadcast-index
  (assert (equal (project-broadcast-index '(1 2) '(1 3) '(2 3)) '(0 2)))
  (assert (equal (project-broadcast-index '(1 2) '(2 1) '(2 3)) '(1 0)))
  ;; 0D in to 1D/2D out
  (assert (equal (project-broadcast-index '(0) nil '(3)) nil))
  (assert (equal (project-broadcast-index '(1 2) nil '(2 3)) nil))
  ;; Out-index length mismatch
  (assert (expect-error (lambda () (project-broadcast-index '(0 1) '(3) '(3)))))
  ;; Not broadcastable
  (assert (expect-error (lambda () (project-broadcast-index '(0) '(2) '(3)))))

  (format t "broadcast-tests.lisp: All tests passed.~%")
  t)

(main)
