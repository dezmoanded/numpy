;; numpy-speedrun-poc/tests/core-tests.lisp
;; Standalone module-scoped tests for CORE
;; Run from repo root: sbcl --noinform --script tests/core-tests.lisp

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
      ;; Load CORE (packages then impl) — do NOT load src/mini-array.lisp here
      (load-rel "core/package.lisp")
      (load-rel "core/shape.lisp")
      (load-rel "core/dtype.lisp")
      ;; Optionally expose :mini-array re-exports if needed (not required here):
      ;; (load (merge-pathnames "package.lisp" src))
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
  ;; product
  (assert-equal (mini-array.core:product nil) 1 "product nil -> 1")
  (assert-equal (mini-array.core:product '(2 3)) 6 "product 2x3 -> 6")

  ;; rank
  (assert-equal (mini-array.core:rank nil) 0)
  (assert-equal (mini-array.core:rank '(5)) 1)
  (assert-equal (mini-array.core:rank '(2 3)) 2)

  ;; valid-shape-p
  (assert-true (mini-array.core:valid-shape-p nil) "0D allowed")
  (assert-true (mini-array.core:valid-shape-p '(2)))
  (assert-true (mini-array.core:valid-shape-p '(2 3)))
  (assert-true (not (mini-array.core:valid-shape-p '(0))))
  (assert-true (not (mini-array.core:valid-shape-p '(2 0))))
  (assert-true (not (mini-array.core:valid-shape-p '(1 2 3))))

  ;; default-strides
  (assert-equal (mini-array.core:default-strides nil) nil)
  (assert-equal (mini-array.core:default-strides '(5)) '(1))
  (assert-equal (mini-array.core:default-strides '(2 3)) '(3 1))
  (let ((res (expect-error (mini-array.core:default-strides '(1 2 3)))))
    (assert-true (eq res :errored) "rank>2 should error"))

  ;; dtype-p
  (assert-true (mini-array.core:dtype-p :float64))
  (assert-true (mini-array.core:dtype-p :bool))
  (assert-true (not (mini-array.core:dtype-p :int32)))
  (assert-true (not (mini-array.core:dtype-p 'foo)))

  ;; coerce-dtype-value :float64
  (let ((x (mini-array.core:coerce-dtype-value 1 :float64))
        (y (mini-array.core:coerce-dtype-value 2.5d0 :float64)))
    (assert-true (typep x 'double-float))
    (assert-true (typep y 'double-float))
    (assert-equal x 1.0d0)
    (assert-equal y 2.5d0))
  (assert-true (eq (expect-error (mini-array.core:coerce-dtype-value #c(1 2) :float64)) :errored))
  (assert-true (eq (expect-error (mini-array.core:coerce-dtype-value 'a :float64)) :errored))

  ;; coerce-dtype-value :bool
  (assert-true (eq (mini-array.core:coerce-dtype-value t :bool) t))
  (assert-true (eq (mini-array.core:coerce-dtype-value nil :bool) nil))
  (assert-true (eq (expect-error (mini-array.core:coerce-dtype-value 1 :bool)) :errored))
  (assert-true (eq (expect-error (mini-array.core:coerce-dtype-value 'x :bool)) :errored))

  ;; infer-scalar-dtype
  (assert-equal (mini-array.core:infer-scalar-dtype t) :bool)
  (assert-equal (mini-array.core:infer-scalar-dtype nil) :bool)
  (assert-equal (mini-array.core:infer-scalar-dtype 0) :float64)
  (assert-equal (mini-array.core:infer-scalar-dtype 3.14) :float64)
  (assert-true (eq (expect-error (mini-array.core:infer-scalar-dtype #\A)) :errored))

  (format t "OK: CORE tests passed~%")
  #+sbcl (sb-ext:exit :code 0))
