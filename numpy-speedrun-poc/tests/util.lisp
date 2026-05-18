;;; numpy-speedrun-poc/tests/util.lisp
;;; Common test utilities: path helpers, loaders, and assertions.

(in-package :cl)

(defpackage :mini-array.test-util
  (:use :cl)
  (:export
   ;; path helpers
   :%here :%dir :%tests-root :%src-root
   ;; loaders
   :load-integration :load-modules
   ;; assertions
   :assert-true :approx= :assert-approx= :expect-error))

(in-package :mini-array.test-util)

(defun %here ()
  (or *load-truename* *compile-file-truename* (truename ".")))

(defun %dir (p)
  (make-pathname :name nil :type nil :version nil :defaults p))

(defun %tests-root ()
  (merge-pathnames "../" (%dir (%here))))

(defun %src-root ()
  (merge-pathnames "src/" (%tests-root)))

(defun load-integration ()
  "Load the integration loader and top-level package. CWD-agnostic."
  (let ((src (%src-root)))
    (load (merge-pathnames "mini-array.lisp" src))
    (load (merge-pathnames "package.lisp" src))))

(defun load-modules ()
  "Load modular packages first, then implementations in dependency order."
  (let ((src (%src-root)))
    ;; Packages first
    (load (merge-pathnames "core/package.lisp" src))
    (load (merge-pathnames "tensor/package.lisp" src))
    (load (merge-pathnames "broadcast/package.lisp" src))
    (load (merge-pathnames "ufunc/package.lisp" src))
    (load (merge-pathnames "reductions/package.lisp" src))
    (load (merge-pathnames "indexing/package.lisp" src))
    ;; Implementations
    (load (merge-pathnames "core/shape.lisp" src))
    (load (merge-pathnames "core/dtype.lisp" src))
    (load (merge-pathnames "tensor/ndarray.lisp" src))
    (load (merge-pathnames "tensor/indexing.lisp" src))
    (load (merge-pathnames "tensor/conversion.lisp" src))
    (load (merge-pathnames "broadcast/shape.lisp" src))
    (load (merge-pathnames "broadcast/projection.lisp" src))
    (load (merge-pathnames "broadcast/iterator.lisp" src))
    (load (merge-pathnames "ufunc/engine.lisp" src))
    (load (merge-pathnames "ufunc/numeric-ops.lisp" src))
    (load (merge-pathnames "reductions/sum.lisp" src))
    (load (merge-pathnames "indexing/boolean-select.lisp" src))))

;; Assertions and helpers
(defun assert-true (cond &optional (msg "assert-true failed"))
  (unless cond (error msg)))

(defun approx= (a b &optional (eps 1.0d-6))
  (cond
    ((and (numberp a) (numberp b))
     (<= (abs (- (coerce a 'double-float)
                 (coerce b 'double-float))) eps))
    ((and (null a) (null b)) t)
    ((and (listp a) (listp b))
     (and (= (length a) (length b))
          (every #'identity (mapcar (lambda (x y) (approx= x y eps)) a b))))
    (t nil)))

(defun assert-approx= (x y &optional (eps 1.0d-6))
  (unless (approx= x y eps)
    (error "Approx equal failed: ~S vs ~S" x y)))

(defun expect-error (thunk)
  (handler-case (progn (funcall thunk)
                       (error "Expected an error, but none was signaled"))
    (error (c) (declare (ignore c)) t)))