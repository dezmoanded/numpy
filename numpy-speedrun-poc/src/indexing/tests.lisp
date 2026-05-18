;;;; numpy-speedrun-poc/src/indexing/tests.lisp
;; Run with: sbcl --script numpy-speedrun-poc/src/indexing/tests.lisp

(in-package :cl)

(defun %here-pathname ()
  (or *load-truename* *compile-file-truename* (truename ".")))

(defun %dir-of (pathname)
  (make-pathname :name nil :type nil :version nil :defaults pathname))

(let* ((here (%here-pathname))
       (root (merge-pathnames "../../" (%dir-of here)))
       (src  (merge-pathnames "src/" root)))
  ;; Load modular packages first (so subpackages exist)
  (load (merge-pathnames "core/package.lisp" src))
  (load (merge-pathnames "tensor/package.lisp" src))
  (load (merge-pathnames "broadcast/package.lisp" src))
  (load (merge-pathnames "ufunc/package.lisp" src))
  (load (merge-pathnames "reductions/package.lisp" src))
  ;; Define the top-level :mini-array package next, since indexing/package
  ;; imports-from :mini-array.
  (load (merge-pathnames "package.lisp" src))
  ;; Now define the indexing package (imports from :mini-array)
  (load (merge-pathnames "indexing/package.lisp" src))
  ;; Implementations in dependency order (core -> tensor -> indexing)
  (load (merge-pathnames "core/shape.lisp" src))
  (load (merge-pathnames "core/dtype.lisp" src))
  (load (merge-pathnames "tensor/ndarray.lisp" src))
  (load (merge-pathnames "tensor/indexing.lisp" src))
  (load (merge-pathnames "tensor/conversion.lisp" src))
  (load (merge-pathnames "indexing/boolean-select.lisp" src)))

(in-package :mini-array.indexing)

(defun assert-true (cond &optional (msg "Assertion failed"))
  (unless cond (error msg)))

(format t "[indexing/tests] Starting...~%")

;; Happy path: mixed true/false
(let* ((a (mini-array:asarray '(10 20 30 40 50)))
       (mask '(t nil t nil t))
       (b (boolean-select a mask)))
  (assert-true (equal '(3) (mini-array:shape-of b)) "shape mismatch")
  (assert-true (equal '(10.0d0 30.0d0 50.0d0) (mini-array:to-list b)) "values mismatch"))

;; Error: all-false mask
(handler-case
    (progn
      (let* ((a (mini-array:asarray '(1 2 3)))
             (mask '(nil nil nil)))
        (boolean-select a mask))
      (error "expected error for all-false mask"))
  (error (e)
    (declare (ignore e))
    (format t "[ok] all-false mask signaled error~%")))

;; Error: rank != 1 (use 2D)
(handler-case
    (progn
      (let* ((a (mini-array:asarray '((1 2) (3 4))))
             (mask '(t t)))
        (boolean-select a mask))
      (error "expected error for rank != 1"))
  (error (e)
    (declare (ignore e))
    (format t "[ok] rank!=1 signaled error~%")))

;; Error: mask length mismatch
(handler-case
    (progn
      (let* ((a (mini-array:asarray '(1 2 3 4)))
             (mask '(t t t)))
        (boolean-select a mask))
      (error "expected error for mask length mismatch"))
  (error (e)
    (declare (ignore e))
    (format t "[ok] mask length mismatch signaled error~%")))

;; Error: non-boolean entries in mask
(handler-case
    (progn
      (let* ((a (mini-array:asarray '(1 2 3 4)))
             (mask '(t 1 nil 0)))
        (boolean-select a mask))
      (error "expected error for non-boolean mask entries"))
  (error (e)
    (declare (ignore e))
    (format t "[ok] non-boolean mask entries signaled error~%")))

(format t "[indexing/tests] All assertions passed.~%")
