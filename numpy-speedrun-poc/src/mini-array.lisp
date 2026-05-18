;;;; numpy-speedrun-poc/src/mini-array.lisp
;;;; Integration loader (V2). Safe, idempotent, and dependency-ordered.
;;;; Note: Submodule tests must NOT load this file; only integration tests/demos should.

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %ma--here-dir ()
    (let ((here (or *load-truename* *compile-file-truename*)))
      (and here (make-pathname :name nil :type nil :version nil :defaults here))))
  (defun %ma--load-rel (rel)
    (let* ((dir (%ma--here-dir))
           (path (and dir (merge-pathnames rel dir))))
      (when (and path (probe-file path))
        (load path))))
  (defun %ma--load-list (rels)
    (dolist (r rels)
      (%ma--load-rel r)))

  ;; 1) Load modularized DAG in order (packages first, then impls)
  ;;    IMPORTANT: Some submodules (e.g., indexing) import-from :mini-array,
  ;;    so we defer loading them until after the top-level package is present.
  (%ma--load-list '(
    ;; core
    "core/package.lisp" "core/shape.lisp" "core/dtype.lisp"
    ;; tensor
    "tensor/package.lisp" "tensor/ndarray.lisp" "tensor/indexing.lisp" "tensor/conversion.lisp"
    ;; broadcast
    "broadcast/package.lisp" "broadcast/shape.lisp" "broadcast/projection.lisp" "broadcast/iterator.lisp"
    ;; ufunc (files are optional; load if present)
    "ufunc/package.lisp" "ufunc/engine.lisp" "ufunc/numeric-ops.lisp" "ufunc/comparison-ops.lisp"
    ;; reductions
    "reductions/package.lisp" "reductions/sum.lisp"))

  ;; 2) Fallback: if modular tensor package is absent, load legacy flat files
  (unless (find-package :mini-array.tensor)
    (dolist (legacy '("core.lisp" "tensor.lisp" "ops.lisp" "reductions.lisp"))
      (%ma--load-rel legacy)))

  ;; 3) Always load top-level package to expose :mini-array (legacy or re-export)
  (%ma--load-rel "package.lisp")

  ;; 4) Load modules that import from :mini-array (e.g., indexing) after top-level exists
  (%ma--load-list '(
    "indexing/package.lisp" "indexing/boolean-select.lisp"))
)