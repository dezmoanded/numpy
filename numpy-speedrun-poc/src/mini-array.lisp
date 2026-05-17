;;;; numpy-speedrun-poc/src/mini-array.lisp
;;;; Integration loader only (V1). Do not use in subgroup tests.

(in-package :mini-array)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (let* ((here (or *load-truename* *compile-file-truename*))
         (dir  (and here (make-pathname :name nil :type nil :version nil :defaults here))))
    (when dir
      (load (merge-pathnames "core.lisp" dir))
      (load (merge-pathnames "tensor.lisp" dir))
      (load (merge-pathnames "ops.lisp" dir))
      (load (merge-pathnames "reductions.lisp" dir)))))
