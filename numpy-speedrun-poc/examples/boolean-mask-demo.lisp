;;;; numpy-speedrun-poc/examples/boolean-mask-demo.lisp
;;;; Run with: sbcl --script numpy-speedrun-poc/examples/boolean-mask-demo.lisp

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %here-dir ()
    (let ((here (or *load-truename* *compile-file-truename*)))
      (and here (make-pathname :name nil :type nil :version nil :defaults here))))
  (defun %load-rel (rel)
    (let* ((dir (%here-dir))
           (path (and dir (merge-pathnames rel (make-pathname :directory (pathname-directory dir))))))
      (load path)))
  (%load-rel "../src/mini-array.lisp"))

(defun pplist (label thing)
  (format t "~A: ~S~%" label thing))

(defun main ()
  (let* ((v (mini-array:asarray '(10 20 30 40)))
         ;; A simple list mask is acceptable for demo purposes.
         (mask '(t nil t nil))
         (selected (mini-array.indexing:boolean-select v mask)))
    (pplist "input" (mini-array:to-list v))
    (pplist "mask" mask)
    (pplist "selected" (mini-array:to-list selected)))

  ;; Demonstrate error on all-false
  (handler-case
      (progn
        (mini-array.indexing:boolean-select (mini-array:asarray '(1 2 3)) '(nil nil nil))
        (format t "ERROR: expected failure for all-false mask but succeeded.~%"))
    (error (e) (format t "All-false mask correctly errored: ~A~%" e)))

  (format t "Done.~%"))

(main)
