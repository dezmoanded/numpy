;; Reductions module-scoped tests (standalone).
;; - Does NOT load src/mini-array.lisp.
;; - Loads only modular packages/impl for core, tensor, reductions in dependency order.
;; - Uses module packages directly (no top-level :mini-array).

(in-package :cl)

;; File-relative loader per instructions
(defun %here () (or *load-truename* *compile-file-truename* (truename ".")))
(defun %dir (p) (make-pathname :name nil :type nil :version nil :defaults p))
(let* ((here (%here))
       (root (merge-pathnames "../" (%dir here)))
       (src  (merge-pathnames "src/" root)))
  (labels ((load-rel (rel)
             (let ((p (merge-pathnames rel src)))
               (when (probe-file p) (load p)))))
    ;; core
    (load-rel "core/package.lisp")
    (load-rel "core/shape.lisp")
    (load-rel "core/dtype.lisp")
    ;; tensor
    (load-rel "tensor/package.lisp")
    (load-rel "tensor/ndarray.lisp")
    (load-rel "tensor/indexing.lisp")
    (load-rel "tensor/conversion.lisp")
    ;; reductions
    (load-rel "reductions/package.lisp")
    (load-rel "reductions/sum.lisp")))

;; Simple test helpers
(defun fail (fmt &rest args)
  (format t "FAIL: ")
  (apply #'format t fmt args)
  (terpri)
  (sb-ext:exit :code 1))

(defun assert-equal (got expect &optional (msg ""))
  (unless (equal got expect)
    (fail "~A Expected ~S, got ~S" msg expect got)))

(defun expect-error (thunk &optional (msg ""))
  (handler-case
      (progn (funcall thunk)
             (fail "~A Expected an error, but none was signaled" msg))
    (error () t)))

(defun list-approx= (a b &optional (eps 1d-9))
  (and (= (length a) (length b))
       (every (lambda (x y) (< (abs (- (coerce x 'double-float)
                                       (coerce y 'double-float)))
                                eps))
              a b)))

(defun assert-list-approx (got expect &optional (msg ""))
  (unless (list-approx= got expect)
    (fail "~A Expected ~S, got ~S" msg expect got)))

;; Happy paths: 2x3 matrix, axis 0 and 1
(let* ((A (mini-array.tensor:asarray '((1 2 3) (4 5 6)))))
  (let* ((s0 (mini-array.reductions:sum-axis A 0))
         (l0 (mini-array.tensor:to-list s0)))
    (assert-equal (length l0) 3 "axis=0 length: ")
    (assert-list-approx l0 '(5 7 9) "axis=0 sums: "))
  (let* ((s1 (mini-array.reductions:sum-axis A 1))
         (l1 (mini-array.tensor:to-list s1)))
    (assert-equal (length l1) 2 "axis=1 length: ")
    (assert-list-approx l1 '(6 15) "axis=1 sums: ")))

;; Error paths: 0D, 1D, invalid axes
(expect-error (lambda () (mini-array.reductions:sum-axis (mini-array.tensor:asarray 7) 0))
              "0D should error: ")
(expect-error (lambda () (mini-array.reductions:sum-axis (mini-array.tensor:asarray '(1 2 3)) 0))
              "1D should error: ")

(let ((A (mini-array.tensor:asarray '((1 2) (3 4)))))
  (expect-error (lambda () (mini-array.reductions:sum-axis A -1)) "axis=-1 should error: ")
  (expect-error (lambda () (mini-array.reductions:sum-axis A 2))  "axis=2 should error: "))

;; Alias check: sum_axis
(let* ((A (mini-array.tensor:asarray '((1 2 3) (4 5 6))))
       (s0 (mini-array.reductions:sum_axis A 0))
       (l0 (mini-array.tensor:to-list s0)))
  (assert-equal (length l0) 3 "alias sum_axis axis=0 length: ")
  (assert-list-approx l0 '(5 7 9) "alias sum_axis axis=0: "))

(format t "OK: reductions module tests passed~%")
(sb-ext:exit :code 0)
