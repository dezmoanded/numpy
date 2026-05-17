;;;; numpy-speedrun-poc/src/reductions.lisp
;;;; Reductions: sum-axis for 2D arrays (V1)

(in-package :mini-array)

;;; Self-load minimal dependencies (dir-relative): core.lisp, tensor.lisp
(eval-when (:compile-toplevel :load-toplevel :execute)
  (let* ((here (or *load-truename* *compile-file-truename*))
         (dir  (and here (make-pathname :name nil :type nil :version nil :defaults here))))
    (when dir
      (load (merge-pathnames "core.lisp" dir))
      (load (merge-pathnames "tensor.lisp" dir)))))

(defun sum-axis (a axis)
  "Sum across AXIS for 2D ndarray A (V1).
Rules:
- Only 2D reductions supported; 0D/1D raise errors
- AXIS must be 0 or 1
- Returns a new contiguous float64 ndarray"
  (let* ((A a)
         (shape (ndarray-shape A)))
    (unless (= (length shape) 2)
      (error "sum-axis: only 2D inputs supported in V1, got rank ~A" (length shape)))
    (unless (member axis '(0 1))
      (error "sum-axis: invalid axis ~A (must be 0 or 1)" axis))
    (let* ((r (first shape))
           (c (second shape)))
      (cond
        ((= axis 0)
         (let ((out (make-array-from-flat (list c)
                                          (make-list c :initial-element 0.0d0))))
           (dotimes (j c)
             (let ((acc 0.0d0))
               (dotimes (i r)
                 (incf acc (aref-nd A (list i j))))
               (%set-aref-nd out (list j) acc)))
           out))
        (t
         (let ((out (make-array-from-flat (list r)
                                          (make-list r :initial-element 0.0d0))))
           (dotimes (i r)
             (let ((acc 0.0d0))
               (dotimes (j c)
                 (incf acc (aref-nd A (list i j))))
               (%set-aref-nd out (list i) acc)))
           out))))))

;; Alias kept locally so reductions tests can rely only on this file
(defun sum_axis (a axis)
  (sum-axis a axis))
