;;;; numpy-speedrun-poc/src/reductions/sum.lisp

(in-package :mini-array.reductions)

(defun %ensure-double (x)
  "Return x as a double-float. For booleans, T->1.0d0, NIL->0.0d0. Signal on others."
  (cond
    ((typep x 'double-float) x)
    ((numberp x) (coerce x 'double-float))
    ((eq x t) 1.0d0)
    ((null x) 0.0d0)
    (t (error "sum-axis: unsupported element ~S for numeric reduction" x))))

(defun sum-axis (a axis)
  "Reduce 2D input along axis (0 or 1). Output dtype is :float64. Errors otherwise."
  (let* ((arr (asarray a))
         (shape (shape-of arr)))
    (unless (= (rank shape) 2)
      (error "sum-axis: expected 2D input, got shape ~S" shape))
    (unless (or (= axis 0) (= axis 1))
      (error "sum-axis: axis must be 0 or 1, got ~S" axis))
    (let* ((r (first shape))
           (c (second shape))
           (out-shape (if (= axis 0) (list c) (list r)))
           (n (product out-shape))
           (vec (make-array n :initial-element 0.0d0)))
      (cond
        ((= axis 0)
         ;; Sum down rows into columns → length c
         (dotimes (j c)
           (let ((s 0.0d0))
             (dotimes (i r)
               (incf s (%ensure-double (aref-nd arr (list i j)))))
             (setf (svref vec j) s))))
        (t
         ;; Sum across columns into rows → length r
         (dotimes (i r)
           (let ((s 0.0d0))
             (dotimes (j c)
               (incf s (%ensure-double (aref-nd arr (list i j)))))
             (setf (svref vec i) s)))))
      (make-array-from-flat out-shape :float64 vec))))

(defun sum_axis (a axis)
  "Temporary alias; delegates to sum-axis."
  (sum-axis a axis))
