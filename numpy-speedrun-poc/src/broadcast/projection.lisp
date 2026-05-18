;;;; mini-array.broadcast index projection

(in-package :mini-array.broadcast)

(defun project-broadcast-index (out-index in-shape out-shape)
  "Project OUT-INDEX (for OUT-SHAPE) back into an index for IN-SHAPE under
NumPy-style right-aligned broadcasting (ranks 0..2).

Validation:
- SHAPES must be valid (rank<=2, positive dims)
- OUT-INDEX length must equal rank(OUT-SHAPE)
- IN-SHAPE must be broadcastable to OUT-SHAPE

Semantics:
- Left-pad IN-SHAPE with 1s to OUT rank
- For each dim: if in-dim==1 -> 0, else copy from OUT-INDEX
- Drop the left padding in the returned index; 0D returns NIL."
  (unless (and (mini-array.core:valid-shape-p in-shape)
               (mini-array.core:valid-shape-p out-shape))
    (error "project-broadcast-index: invalid shapes ~S and ~S" in-shape out-shape))
  (let* ((r (mini-array.core:rank out-shape))
         (ri (length (or out-index nil))))
    (unless (= ri r)
      (error "project-broadcast-index: out-index ~S rank mismatch for out-shape ~S" out-index out-shape))
    ;; Ensure broadcast compatibility
    (let ((br (broadcast-shape in-shape out-shape)))
      (declare (ignore br))
      (unless (equal (broadcast-shape in-shape out-shape) out-shape)
        (error "project-broadcast-index: in-shape ~S not broadcastable to ~S" in-shape out-shape)))
    (let* ((pad (- r (mini-array.core:rank in-shape)))
           (ip (%pad-left-with-ones in-shape r))
           (proj '()))
      (dotimes (i r)
        (let* ((in-d (nth i ip))
               (oi (nth i out-index))
               (proj-elt (if (= in-d 1) 0 oi)))
          (push proj-elt proj)))
      (setf proj (nreverse proj))
      ;; drop left padding elements
      (nthcdr pad proj))))
