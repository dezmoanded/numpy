;;;; mini-array.broadcast shape

(in-package :mini-array.broadcast)

(defun %pad-left-with-ones (shape target-rank)
  "Return SHAPE padded on the left with 1s to reach TARGET-RANK."
  (let* ((len (mini-array.core:rank shape))
         (pad (max 0 (- target-rank len))))
    (nconc (make-list pad :initial-element 1)
           (copy-list (or shape nil)))))

(defun broadcast-shape (a-shape b-shape)
  "Compute NumPy-style right-aligned broadcast shape for ranks 0..2.

Rules per-dimension (after left-padding with 1s):
- if dims equal -> that dim
- else if one is 1 -> the other
- else -> error (incompatible)

NIL is rank-0. Errors if either shape invalid or rank>2."
  (unless (and (mini-array.core:valid-shape-p a-shape)
               (mini-array.core:valid-shape-p b-shape))
    (error "broadcast-shape: invalid shapes ~S and ~S (rank<=2, positive integers)"
           a-shape b-shape))
  (let* ((ra (mini-array.core:rank a-shape))
         (rb (mini-array.core:rank b-shape))
         (r (max ra rb))
         (ap (%pad-left-with-ones a-shape r))
         (bp (%pad-left-with-ones b-shape r))
         (out '()))
    (dotimes (i r)
      (let* ((da (nth i ap))
             (db (nth i bp))
             (d (cond ((= da db) da)
                      ((= da 1) db)
                      ((= db 1) da)
                      (t (error "broadcast-shape: incompatible shapes ~S and ~S" a-shape b-shape)))))
        (push d out)))
    (nreverse out)))
