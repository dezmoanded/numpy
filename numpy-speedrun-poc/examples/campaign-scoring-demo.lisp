;;;; numpy-speedrun-poc/examples/campaign-scoring-demo.lisp
;; Run with: sbcl --noinform --non-interactive --script examples/campaign-scoring-demo.lisp

(in-package :cl)

;; Resolve src/ relative to this file so it works from any CWD
(defun %here () (or *load-truename* *compile-file-truename* (truename ".")))
(defun %dir (p) (make-pathname :name nil :type nil :version nil :defaults p))
(let* ((here (%here))
       (root (merge-pathnames "../" (%dir here)))
       (src  (merge-pathnames "src/" root)))
  ;; Load modular integration loader first; it brings in subpackages/impls.
  (load (merge-pathnames "mini-array.lisp" src))
  ;; Ensure top-level re-exports are visible (idempotent if already loaded)
  (load (merge-pathnames "package.lisp" src)))

(in-package :mini-array)

(defun pplist (label thing)
  (format t "~A: ~S~%" label thing))

(defun main ()
  ;; Scenario: campaign scoring over a (users x features) matrix.
  ;; Pipeline: (X .* weights_row) .+ bias_col, then reductions and selection.
  (let* ((usersxfeat (asarray '((1  2  3  4)
                                (2  1  0  1)
                                (5 10  2  1)))) ; 3 users x 4 features
         ;; Column weights (1x4): emphasize middle columns
         (weights     (asarray '((0.5d0 1.5d0 2.0d0 0.1d0))))
         ;; Row bias (3x1): per-user baseline
         (bias        (asarray '((1.0d0)
                                 (0.5d0)
                                 (2.0d0))))
         ;; Broadcasted elementwise ops
         (weighted    (mul usersxfeat weights))  ; 3x4 .* 1x4 => 3x4
         (scores      (add weighted bias))       ; 3x4 .+ 3x1 => 3x4
         ;; Reductions
         (per-user    (sum-axis scores 1))       ; shape (3)
         (per-feature (sum-axis scores 0))       ; shape (4)
         ;; Host-side thresholding to produce a 1D boolean mask (length=3)
         (threshold   12.0d0)
         (user-mask   (mapcar (lambda (x) (> x threshold)) (to-list per-user)))
         ;; Select users whose score exceeds threshold (1D boolean-select)
         (user-ids    (asarray '(101 102 103)))
         (selected    (mini-array.indexing:boolean-select user-ids user-mask))
         ;; Extra: scalar + vector demo
         (daily-bonus  (asarray 1.0d0))
         (boosted-per-user (add per-user daily-bonus))
         ;; Pure 1D boolean-select mini demo
         (v1d (asarray '(10 20 30 40)))
         (m1d '(t nil t nil))
         (sel1d (mini-array.indexing:boolean-select v1d m1d)))

    ;; Prints (labeled)
    (pplist "Input usersxfeat" (to-list usersxfeat))
    (pplist "Weights (row)" (to-list weights))
    (pplist "Bias (col)" (to-list bias))
    (pplist "Weighted = usersxfeat * weights" (to-list weighted))
    (pplist "Scores = weighted + bias" (to-list scores))
    (pplist "Per-user scores (axis=1)" (to-list per-user))
    (pplist "Per-feature totals (axis=0)" (to-list per-feature))
    (pplist "User mask (> threshold)" user-mask)
    (pplist "Selected user-ids" (to-list selected))
    (pplist "Boosted per-user (+1.0)" (to-list boosted-per-user))
    (pplist "1D input" (to-list v1d))
    (pplist "1D mask" m1d)
    (pplist "1D selected" (to-list sel1d))

    ;; End with success line for CI/logs
    (format t "Campaign scoring demo OK.~%")))

(main)
