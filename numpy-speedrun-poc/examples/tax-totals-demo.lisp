;;;; numpy-speedrun-poc/examples/tax-totals-demo.lisp
;; Run with: sbcl --script examples/tax-totals-demo.lisp

(load "src/package.lisp")
(load "src/mini-array.lisp")

(in-package :mini-array)

(let* ((prices (make-array-from-flat '(2 3) '(10 20 30 40 50 60)))
       (tax    (asarray '(0.07d0 0.08d0 0.09d0)))
       (ones   (asarray '(1.0d0 1.0d0 1.0d0)))
       (totals (mul prices (add ones tax)))
       (row-totals (sum-axis totals 1)))
  (format t "Totals: ~S~%" (to-list totals))
  (format t "Row totals: ~S~%" (to-list row-totals))
  (format t "Expected totals: ((10.7d0 21.6d0 32.7d0) (42.8d0 54.0d0 65.4d0))~%")
  (format t "Expected row_totals: (65.0d0 162.2d0)~%"))
