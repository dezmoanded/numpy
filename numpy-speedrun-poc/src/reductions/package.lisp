;;;; numpy-speedrun-poc/src/reductions/package.lisp

(defpackage :mini-array.reductions
  (:use :cl)
  (:import-from :mini-array.core
    :rank :product :valid-shape-p :default-strides :dtype-p :coerce-dtype-value)
  (:import-from :mini-array.tensor
    :ndarray :make-array-from-flat :asarray :shape-of :strides-of :dtype-of
    :flat-offset :aref-nd :to-list)
  (:export :sum-axis :sum_axis))

(in-package :mini-array.reductions)
