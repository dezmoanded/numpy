;;;; numpy-speedrun-poc/src/package.lisp

(defpackage :mini-array
  (:use :cl)
  (:export
   ;; data type
   :ndarray
   ;; constructors / core API
   :make-array-from-flat
   :asarray
   ;; core utils
   :product
   :rank
   :valid-shape-p
   ;; introspection / materialization
   :shape-of
   :strides-of
   :to-list
   ;; indexing helpers
   :default-strides
   :flat-offset
   :aref-nd
   ;; broadcasting and iteration
   :broadcast-shape
   :all-indices
   :project-broadcast-index
   ;; ufuncs
   :binary-ufunc
   :add
   :mul
   ;; reductions
   :sum-axis
   :sum_axis))

(in-package :mini-array)
