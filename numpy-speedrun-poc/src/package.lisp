;;;; numpy-speedrun-poc/src/package.lisp
;;;; Top-level integration package that re-exports the stable API
;;;; from modular subpackages. Do not put implementations here.

(defpackage :mini-array
  (:use :cl)
  ;; Import symbols from subpackages so :mini-array can re-export them.
  (:import-from :mini-array.core
    :product :rank :valid-shape-p :default-strides)
  (:import-from :mini-array.tensor
    :ndarray :make-array-from-flat :asarray
    :shape-of :strides-of :dtype-of
    :flat-offset :aref-nd :to-list)
  (:import-from :mini-array.broadcast
    :broadcast-shape :all-indices :project-broadcast-index)
  (:import-from :mini-array.ufunc
    :binary-ufunc :add :mul)
  (:import-from :mini-array.reductions
    :sum-axis :sum_axis)
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
