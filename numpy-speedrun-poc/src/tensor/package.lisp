;;;; numpy-speedrun-poc/src/tensor/package.lisp

(defpackage :mini-array.tensor
  (:use :cl)
  (:nicknames :ma.tensor)
  (:import-from :mini-array.core
    :product :rank :valid-shape-p :default-strides)
  (:export
    ;; data model / accessors
    :ndarray
    :make-array-from-flat
    :shape-of
    :strides-of
    :dtype-of
    ;; indexing
    :flat-offset
    :aref-nd
    ;; conversions
    :asarray
    :to-list))

(in-package :mini-array.tensor)
