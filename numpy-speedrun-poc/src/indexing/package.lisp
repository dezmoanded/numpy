;;;; numpy-speedrun-poc/src/indexing/package.lisp

(defpackage :mini-array.indexing
  (:use :cl)
  (:import-from :mini-array
    :ndarray
    :make-array-from-flat
    :shape-of
    :strides-of
    :flat-offset
    :aref-nd
    :rank)
  (:export
    :boolean-select))

(in-package :mini-array.indexing)
