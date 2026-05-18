;;;; mini-array core package

(defpackage :mini-array.core
  (:use :cl)
  (:nicknames :ma.core)
  (:export
   ;; shape
   :product
   :rank
   :valid-shape-p
   :default-strides
   ;; dtype
   :dtype-p
   :coerce-dtype-value
   :infer-scalar-dtype))

(in-package :mini-array.core)
