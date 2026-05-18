;;;; UFUNC package definition

(defpackage :mini-array.ufunc
  (:nicknames :ma.ufunc)
  (:use :cl)
  (:export #:binary-ufunc
           #:add
           #:mul
           #:eq
           #:lt
           #:gt))

(in-package :mini-array.ufunc)
