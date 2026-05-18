;;;; mini-array.broadcast package

(defpackage :mini-array.broadcast
  (:use :cl)
  (:nicknames :ma.broadcast)
  (:import-from :mini-array.core :rank :valid-shape-p)
  (:export
   :broadcast-shape
   :all-indices
   :project-broadcast-index))

(in-package :mini-array.broadcast)
