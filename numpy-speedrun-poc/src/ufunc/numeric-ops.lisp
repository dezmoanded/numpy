;;;; UFUNC numeric ops — add/mul wrappers over binary-ufunc

(in-package :mini-array.ufunc)

(defun add (a b)
  "Elementwise addition with broadcasting. Result dtype is :float64."
  (binary-ufunc a b #'+ :out-dtype :float64 :name 'add :allow-bool-input-p nil))

(defun mul (a b)
  "Elementwise multiplication with broadcasting. Result dtype is :float64."
  (binary-ufunc a b #'* :out-dtype :float64 :name 'mul :allow-bool-input-p nil))
