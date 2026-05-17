;; numpy-speedrun-poc/tests/core-tests.lisp
;; SBCL --script

(load "src/package.lisp")
(load "src/core.lisp")

(defun fail (fmt &rest args)
  (format t "FAIL: ")
  (apply #'format t fmt args)
  (terpri)
  #+sbcl (sb-ext:exit :code 1))

(defun assert-true (cond &optional (msg ""))
  (unless cond (fail "~A" msg)))

(defun assert-equal (got expect &optional (msg ""))
  (unless (equal got expect)
    (fail "~A Expected ~S, got ~S" msg expect got)))

;; product
(assert-equal (mini-array:product nil) 1 "product nil -> 1")
(assert-equal (mini-array:product '(2 3)) 6 "product 2x3 -> 6")

;; rank
(assert-equal (mini-array:rank nil) 0)
(assert-equal (mini-array:rank '(5)) 1)
(assert-equal (mini-array:rank '(2 3)) 2)

;; valid-shape-p
(assert-true (mini-array:valid-shape-p nil) "0D allowed")
(assert-true (mini-array:valid-shape-p '(2)))
(assert-true (mini-array:valid-shape-p '(2 3)))
(assert-true (not (mini-array:valid-shape-p '(0))))
(assert-true (not (mini-array:valid-shape-p '(2 0))))
(assert-true (not (mini-array:valid-shape-p '(1 2 3))))

;; default-strides
(assert-equal (mini-array:default-strides nil) nil)
(assert-equal (mini-array:default-strides '(5)) '(1))
(assert-equal (mini-array:default-strides '(2 3)) '(3 1))
(let ((errored nil))
  (handler-case (progn (mini-array:default-strides '(1 2 3)) (setf errored t))
    (error () nil))
  (assert-true (not errored) "rank>2 should error"))

(format t "OK: core tests passed~%")
#+sbcl (sb-ext:exit :code 0)
