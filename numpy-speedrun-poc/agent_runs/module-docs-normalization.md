# module-docs-normalization

Summary
- Normalized all module docs to match docs/modules/core.md: consistent sections, concrete Common Lisp code blocks, explicit error cases, and See also links.
- Cross-checked exports against src/*/package.lisp and top-level re-exports in src/package.lisp; documented only the public :mini-array surface unless explicitly module-scoped.

Updated pages
- docs/modules/tensor.md — ndarray, asarray, make-array-from-flat, shape-of, strides-of, to-list, flat-offset, aref-nd, %set-aref-nd
- docs/modules/broadcast.md — broadcast-shape, all-indices, project-broadcast-index
- docs/modules/ufunc.md — binary-ufunc, add, mul (notes on non-reexported eq/lt/gt)
- docs/modules/reductions.md — sum-axis (and alias note sum_axis)
- docs/modules/indexing.md — boolean-select (module-level export)

Template used
- docs/modules/core.md (section order, tone, code style)

Representative examples added (one per page)
- tensor: (to-list (make-array-from-flat '(2 2) '(1 2 3 4))) => ((1.0d0 2.0d0) (3.0d0 4.0d0))
- broadcast: (to-list (add (asarray '((1 2 3))) (asarray '((100) (200))))) => ((110.0d0 120.0d0 130.0d0) (210.0d0 220.0d0 230.0d0))
- ufunc: (to-list (add (asarray '((1 2 3))) (asarray '((10) (20))))) => ((11.0d0 12.0d0 13.0d0) (21.0d0 22.0d0 23.0d0))
- reductions: (let* ((c (add (asarray '((1 2) (3 4))) (asarray '((10 20) (30 40))))) (row-sums (sum-axis c 1))) (to-list row-sums)) => (33.0d0 77.0d0)
- indexing: (let* ((v (asarray '(1 2 3 4 5 6))) (mask '(t nil t nil t nil)) (sel (mini-array.indexing:boolean-select v mask))) (to-list (add sel 10))) => (11.0d0 13.0d0 15.0d0)

Most important files/docs inspected
- docs/modules/core.md (canonical template)
- src/*/package.lisp and src/package.lisp (export surfaces)

Policies reflected in docs
- Dtypes: :float64 numerics; :bool masks
- Ranks: 0D/1D/2D; row-major contiguous; positive strides
- Broadcasting: NumPy-style right-aligned; NIL as scalar
- Ufuncs: add/mul via binary-ufunc; boolean inputs disallowed; outputs :float64
- Reductions: sum-axis 2D only, axis ∈ {0,1}; no keepdims
- Indexing: 1D boolean-select only; mask rank=1, same length, :bool entries; all-false mask is error

Uncertainties / follow-ups
- ufunc eq/lt/gt exist at module level but are not re-exported; excluded from :mini-array docs intentionally.
- sum_axis alias deprecation timing TBD; currently documented as temporary.
- Error message text is non-contractual; behavior/policy is normative.
