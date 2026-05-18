# numpy-speedrun-poc

## Tiny clean-room Lisp numerical core extracted from NumPy architecture

> Agent-assisted architectural extraction and implementation of a minimal NumPy-like array runtime in a constrained Lisp subset.

---

# Why this exists

Most AI coding demos show agents generating isolated scripts.

This project explores a harder problem:

> Can agents study a mature numerical ecosystem like NumPy, extract a bounded architectural slice, and rapidly stand up a clean-room implementation in a regular Lisp-like language suitable for future validation and compliance?

This repository is a proof-of-concept for that workflow.

The goal is **not** to clone NumPy.

The goal is to demonstrate:

* architectural decomposition
* agent-readable modularity
* clean-room behavioral extraction
* deterministic interfaces
* Lisp-like numerical IR structure
* rapid ecosystem replication
* future suitability for validation and compliance-oriented systems

The implementation intentionally avoids:

* opaque magic
* hidden runtime behavior
* macro-heavy abstractions
* dynamic dependency tricks
* broad implicit state

Instead, the system is built from explicit layers:

```text
ndarray metadata
  → shape/stride semantics
  → broadcast planning
  → broadcasted iteration
  → ufunc-style execution
  → reductions
  → boolean masking/indexing
```

---

# Why Lisp?

This project intentionally uses a constrained Lisp-style architecture because:

* regular syntax is easier for agents to reason about
* explicit structure is easier to validate
* semantics can be made highly deterministic
* interfaces are naturally inspectable
* the architecture maps well onto future typed/validated IR systems

This is **not** intended to demonstrate Common Lisp wizardry.

In fact, many advanced/dynamic CL features are intentionally avoided.

The implementation is treated as a restricted, modular, validation-oriented Lisp subset.

---

# What was extracted from NumPy

Agents studied NumPy documentation, tests, and architectural patterns and extracted a deliberately small but meaningful slice:

## ndarray metadata

```text
shape
strides
dtype
flat storage
```

## Shape and indexing semantics

* row-major contiguous layout
* element-based strides
* multidimensional → flat offset mapping

## Broadcasting

* right-aligned shape comparison
* singleton expansion semantics
* broadcasted index projection

## Ufunc-style execution

* iteration separated from scalar operations
* elementwise add/mul
* comparison operations

## Reductions

* axis-oriented reduction
* output shape transformation

## Boolean masking

* comparison-generated boolean arrays
* mask-based selection

The implementation re-expresses these behaviors in original code.

No NumPy implementation code is copied.

---

# Current scope (V1 + V2)

## Supported dtypes

### `:float64`

Used for all numeric computation.

### `:bool`

Used for boolean masks.

---

## Supported ranks

* scalar / 0D
* 1D
* 2D

---

## Supported semantics

### Array metadata

* shape
* strides
* dtype
* contiguous flat storage

### Broadcasting

NumPy-style right-aligned broadcasting.

### Elementwise ufuncs

* add
* mul
* gt
* lt
* eq

### Reductions

* sum-axis over axis 0 or 1 for 2D arrays

### Boolean masking

* 1D boolean selection

---

## Explicit exclusions

The following are intentionally excluded to keep the architectural slice small and reviewable:

* dtype promotion/casting
* non-contiguous views
* slicing
* advanced indexing
* masked arrays
* object arrays
* BLAS/SIMD
* matrix multiplication
* > 2D arrays
* negative/zero strides
* keepdims
* axis tuples
* NaN-special semantics
* full NumPy compatibility

---

# Architecture

The system is intentionally decomposed into small agent-readable subsystems.

```text
src/
  core/
  tensor/
  broadcast/
  ufunc/
  indexing/
  reductions/
```

The top-level integration loader is:

```text
src/mini-array.lisp
```

Each subsystem owns a narrow responsibility and contract.

---

# Module overview

## core/

Lowest-level shape and dtype helpers.

Responsibilities:

* rank
* product
* shape validation
* default strides
* dtype normalization

Key idea:

```text
All higher systems depend on shape semantics.
```

---

## tensor/

Defines ndarray representation and indexing.

Responsibilities:

* ndarray struct
* array construction
* flat indexing
* scalar/list conversion
* shape/stride access

Core ndarray representation:

```lisp
(ndarray
  :shape '(2 3)
  :strides '(3 1)
  :dtype :float64
  :data #(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0))
```

---

## broadcast/

Responsible only for shape and index transformation.

Responsibilities:

* broadcast-shape
* output index iteration
* broadcasted index projection

Important:

```text
broadcast/ does NOT know about ufuncs.
```

This separation was intentional.

---

## ufunc/

Tiny ufunc execution layer.

Responsibilities:

* binary-ufunc engine
* add
* mul
* comparison operations

Architecture:

```text
broadcast planning
  → iterate output indices
  → project operand indices
  → apply scalar operation
  → write output array
```

This mirrors NumPy’s architectural pattern at a much smaller scale.

---

## indexing/

Currently contains:

* boolean-select

This is the first orthogonal extension slice beyond arithmetic.

Why it matters:

```text
matrix arithmetic proves broadcasting
boolean masking proves generalized array semantics
```

---

## reductions/

Axis-oriented reductions.

Currently:

* sum-axis

---

# Public API

## Array and metadata

* ndarray
* asarray
* make-array-from-flat
* shape-of
* strides-of
* to-list

## Core helpers

* product
* rank
* valid-shape-p
* default-strides
* flat-offset
* aref-nd

## Broadcasting

* broadcast-shape
* all-indices
* project-broadcast-index

## Ufuncs

* binary-ufunc
* add
* mul
* gt
* lt
* eq

## Reductions

* sum-axis

## Indexing

* boolean-select

---

# Agent-oriented development process

This repository was intentionally structured for parallel agent work.

The workflow was:

```text
research
  → architectural extraction
  → subsystem decomposition
  → implementation contracts
  → modular implementation
  → narrow module tests
  → integration tests
```

Research artifacts live under:

```text
agent_runs/
```

These contain:

* extracted NumPy invariants
* behavioral rules
* implementation handoffs
* subsystem contracts
* clean-room pseudocode

---

# Testing philosophy

The project intentionally avoids tangled integration-first testing.

Each subsystem has:

* narrow responsibilities
* narrow tests
* minimal dependencies

Module tests:

```text
src/core/tests.lisp
src/tensor/tests.lisp
src/broadcast/tests.lisp
src/ufunc/tests.lisp
src/indexing/tests.lisp
src/reductions/tests.lisp
```

Integration tests:

```text
tests/integration-tests.lisp
```

The goal is to make the system:

```text
reviewable
agent-expandable
validation-friendly
```

rather than merely concise.

---

# Example: tax totals demo

```lisp
(let* ((prices (make-array-from-flat '(2 3)
                                     '(10 20 30
                                       40 50 60)))
       (tax    (asarray '(0.07d0 0.08d0 0.09d0)))
       (ones   (asarray '(1.0d0 1.0d0 1.0d0)))
       (totals (mul prices (add ones tax)))
       (row-totals (sum-axis totals 1)))
  ...)
```

This demonstrates:

```text
broadcasting
→ ufunc execution
→ axis reduction
```

Expected output:

```text
Totals:
((10.7d0 21.6d0 32.7d0)
 (42.8d0 54.0d0 65.4d0))

Row totals:
(65.0d0 162.2d0)
```

---

# Example: campaign scoring + masking

```text
users × weights
  → broadcasted multiplication
  → broadcasted bias addition
  → reductions
  → comparison mask
  → boolean selection
```

This second slice forced the architecture to expand beyond arithmetic into:

* boolean dtype handling
* comparison ufuncs
* generalized indexing semantics

---

# Running the project

## Run all tests

```bash
cd numpy-speedrun-poc
./run-all-tests.sh
```

## Run integration tests

```bash
sbcl --script tests/integration-tests.lisp
```

## Run demos

```bash
sbcl --script examples/tax-totals-demo.lisp
sbcl --script examples/campaign-scoring-demo.lisp
```

---

# Why this architecture matters

The important part of this repository is not the specific operations.

The important part is that:

```text
agents extracted a reusable architectural spine
from a mature numerical ecosystem
and reconstructed it in a modular,
reviewable,
validation-oriented Lisp system.
```

The design intentionally favors:

* explicit semantics
* deterministic structure
* subsystem isolation
* inspectability
* narrow contracts
* modular expansion

over raw feature count.

---

# Near-term roadmap

The next architectural steps are likely:

## Generalize rank

Move beyond 2D.

---

## Replace all-indices with reusable iterators

The current implementation is intentionally simple.

Future iterator abstractions should:

* avoid index list allocation
* operate directly on strides
* generalize across ufuncs/reductions

---

## Expand ufunc system

Add:

* unary ufuncs
* more comparison ops
* logical ops
* reductions implemented via generalized iteration

---

## Introduce richer dtype semantics

Eventually:

* dtype promotion
* casting rules
* shape/type validation
* static contracts

---

## Add views/slicing later

Only after stride semantics fully stabilize.

---

# Clean-room principle

This project intentionally separates:

```text
behavior
from
implementation
```

We study:

* NumPy semantics
* NumPy architectural patterns
* NumPy invariants

Then re-express them in original code.

No NumPy implementation code is copied.

---

# Long-term direction

This repository is ultimately exploring a broader question:

> Can agents rapidly reconstruct the architectural foundations of mature software ecosystems inside highly regular, validation-oriented languages?

This POC argues that the answer may be yes — if the system is decomposed into:

* explicit contracts
* narrow semantics
* modular subsystems
* deterministic interfaces
* agent-readable structure

rather than monolithic opaque runtimes.
