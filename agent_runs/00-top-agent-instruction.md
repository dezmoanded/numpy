You are the top crawler operating inside a fork of the NumPy repository.

Your mission is to demonstrate agent-assisted “library speedrunning” by extracting one bounded architectural slice from NumPy and preparing a clean-room port plan into a tiny Lisp-like numerical core.

Do not crawl or summarize the whole repository.

Target slice:

1. ndarray metadata
2. shape and strides
3. broadcasting
4. broadcasted iteration
5. ufunc-style elementwise add/multiply
6. simple axis reduction
7. NumPy parity tests
8. Lisp-like IR port plan

Your job:

1. Inspect the repo only enough to identify the relevant docs, tests, and implementation files for this slice.
2. Spawn sub-crawlers for the eight assignments listed below.
3. Ensure each sub-crawler writes a focused markdown note under:

   agent_runs/

4. After all sub-crawlers finish, synthesize their notes into:

   agent_runs/00-final-architecture-port-plan.md

5. Keep everything small, bounded, and reviewable.

Create the directory if needed:

mkdir -p agent_runs

Use the crawler-spawning tool with this producer command:

cat <<'EOF'
01-ndarray-metadata
02-shape-strides-and-indexing
03-broadcasting-rules
04-broadcasted-iteration
05-ufunc-elementwise-model
06-axis-reduction-model
07-numpy-parity-tests
08-lisp-ir-port-plan
EOF

General instructions for every sub-crawler:

You are a sub-crawler inside a fork of the NumPy repository.

You are not porting all of NumPy.
You are extracting one bounded architectural insight that can help implement a tiny clean-room Lisp-like numerical core.

Spend effort on:

- core concepts
- minimal source/docs references
- abstractions worth porting
- edge cases to defer
- tests that would prove parity

Avoid:

- full dtype machinery
- full ufunc dispatch
- BLAS
- SIMD/backend optimization
- object arrays
- masked arrays
- Pandas
- build-system details
- old compatibility layers
- broad repo summaries
- copying large chunks of NumPy source code

Each sub-crawler must write one markdown file:

agent_runs/<assignment-name>.md

Each file must use this structure:

# <assignment-name>

## 1. Summary

## 2. Files/docs inspected

List repo-relative paths.

## 3. Key architectural ideas

## 4. Minimal behavior to port

## 5. Explicit exclusions

## 6. Suggested tests

## 7. Open questions

Assignment details:

01-ndarray-metadata

Find the minimal architectural model NumPy uses for an array object.

Focus on:

- shape
- ndim / rank
- dtype
- item size
- data buffer
- strides
- row-major contiguous layout
- relationship between logical indices and flat storage

Find the relevant documentation and implementation references inside this repo.

Do not investigate the full Python/C object lifecycle except where it clarifies the array model.

Produce a clean-room representation suitable for a Lisp-like IR, for example:

(array :shape '(2 3)
       :strides '(3 1)
       :dtype 'float64
       :data '(1.0 2.0 3.0 4.0 5.0 6.0))

Identify the metadata necessary for the POC and what can be ignored.

02-shape-strides-and-indexing

Extract the rules needed to map a logical multidimensional index into a flat data buffer.

Focus on:

- row-major contiguous arrays
- shape
- strides
- offset calculation
- positive strides only
- simplified assumptions acceptable for a POC

Produce clean-room pseudocode for:

- default_strides(shape)
- flat_offset(indices, strides)
- get(array, indices)
- set(array, indices, value), only if needed

Mention how NumPy’s architecture generalizes beyond the simplified POC, but do not try to port that generality.

03-broadcasting-rules

Extract NumPy’s broadcasting rules at the semantic level.

Focus on:

- comparing shapes from trailing dimensions
- missing dimensions treated as 1
- dimensions compatible if equal or one is 1
- result broadcast shape
- scalar broadcasting
- failure cases

Find relevant docs/tests/source references inside the repo.

Produce pseudocode for:

broadcast_shape(shape_a, shape_b)

Include examples:

- (2, 3) with (3)
- (2, 3) with scalar
- (3, 1) with (1, 4)
- incompatible shapes

Do not implement dtype promotion or ufunc dispatch.

04-broadcasted-iteration

Extract the architectural idea behind broadcasted iteration.

Focus on:

- iterating over the output shape
- mapping each output index back to each operand
- broadcast dimensions of size 1 reusing the same value
- how this relates to strides or adjusted indexing
- why this abstraction is reusable across elementwise operations

Produce clean-room pseudocode for:

- all_indices(shape)
- project_broadcast_index(output_index, input_shape, output_shape)

Use this conceptual shape:

for each output_index in result_shape:
    a_index = project_index(output_index, a.shape, result_shape)
    b_index = project_index(output_index, b.shape, result_shape)
    out[output_index] = op(a[a_index], b[b_index])

Keep it simple and suitable for a first port.

05-ufunc-elementwise-model

Extract the minimal idea of NumPy’s ufunc-style architecture.

Focus on:

- separating scalar operation kernels from array iteration
- binary elementwise operations
- add and multiply only
- applying one scalar kernel repeatedly over broadcasted operands
- allocating an output array

Find the relevant docs/tests/source references inside the repo.

Do not investigate or port the full C-level ufunc dispatch system.

Produce a clean-room API:

- binary_ufunc(op, a, b)
- add(a, b)
- mul(a, b)

binary_ufunc should handle:

- scalar vs array
- broadcasting
- output shape
- iteration
- flat result storage

06-axis-reduction-model

Extract the minimal model for reducing an array along one axis.

Focus on:

- sum only
- axis=0 and axis=1 for 2D arrays
- output shape after reduction
- iteration strategy
- relation to ufunc reduce, conceptually

Find relevant docs/tests/source references inside the repo.

Do not investigate full NumPy reduction semantics.

Produce clean-room pseudocode for:

sum_axis(array, axis)

Examples:

- shape (2, 3), axis=0 -> shape (3)
- shape (2, 3), axis=1 -> shape (2)

Include suggested parity examples against NumPy.

07-numpy-parity-tests

Design a tiny pytest suite that verifies the clean-room port against NumPy.

Focus only on:

- array construction
- shape
- broadcasting
- add
- multiply
- scalar broadcasting
- sum(axis=0)
- sum(axis=1)

Use small explicit arrays, not large random tests.

Include tests for:

1. 2D + 1D broadcasting
2. 2D * scalar
3. 2D * 1D then sum(axis=1)
4. incompatible broadcast shape raises an error
5. sum(axis=0)
6. sum(axis=1)

Write the proposed tests into the markdown file as code blocks.

08-lisp-ir-port-plan

Design the small Lisp-like target representation and API for the port.

Focus on expressing NumPy’s architecture regularly, not copying NumPy implementation details.

The IR should support:

(array :shape '(2 3)
       :strides '(3 1)
       :dtype 'float64
       :data '(1.0 2.0 3.0 4.0 5.0 6.0))

Functions:

- shape
- strides
- broadcast-shape
- binary-ufunc
- add
- mul
- sum-axis

Produce:

1. proposed Lisp-like surface syntax
2. minimal internal representation
3. small demo program
4. expected output
5. notes on how this could map to a typed IR later

Final synthesis instructions:

After the eight sub-crawler notes exist, read them and write:

agent_runs/00-final-architecture-port-plan.md

Use this structure:

# NumPy-Guided Mini Numerical Core Port Plan

## 1. Goal

Explain that the goal is not to port NumPy, but to demonstrate that agents can extract a bounded architectural slice from a mature numerical library and prepare a clean-room port.

## 2. Why this slice was chosen

Explain why ndarray metadata, strides, broadcasting, ufunc-style iteration, and simple reduction form a meaningful small slice.

## 3. NumPy architectural concepts extracted

### 3.1 ndarray metadata

### 3.2 shape and strides

### 3.3 broadcasting

### 3.4 broadcasted iteration

### 3.5 ufunc-style elementwise operations

### 3.6 axis reduction

## 4. Clean-room Lisp-like target design

Include the proposed array representation and function API.

## 5. Minimal implementation plan

Give a small ordered plan:

1. implement Array record
2. implement default strides
3. implement flat indexing
4. implement broadcast_shape
5. implement broadcasted iterator
6. implement binary_ufunc
7. implement add/mul
8. implement sum_axis
9. implement NumPy parity tests

## 6. Test plan against NumPy

Summarize the tests from 07-numpy-parity-tests.

## 7. Explicit exclusions

Mention at least:

- dtype promotion
- non-contiguous views
- negative strides
- slicing
- advanced indexing
- BLAS
- SIMD
- full ufunc dispatch
- object arrays
- masked arrays
- Pandas
- full NumPy compatibility

## 8. Suggested repo structure

Recommend:

numpy-speedrun-poc/
  README.md
  agent_runs/
    00-final-architecture-port-plan.md
    01-ndarray-metadata.md
    02-shape-strides-and-indexing.md
    03-broadcasting-rules.md
    04-broadcasted-iteration.md
    05-ufunc-elementwise-model.md
    06-axis-reduction-model.md
    07-numpy-parity-tests.md
    08-lisp-ir-port-plan.md
  src/
    mini_array.py
    mini_lisp_ir.py
  tests/
    test_numpy_parity.py
  examples/
    tax_totals_demo.py

## 9. Demo script

Use this demo:

prices = array(
    shape=(2, 3),
    dtype="float64",
    data=[
        10.0, 20.0, 30.0,
        40.0, 50.0, 60.0,
    ],
)

tax = array(
    shape=(3,),
    dtype="float64",
    data=[0.07, 0.08, 0.09],
)

totals = mul(prices, add(1.0, tax))
row_totals = sum_axis(totals, axis=1)

Expected totals:

[[10.7, 21.6, 32.7],
 [42.8, 54.0, 65.4]]

Expected row_totals:

[65.0, 162.2]

## 10. Next steps

Suggest the next small implementation step.

Important constraints:

- Keep the crawl bounded.
- Prefer repo-relative citations/paths.
- Do not copy large source chunks.
- Do not summarize all of NumPy.
- Do not let the sub-crawlers drift into unrelated internals.
- The final output should be understandable in under 10 minutes by a strong engineer.
- The project should look like a controlled demonstration of agentic decomposition, architectural extraction, and clean-room implementation planning.
