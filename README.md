## MaxPooling2D Forward and Backward — Tracking the Maximum Element

The key idea behind MaxPooling is simple:

* During the **forward pass**, each pooling window outputs only its maximum value.
* During the **backward pass**, the gradient must be sent back only to the input element that produced that maximum.

Therefore, during the forward pass we must store enough information to remember **where each maximum came from**.

---

### Forward pass

The input uses NHWC format:

```python
input.shape == (B, H, W, C)
```

where:

* `B` = batch size
* `H` = input height
* `W` = input width
* `C` = number of channels

For every spatial position of the pooling window, we extract:

```python
window = input[:, row:row + Kh, col:col + Kw, :]
```

with shape:

```python
(B, Kh, Kw, C)
```

This means that:

```python
window[b, :, :, c]
```

is the actual 2D pooling window of shape:

```python
(Kh, Kw)
```

for image `b` and channel `c`.

For example, if:

```python
pool_size = (2, 2)
```

then for a particular image and channel we may have:

```text
3  8
5  2
```

The maximum is `8`.

To obtain the maximum value for every image and every channel simultaneously:

```python
max_element = np.max(window, axis=(1, 2))
```

The spatial axes `Kh` and `Kw` disappear, leaving:

```python
max_element.shape == (B, C)
```

Therefore:

```python
max_element[b, c]
```

is the maximum value of the pooling window for image `b` and channel `c`.

That `(B, C)` result can then be written into one spatial position of the output:

```python
output[:, output_row, output_col, :] = max_element
```

---

## Storing the position of the maximum

During backpropagation, knowing only the maximum value is not enough.

We also need to know **which input element produced that maximum**.

The spatial part of the window currently has two dimensions:

```python
(Kh, Kw)
```

but `np.argmax()` operates along a single axis.

Therefore, the two spatial axes are first flattened:

```python
flat_window = np.reshape(window, (B, Kh * Kw, C))
```

Now:

```python
flat_window.shape == (B, Kh * Kw, C)
```

The axes mean:

```text
axis 0 -> batch
axis 1 -> flattened spatial window
axis 2 -> channels
```

Then:

```python
np.argmax(flat_window, axis=1)
```

asks NumPy to find the maximum along the flattened spatial window.

For every fixed pair `(b, c)`, NumPy effectively looks at:

```python
flat_window[b, :, c]
```

which is a vector of length:

```python
Kh * Kw
```

and returns the index of its maximum.

Therefore:

```python
np.argmax(flat_window, axis=1).shape == (B, C)
```

Each element:

```python
max_element_idx[b, c]
```

is the flattened position of the maximum inside the local pooling window for image `b` and channel `c`.

For a `2x2` window:

```text
0  1
2  3
```

so if the maximum is at local position:

```text
row = 1
col = 0
```

its flattened index is:

```text
2
```

These indices are stored in:

```python
self.max_element_idx[:, output_row, output_col, :]
```

Therefore, after the complete forward pass:

```python
self.max_element_idx[b, output_row, output_col, c]
```

contains the **local flattened index** of the maximum element for image `b`, channel `c`, and that particular pooling-window position.

An important distinction:

```text
self.max_element_idx
```

does **not** contain coordinates in the original input image.

It contains coordinates relative to the local pooling window.

For example, if:

```text
pool_size = (2, 2)
```

then every stored index is between:

```text
0 and 3
```

regardless of where the pooling window is located in the original image.

---

# Backward pass

Suppose the upstream gradient has shape:

```python
dout.shape == (B, H_out, W_out, C)
```

The goal is to construct:

```python
din.shape == (B, H, W, C)
```

Initially:

```python
din = np.zeros(self.input_shape)
```

MaxPooling only lets one input element influence each output element: the maximum.

Therefore, for each output gradient:

```python
dout[b, output_row, output_col, c]
```

we must send that gradient back to the input element that was the maximum during the forward pass.

Every other element in the pooling window receives zero gradient.

---

## The simple loop-based interpretation

The easiest way to understand the backward pass is to first imagine a completely explicit version.

Conceptually, we could write:

```python
for output_row in range(H_out):
    for output_col in range(W_out):
        input_row = output_row * stride
        input_col = output_col * stride

        for b in range(B):
            for c in range(C):
                flat_idx = self.max_element_idx[b, output_row, output_col, c]

                # Convert flat_idx into local window coordinates
                window_row, window_col = ...

                global_row = input_row + window_row
                global_col = input_col + window_col

                din[b, global_row, global_col, c] += dout[b, output_row, output_col, c]
```

This is the fundamental operation.

The vectorized implementation does exactly the same thing, but removes the inner loops over:

```text
b
c
```

---

# Recovering the 2D position from the flattened index

At a particular output position:

```python
flat_window_idxs = self.max_element_idx[:, output_row, output_col, :]
```

has shape:

```python
(B, C)
```

Each element:

```python
flat_window_idxs[b, c]
```

is the flattened local position of the maximum inside the pooling window.

We now want to recover the corresponding local:

```text
(row, col)
```

coordinates.

This is what:

```python
np.unravel_index()
```

does.

For example, for:

```python
pool_size = (2, 3)
```

the local flattened indices correspond to:

```text
0  1  2
3  4  5
```

Therefore:

```text
0 -> (0, 0)
1 -> (0, 1)
2 -> (0, 2)
3 -> (1, 0)
4 -> (1, 1)
5 -> (1, 2)
```

If:

```python
flat_window_idxs = [[0, 4],
                    [5, 2]]
```

then:

```python
window_rows = [[0, 1],
               [1, 0]]
```

and:

```python
window_cols = [[0, 1],
               [2, 2]]
```

Both arrays have the same shape as `flat_window_idxs`:

```python
(B, C)
```

A useful conceptual interpretation is:

```text
for every b:
    for every c:
        flat_idx = flat_window_idxs[b, c]
        row, col = unravel(flat_idx)
        window_rows[b, c] = row
        window_cols[b, c] = col
```

NumPy performs this operation vectorized, but this loop-based interpretation makes its meaning clearer.

Importantly, `np.unravel_index()` does not create one `(B, C)` array whose elements are tuples.

Instead, it returns separate coordinate arrays:

```python
window_rows
window_cols
```

so that:

```python
window_rows[b, c]
window_cols[b, c]
```

together describe the 2D local position of the maximum.

---

# Converting local window coordinates to global input coordinates

The pooling window starts at:

```python
input_row = output_row * stride
input_col = output_col * stride
```

The coordinates returned by `unravel_index()` are local coordinates inside that window.

Therefore:

```python
global_rows = input_row + window_rows
global_cols = input_col + window_cols
```

Both arrays still have shape:

```python
(B, C)
```

For example, suppose:

```python
input_row = 4
input_col = 7
```

and:

```python
window_rows = [[0, 1],
               [1, 0]]

window_cols = [[0, 1],
               [2, 2]]
```

Then:

```python
global_rows = [[4, 5],
               [5, 4]]
```

and:

```python
global_cols = [[7, 8],
               [9, 9]]
```

For example:

```python
global_rows[1, 0] == 5
global_cols[1, 0] == 9
```

means:

> For image `b=1` and channel `c=0`, when the pooling window started at input position `(4, 7)`, the maximum element of that window was located at position `(5, 9)` in the original input.

---

# What we now want to do

At this point, for every `(b, c)` we know the exact input position where the gradient must be sent.

The simple loop-based operation would be:

```python
for b in range(B):
    for c in range(C):
        din[b, global_rows[b, c], global_cols[b, c], c] += dout[b, output_row, output_col, c]
```

In words:

> For every image and every channel, take the gradient of the current MaxPooling output position and send it back to the original input position that produced the maximum.

The remaining goal is simply to vectorize these two loops.

---

# Vectorizing over batch and channels with advanced indexing

We already have:

```python
global_rows.shape == (B, C)
global_cols.shape == (B, C)
```

We now also need arrays that identify which batch element and channel correspond to each position `(b, c)`.

For the batch:

```python
batch_idx = np.arange(B)[:, np.newaxis]
```

which has shape:

```python
(B, 1)
```

For example, with:

```text
B = 2
```

we have:

```text
[[0],
 [1]]
```

For the channels:

```python
channel_idx = np.arange(C)[np.newaxis, :]
```

which has shape:

```python
(1, C)
```

For example, with:

```text
C = 3
```

we have:

```text
[[0, 1, 2]]
```

Through broadcasting, these can conceptually be viewed as:

```text
batch_idx:

[[0, 0, 0],
 [1, 1, 1]]
```

and:

```text
channel_idx:

[[0, 1, 2],
 [0, 1, 2]]
```

Both therefore correspond to the same conceptual `(B, C)` grid as:

```python
global_rows
global_cols
```

Now consider:

```python
din[batch_idx, global_rows, global_cols, channel_idx]
```

NumPy uses the broadcasted index arrays **position-wise**.

Suppose:

```text
B = 2
C = 3
```

Then the selected coordinates are conceptually:

```text
din[0, global_rows[0,0], global_cols[0,0], 0]
din[0, global_rows[0,1], global_cols[0,1], 1]
din[0, global_rows[0,2], global_cols[0,2], 2]

din[1, global_rows[1,0], global_cols[1,0], 0]
din[1, global_rows[1,1], global_cols[1,1], 1]
din[1, global_rows[1,2], global_cols[1,2], 2]
```

In other words, each common `(b, c)` position of the broadcasted index arrays constructs one complete four-dimensional coordinate:

```text
(b, global_rows[b, c], global_cols[b, c], c)
```

This is exactly the same coordinate that appeared in the explicit nested loops.

The advanced-indexing expression therefore vectorizes the two inner loops over batch and channels.

---

# Matching the upstream gradient

For the current output spatial position:

```python
dout[:, output_row, output_col, :]
```

has shape:

```python
(B, C)
```

It contains the upstream gradient for every image and every channel at that particular MaxPooling output position.

The advanced-indexed left-hand side:

```python
din[batch_idx, global_rows, global_cols, channel_idx]
```

also refers to one element for every `(b, c)` pair and therefore has the same conceptual shape:

```python
(B, C)
```

The assignment:

```python
din[batch_idx, global_rows, global_cols, channel_idx] += dout[:, output_row, output_col, :]
```

therefore performs, element-by-element:

```text
dout[b, output_row, output_col, c]
```

into:

```text
din[b, global_rows[b,c], global_cols[b,c], c]
```

for every image `b` and channel `c`.

---

# Why `+=` instead of `=`

The gradient must be **accumulated**, not simply assigned.

This matters when pooling windows overlap.

If two different pooling windows share an input element, and that same element happens to be the maximum for both windows, then two different output values depend on the same input value.

During backpropagation, both gradient paths must contribute to that input element.

Therefore:

```python
+=
```

is required.

Conceptually:

```text
din[input element] = gradient contribution from window 1 + gradient contribution from window 2 + ...
```

---

# Final mental model

The entire MaxPooling backward pass can be summarized as:

> During the forward pass, every output value remembers which input element won the maximum operation.
>
> During the backward pass, each output gradient is routed back only to that winning input element.

For a fixed pooling-window position:

1. Retrieve the stored local flattened maximum index for every image and channel.
2. Convert each flat index back into local `(row, col)` coordinates.
3. Add the pooling window's starting position to obtain global coordinates in the original input.
4. Take all `dout` values at the corresponding output spatial position.
5. For each `(b, c)`, add that gradient to:

```python
din[b, global_rows[b,c], global_cols[b,c], c]
```

The explicit version requires four loops:

```text
output_row
output_col
batch
channel
```

The vectorized version keeps only the two spatial loops:

```text
output_row
output_col
```

and replaces the inner batch/channel loops with NumPy broadcasting and advanced indexing.

So the final vectorized operation is not a mysterious NumPy trick. It is simply the compact form of:

```python
for b in range(B):
    for c in range(C):
        din[b, global_rows[b,c], global_cols[b,c], c] += dout[b, output_row, output_col, c]
```

That loop-based interpretation is the most useful way to reason about the advanced indexing whenever the vectorized expression becomes difficult to read.



## Dense Layer Backpropagation — Single Sample and Batch Case

For a single sample:

[
z = Wx + b
]

with:

```text
x    : (N,)
W    : (M, N)
b    : (M,)
z    : (M,)
dout : (M,)
```

where:

[
dout_i = \frac{\partial L}{\partial z_i}
]

### Gradient with respect to the input

For input feature (x_j):

[
\frac{\partial L}{\partial x_j}
===============================

\sum_i
\frac{\partial L}{\partial z_i}
\frac{\partial z_i}{\partial x_j}
]

Since:

[
\frac{\partial z_i}{\partial x_j}=w_{ij}
]

we get:

[
\frac{\partial L}{\partial x_j}
===============================

\sum_i dout_i w_{ij}
]

Therefore:

[
\boxed{dX = W^T dout}
]

### Gradient with respect to the bias

For bias (b_i):

[
\frac{\partial L}{\partial b_i}
===============================

\frac{\partial L}{\partial z_i}
\frac{\partial z_i}{\partial b_i}
]

and since:

[
\frac{\partial z_i}{\partial b_i}=1
]

we get:

[
\boxed{db = dout}
]

### Gradient with respect to the weights

For weight (w_{ij}):

[
\frac{\partial L}{\partial w_{ij}}
==================================

\frac{\partial L}{\partial z_i}
\frac{\partial z_i}{\partial w_{ij}}
]

Since:

[
\frac{\partial z_i}{\partial w_{ij}}=x_j
]

we get:

[
\frac{\partial L}{\partial w_{ij}}
==================================

dout_i x_j
]

For the whole weight matrix:

[
\boxed{dW = dout , x^T}
]

which is the outer product of `dout` and `x`.

---

# Batch case

With batches:

```text
X    : (B, N)
W    : (M, N)
b    : (M,)
Z    : (B, M)
dout : (B, M)
```

The same weights and biases are shared by every sample.

For sample (s):

[
z_{s,i}
=======

\sum_j w_{ij}x_{s,j}+b_i
]

The component-wise derivatives are still the same as in the single-sample case. The difference is that shared parameters receive gradient contributions from every sample in the batch.

### Input gradient

Each sample has its own input, so its input gradient must remain separate.

For each sample:

[
\frac{\partial L}{\partial x_{s,j}}
===================================

\sum_i dout_{s,i}w_{ij}
]

Vectorized:

[
\boxed{dX = dout , W}
]

with:

```text
(B, M) @ (M, N) -> (B, N)
```

No summation over the batch is performed because each sample has its own input features.

---

### Bias gradient

The same bias (b_i) is used for every sample.

Therefore:

[
\frac{\partial L}{\partial b_i}
===============================

\sum_s
\frac{\partial L}{\partial z_{s,i}}
]

so:

[
\boxed{
db_i = \sum_s dout_{s,i}
}
]

Vectorized:

```python
db = np.sum(dout, axis=0)
```

with:

```text
(B, M) -> (M,)
```

The batch dimension disappears because there is only one shared bias vector.

---

### Weight gradient

The same weight (w_{ij}) is also used by every sample.

For one sample (s), its contribution is:

[
\left(\frac{\partial L}{\partial w_{ij}}\right)_s
=================================================

dout_{s,i}x_{s,j}
]

The total gradient must therefore sum the contributions from all samples:

[
\frac{\partial L}{\partial w_{ij}}
==================================

\sum_s dout_{s,i}x_{s,j}
]

For the complete weight matrix:

[
\boxed{
dW = dout^T X
}
]

because:

```text
dout.T : (M, B)
X      : (B, N)

(M, B) @ (B, N) -> (M, N)
```

and each element of the result is:

[
(dout^T X)_{ij}
===============

\sum_s dout_{s,i}x_{s,j}
]

---

## Final batch formulas

```python
dX = dout @ W
db = np.sum(dout, axis=0)
dW = dout.T @ X
```

The important distinction is:

* `dX` keeps the batch dimension because every sample has its own input.
* `dW` has no batch dimension because there is only one shared weight matrix.
* `db` has no batch dimension because there is only one shared bias vector.
* Therefore, the gradient contributions from all samples must be accumulated for `dW` and `db`.
