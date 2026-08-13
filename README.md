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



# Dense Layer Backpropagation — Single Sample and Batch Case

For a single sample:

```math
z = Wx + b
```

with:

```text
x    : (N,)
W    : (M, N)
b    : (M,)
z    : (M,)
dout : (M,)
```

where:

```math
dout_i = \frac{\partial L}{\partial z_i}
```

### Gradient with respect to the input

For input feature $x_j$:

```math
\frac{\partial L}{\partial x_j}
=
\sum_i
\frac{\partial L}{\partial z_i}
\frac{\partial z_i}{\partial x_j}
```

Since:

```math
\frac{\partial z_i}{\partial x_j} = w_{ij}
```

we get:

```math
\frac{\partial L}{\partial x_j}
=
\sum_i dout_i w_{ij}
```

Therefore:

```math
dX = W^T dout
```

### Gradient with respect to the bias

For bias $b_i$:

```math
\frac{\partial L}{\partial b_i}
=
\frac{\partial L}{\partial z_i}
\frac{\partial z_i}{\partial b_i}
```

Since:

```math
\frac{\partial z_i}{\partial b_i} = 1
```

we get:

```math
db = dout
```

### Gradient with respect to the weights

For weight $w_{ij}$:

```math
\frac{\partial L}{\partial w_{ij}}
=
\frac{\partial L}{\partial z_i}
\frac{\partial z_i}{\partial w_{ij}}
```

Since:

```math
\frac{\partial z_i}{\partial w_{ij}} = x_j
```

we get:

```math
\frac{\partial L}{\partial w_{ij}}
=
dout_i x_j
```

For the whole weight matrix:

```math
dW = dout \, x^T
```

This is the outer product between `dout` and `x`.

---

## Batch Case

With batches:

```text
X    : (B, N)
W    : (M, N)
b    : (M,)
Z    : (B, M)
dout : (B, M)
```

The same weights and biases are shared by every sample in the batch.

For sample $s$:

```math
z_{s,i}
=
\sum_j w_{ij}x_{s,j} + b_i
```

The component-wise derivatives are still the same as in the single-sample case.

The important difference is that the shared parameters $W$ and $b$ receive gradient contributions from every sample in the batch.

### Input gradient

Each sample has its own input, so each sample must also have its own input gradient.

For sample $s$:

```math
\frac{\partial L}{\partial x_{s,j}}
=
\sum_i
\frac{\partial L}{\partial z_{s,i}}w_{ij}
```

Since:

```math
\frac{\partial L}{\partial z_{s,i}}
=
dout_{s,i}
```

we get:

```math
\frac{\partial L}{\partial x_{s,j}}
=
\sum_i dout_{s,i}w_{ij}
```

Vectorized:

```math
dX = dout \, W
```

with shapes:

```text
(B, M) @ (M, N) -> (B, N)
```

The batch dimension remains because every sample has its own input features.

---

### Bias gradient

The same bias $b_i$ is used for every sample.

Therefore, $b_i$ affects:

```math
z_{0,i}, z_{1,i}, \ldots, z_{B-1,i}
```

Using the chain rule:

```math
\frac{\partial L}{\partial b_i}
=
\sum_s
\frac{\partial L}{\partial z_{s,i}}
\frac{\partial z_{s,i}}{\partial b_i}
```

Since:

```math
\frac{\partial z_{s,i}}{\partial b_i} = 1
```

we get:

```math
\frac{\partial L}{\partial b_i}
=
\sum_s dout_{s,i}
```

Vectorized:

```python
db = np.sum(dout, axis=0)
```

with:

```text
(B, M) -> (M,)
```

We sum across the batch axis because there is only one shared bias value for each neuron.

---

### Weight gradient

The same weight $w_{ij}$ is also used by every sample.

For one sample $s$, its contribution to the gradient is:

```math
\left(
\frac{\partial L}{\partial w_{ij}}
\right)_s
=
dout_{s,i}x_{s,j}
```

Since the same weight is used by all samples, all contributions must be accumulated:

```math
\frac{\partial L}{\partial w_{ij}}
=
\sum_s dout_{s,i}x_{s,j}
```

For one sample, the complete weight-gradient matrix is the outer product:

```math
dW_s = dout_s \, x_s^T
```

For the complete batch:

```math
dW
=
\sum_s dout_s \, x_s^T
```

This entire sum can be computed with one matrix multiplication:

```math
dW = dout^T X
```

because:

```text
dout.T : (M, B)
X      : (B, N)

(M, B) @ (B, N) -> (M, N)
```

Each element of the result is:

```math
(dout^T X)_{ij}
=
\sum_s dout_{s,i}x_{s,j}
```

which is exactly the required gradient for weight $w_{ij}$.

---

## Final Batch Formulas

```python
dX = dout @ W
db = np.sum(dout, axis=0)
dW = dout.T @ X
```

The main distinction is:

- `dX` keeps the batch dimension because every sample has its own input.
- `dW` does not have a batch dimension because there is only one shared weight matrix.
- `db` does not have a batch dimension because there is only one shared bias vector.
- Therefore, the gradient contributions from all samples are accumulated for `dW` and `db`.

Whether the batch gradients ultimately correspond to a **sum or a mean over the batch** depends on how the batch loss is defined. That will be handled when implementing the loss function.



## Softmax + Cross-Entropy Loss

The final classification layer of the network produces **logits**, i.e. raw class scores:

[
Z \in \mathbb{R}^{B \times M}
]

where:

* (B): batch size
* (M): number of classes
* (z_{b,i}): logit of class (i) for sample (b)

Instead of implementing Softmax and Cross-Entropy as completely independent operations, they can be combined into a single loss class. This gives a simpler backward pass and allows the loss to be computed in a more numerically stable way.

---

### Softmax

For a single sample, the Softmax probability of class (i) is:

[
p_i =
\frac{e^{z_i}}
{\sum_j e^{z_j}}
]

For a batch:

[
P.shape = (B,M)
]

and each row represents an independent probability distribution.

Therefore Softmax operates across the **class axis** (`axis=1`).

To avoid overflow when exponentiating large logits, the maximum logit of each sample is subtracted:

[
m_b = \max_j z_{b,j}
]

and Softmax is computed as:

[
p_{b,i}
=======

\frac{e^{z_{b,i}-m_b}}
{\sum_j e^{z_{b,j}-m_b}}
]

This is mathematically equivalent to the original Softmax because the common factor (e^{-m_b}) cancels between numerator and denominator.

For the batch:

```python
max_logits = np.max(input, axis=1)
```

Shape:

```text
(B,)
```

To subtract one maximum from every class of the corresponding sample:

```python
input - max_logits[:, np.newaxis]
```

Shapes:

```text
(B,M) - (B,1) -> (B,M)
```

Therefore:

```python
exp_array = np.exp(input - max_logits[:, np.newaxis])
```

has shape:

```text
(B,M)
```

The denominator is computed independently for every sample:

```python
exp_sum = np.sum(exp_array, axis=1)
```

Shape:

```text
(B,)
```

and the probabilities are:

```python
probabilities = exp_array / exp_sum[:, np.newaxis]
```

with broadcasting:

```text
(B,M) / (B,1) -> (B,M)
```

Every row therefore sums to approximately 1.

---

## Cross-Entropy

For a single sample whose correct class is (k):

[
L=-\log(p_k)
]

A high probability for the correct class produces a small loss, while a probability approaching zero produces a very large loss.

With integer class labels, a one-hot vector does not need to be constructed explicitly.

For a batch:

```text
labels.shape = (B,)
```

where:

[
y_b
]

is the correct class index for sample (b).

The correct logit/probability for every sample can be selected using NumPy advanced indexing:

```python
batch_indexes = np.arange(B)
values = array[batch_indexes, labels]
```

which selects:

[
array[0,y_0],;
array[1,y_1],;
\dots,;
array[B-1,y_{B-1}]
]

and produces shape:

```text
(B,)
```

---

## Stable Softmax + Cross-Entropy Forward

Computing:

[
-\log(p_k)
]

directly from the Softmax probability can still create a numerical problem.

Even after subtracting the maximum logit, the exponential corresponding to the correct class may underflow to zero if its logit is much smaller than the maximum:

[
e^{z_k-m}\approx0
]

which would produce:

[
-\log(0)=+\infty
]

Instead, the Softmax and Cross-Entropy expressions can be combined algebraically.

Starting from:

[
L=-\log(p_k)
]

and:

[
p_k=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
]

we obtain:

[
L
=

-\log
\left(
\frac{e^{z_k}}
{\sum_j e^{z_j}}
\right)
]

Using logarithm properties:

[
L
=

-z_k
+
\log
\left(
\sum_j e^{z_j}
\right)
]

Now let:

[
m=\max_j z_j
]

Then:

[
\sum_j e^{z_j}
==============

e^m
\sum_j e^{z_j-m}
]

and therefore:

[
\log
\left(
\sum_j e^{z_j}
\right)
=======

m+
\log
\left(
\sum_j e^{z_j-m}
\right)
]

The final numerically stable single-sample loss is therefore:

[
L
=

-z_k
+
m
+
\log
\left(
\sum_j e^{z_j-m}
\right)
]

This avoids both problems:

* no large positive value is directly exponentiated;
* the quantity inside the logarithm cannot become zero, because at least one shifted logit is exactly zero:

[
e^{m-m}=e^0=1
]

so:

[
\sum_j e^{z_j-m}\ge1
]

For a batch, each sample has its own (m_b):

[
L_b
===

-z_{b,y_b}
+
m_b
+
\log
\left(
\sum_j e^{z_{b,j}-m_b}
\right)
]

The final batch loss is defined as the mean:

[
L_{\text{batch}}
================

\frac{1}{B}
\sum_b L_b
]

so the returned loss is a **scalar**.

---

## Backward — Component-wise Derivation

We want:

[
\frac{\partial L_{\text{batch}}}
{\partial z_{b,i}}
]

for the logit of class (i) of sample (b).

Because:

[
L_{\text{batch}}
================

\frac{1}{B}
\sum_s L_s
]

the probabilities/logits of sample (b) affect only (L_b), not the losses of the other samples.

Therefore:

[
\frac{\partial L_{\text{batch}}}
{\partial z_{b,i}}
==================

\frac{1}{B}
\frac{\partial L_b}
{\partial z_{b,i}}
]

For sample (b), let (k=y_b) be the correct class:

[
L_b=-\log(p_{b,k})
]

Thus:

[
\frac{\partial L_b}
{\partial p_{b,k}}
==================

-\frac{1}{p_{b,k}}
]

Although the Cross-Entropy directly depends only on the probability of the correct class, that probability depends on **every logit of the same sample** through Softmax.

Therefore:

[
\frac{\partial L_{\text{batch}}}
{\partial z_{b,i}}
==================

\frac{1}{B}
\left(
-\frac{1}{p_{b,k}}
\right)
\frac{\partial p_{b,k}}
{\partial z_{b,i}}
]

The Softmax derivative has two cases.

### Incorrect class: (i\neq k)

[
\frac{\partial p_{b,k}}
{\partial z_{b,i}}
==================

-p_{b,k}p_{b,i}
]

Therefore:

[
\frac{\partial L_{\text{batch}}}
{\partial z_{b,i}}
==================

\frac{1}{B}
\left(
-\frac{1}{p_{b,k}}
\right)
(-p_{b,k}p_{b,i})
]

and:

[
\boxed{
\frac{\partial L_{\text{batch}}}
{\partial z_{b,i}}
==================

\frac{p_{b,i}}{B}
}
]

for (i\neq k).

### Correct class: (i=k)

For the correct class:

[
\frac{\partial p_{b,k}}
{\partial z_{b,k}}
==================

p_{b,k}(1-p_{b,k})
]

Therefore:

[
\frac{\partial L_{\text{batch}}}
{\partial z_{b,k}}
==================

\frac{1}{B}
\left(
-\frac{1}{p_{b,k}}
\right)
p_{b,k}(1-p_{b,k})
]

which simplifies to:

[
\boxed{
\frac{\partial L_{\text{batch}}}
{\partial z_{b,k}}
==================

\frac{p_{b,k}-1}{B}
}
]

Therefore the complete component-wise result is:

[
\frac{\partial L_{\text{batch}}}
{\partial z_{b,i}}
==================

\begin{cases}
\dfrac{p_{b,i}}{B},
& i\neq y_b[6pt]

\dfrac{p_{b,i}-1}{B},
& i=y_b
\end{cases}
]

---

## Vectorized Backward

If the labels were represented using a one-hot matrix:

[
Y.shape=(B,M)
]

the complete derivative could be written compactly as:

[
\boxed{
\frac{\partial L_{\text{batch}}}{\partial Z}
============================================

\frac{P-Y}{B}
}
]

However, constructing a full one-hot matrix is unnecessary when the labels are stored as integer class indices.

First initialize every gradient with:

[
\frac{p_{b,i}}{B}
]

for all samples and classes.

Then, for each sample (b), subtract (1/B) only from its correct class:

```python
gradients = self.probabilities / self.batch_size
gradients[batch_indexes, self.labels] -= 1 / self.batch_size
```

The first operation gives:

```text
gradients.shape = (B,M)
```

with:

[
gradients_{b,i}
===============

\frac{p_{b,i}}{B}
]

The advanced-indexing operation selects exactly:

[
[b,y_b]
]

for every sample and changes the correct-class gradient to:

[
\frac{p_{b,y_b}}{B}
-------------------

# \frac1B

\frac{p_{b,y_b}-1}{B}
]

This produces exactly the same result as:

[
(P-Y)/B
]

without explicitly constructing (Y).

---

## What This Derivation Achieves

Combining Softmax and Cross-Entropy gives several useful results.

First, the backward pass becomes extremely simple:

[
\frac{\partial L}{\partial Z}
=============================

\frac{P-Y}{B}
]

instead of explicitly propagating the Cross-Entropy gradient through the full Softmax Jacobian.

Second, the implementation does not need to construct one-hot target vectors. Integer labels are enough, because NumPy advanced indexing allows the correct class of every sample to be selected directly.

Third, the forward loss can be computed directly from the logits using the stable **log-sum-exp** form. This avoids both large exponentials and the possibility of evaluating `log(0)` after a correct-class Softmax probability underflows to zero.

Finally, the batch dimension remains conceptually simple:

* each sample has its own logits and probability distribution;
* Softmax never mixes information between different samples;
* each sample contributes independently to the total loss;
* the final batch loss is the mean of those sample losses;
* therefore every sample's gradient receives the factor (1/B).

The resulting layer takes logits of shape:

```text
(B,M)
```

and integer labels of shape:

```text
(B,)
```

returns a scalar training loss in the forward pass, and returns:

```text
(B,M)
```

in the backward pass, representing:

[
\frac{\partial L_{\text{batch}}}{\partial Z}
]

which can then be passed directly into the backward method of the final Dense layer.
