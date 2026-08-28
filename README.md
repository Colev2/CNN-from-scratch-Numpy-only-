# CNN from Scratch with NumPy

#### Video Demo: https://youtu.be/CfTh9dmLmEw?si=aSngNFH0FmFdadTB
#### Description:

A small convolutional neural network framework implemented **from scratch with NumPy**.

In this project I implemented a CNN without using a deep-learning framework. `torchvision` is used only to download and load the supported image datasets. The neural-network forward pass, backpropagation, parameter updates, loss function, regularization, and training loop are implemented using only the NumPy library.

The program can train the same CNN pipeline on four datasets:

- **MNIST**
- **Fashion-MNIST**
- **CIFAR-10**
- **CIFAR-100**

---

## Features

The project currently includes:

- 2D convolution with configurable filters, kernel size, padding, and stride
- Vectorized convolution using NumPy window views and matrix multiplication
- Backward pass for:
  - input gradients
  - weight gradients
  - bias gradients
- Max pooling 
- Fully connected (`Dense`) layers
- ReLU activation
- Flatten layer
- Batch Normalization for both convolutional and dense inputs
- Inverted Dropout
- Numerically stable Softmax + Cross-Entropy loss
- He and Xavier weight initialization
- Normal and uniform initialization distributions
- SGD
- SGD with Momentum
- Adam
- AdamW with decoupled weight decay
- `ReduceLROnPlateau` learning-rate scheduling
- Early stopping with restoration of the best model weights
- Data Augmentation with random crop and random horizontal-flip
- Stratified train/validation splitting
- Mini-batch training with optional shuffling
- Training/evaluation modes for layers such as BatchNorm and Dropout
- Numerical gradient checks with `pytest`

---

## Best Training Results

Training a CNN implemented entirely with NumPy is computationally expensive, since both the forward and backward passes involve large matrix operations without the highly optimized kernels used by modern deep-learning frameworks.

Each epoch takes several minutes to complete, since forward and backward passes are heavy multiplication operations. I tried to optimize them to the best of my ability. Obviously, there is a lot more that can be done. 

The best validation performance was obtained with the following configuration:

| Configuration           | Value                         |
| ----------------------- | ----------------------------- |
| Dataset                 | CIFAR-10                      |
| Optimizer               | Adam                          |
| Initial learning rate   | `0.001`                       |
| Batch size              | `32`                          |
| Early Stopping patience | `7`                           |
| LR Scheduler            | ReduceLROnPlateau             |
| LR reduction factor     | `0.5`                         |
| LR Scheduler patience   | `3`                           |
| Dropout probability     | `0.5`                         |
| BatchNorm momentum      | `0.99`                        |
| BatchNorm epsilon       | `1e-3`                        |
| Data augmentation       | Random crop + horizontal flip |

### Architecture

```text
Conv2D(32) → BatchNorm → ReLU
Conv2D(32) → BatchNorm → ReLU
MaxPooling2D

Conv2D(64) → BatchNorm → ReLU
Conv2D(64) → BatchNorm → ReLU
MaxPooling2D

Flatten
Dropout(0.5)

Dense(256) → BatchNorm → ReLU
Dense(10)
```

## Best Training Results

Most of the development, experimentation, and hyperparameter tuning for this project was performed on **CIFAR-10**, since it provides a more challenging benchmark than MNIST and Fashion-MNIST while remaining practical to train with a NumPy-only implementation.

The framework also supports **MNIST, Fashion-MNIST, and CIFAR-100**, and the training pipeline can be run on any of the four datasets through the interactive dataset selection.

The results reported below correspond specifically to **CIFAR-10**.


### Best Validation Result - CIFAR-10

The best model weights were obtained at **epoch 29**:

| Metric              |      Result |
| ------------------- | ----------: |
| Training accuracy   |  **84.71%** |
| Training loss       |  **0.4326** |
| Validation accuracy |  **87.52%** |
| Validation loss     |  **0.3634** |
| Learning rate       | **0.00025** |

The learning-rate scheduler reduced the learning rate during training when validation loss stopped improving. Early stopping tracked the best validation loss and restored the weights from epoch 29 after training.

### Final Test Result - CIFAR-10

| Metric | Result |
|---|---:|
| Test accuracy | **XX.XX%** |
| Test loss | **X.XXXX** |

### Training History

|  Epoch | Train Accuracy | Train Loss | Validation Accuracy | Validation Loss | Learning Rate |
| -----: | -------------: | ---------: | ------------------: | --------------: | ------------: |
|      1 |         47.39% |     1.4575 |              59.92% |          1.1413 |        0.0010 |
|      2 |         60.95% |     1.0926 |              67.72% |          0.8766 |        0.0010 |
|      3 |         66.02% |     0.9605 |              73.80% |          0.7439 |        0.0010 |
|      4 |         69.52% |     0.8709 |              72.70% |          0.7854 |        0.0010 |
|      5 |         71.04% |     0.8184 |              74.24% |          0.7443 |        0.0010 |
|      6 |         72.60% |     0.7757 |              77.40% |          0.6290 |        0.0010 |
|      7 |         73.76% |     0.7447 |              76.00% |          0.6998 |        0.0010 |
|      8 |         74.93% |     0.7080 |              76.66% |          0.6664 |        0.0010 |
|      9 |         75.92% |     0.6839 |              78.22% |          0.6349 |        0.0010 |
|     10 |         76.83% |     0.6635 |              80.30% |          0.5776 |        0.0010 |
|     11 |         77.62% |     0.6417 |              80.26% |          0.5609 |        0.0010 |
|     12 |         78.39% |     0.6194 |              81.22% |          0.5323 |        0.0010 |
|     13 |         78.49% |     0.6149 |              80.48% |          0.5677 |        0.0010 |
|     14 |         79.12% |     0.5986 |              79.50% |          0.6238 |        0.0010 |
|     15 |         79.77% |     0.5801 |              83.44% |          0.4794 |        0.0010 |
|     16 |         80.14% |     0.5695 |              82.86% |          0.4848 |        0.0010 |
|     17 |         80.43% |     0.5623 |              82.52% |          0.5042 |        0.0010 |
|     18 |         80.88% |     0.5513 |              82.84% |          0.5023 |        0.0010 |
|     19 |         81.21% |     0.5409 |              82.02% |          0.5179 |        0.0010 |
|     20 |         82.46% |     0.5042 |              86.00% |          0.4055 |        0.0005 |
|     21 |         83.28% |     0.4878 |              86.30% |          0.3942 |        0.0005 |
|     22 |         83.23% |     0.4818 |              86.70% |          0.3974 |        0.0005 |
|     23 |         83.47% |     0.4731 |              86.62% |          0.3932 |        0.0005 |
|     24 |         83.80% |     0.4685 |              86.72% |          0.3833 |        0.0005 |
|     25 |         83.97% |     0.4651 |              86.58% |          0.3931 |        0.0005 |
|     26 |         83.96% |     0.4602 |              86.32% |          0.3935 |        0.0005 |
|     27 |         84.32% |     0.4531 |              86.70% |          0.3969 |        0.0005 |
|     28 |         84.33% |     0.4462 |              85.88% |          0.4116 |        0.0005 |
| **29** |     **84.71%** | **0.4326** |          **87.52%** |      **0.3634** |   **0.00025** |
|     30 |         85.13% |     0.4258 |              87.42% |          0.3657 |       0.00025 |

The complete architecture and implementation of each layer are described in more detail below.




## Why This Project?

My purpose on starting this project was to get a deeper understanding of what is happening "under the hood" when a modern framework like Pytorch or Keras executes forward(x) and backward(x). I wanted to go deeper into the mathematics and implementation details that are "hidden" inside them, and to try to build a complete, decently-structured CNN pipeline without using a framework. I've built CNNs both in Keras and Pytorch, in the context of the deep learning courses I took at university, so I  had an okay understanding of the theory, but never implemented one from scratch. This was my first try.

More specifically, in this project I had to: 

1. construct the network layer by layer,
2. compute the forward pass,
3. calculate the loss,
4. manually propagate gradients backward,
5. update the parameters with an optimizer,
6. regularize training,
7. evaluate the model on unseen data.

Of course NumPy is still used for efficient array operations and vectorization.

---

## Project Structure

```text
.
├── project.py
├── test_project.py
│
└── cnn_numpy/
    ├── sequential.py
    ├── optimizers.py
    ├── early_stopping.py
    ├── lr_scheduler.py
    ├── data_augmentation.py
    │
    ├── layers/
    │   ├── conv.py
    │   ├── batchnorm.py
    │   ├── relu.py
    │   ├── maxpooling.py
    │   ├── flatten.py
    │   ├── dropout.py
    │   └── dense.py
    │
    └── losses/
        └── softmax_cross_entropy_loss.py
```

A `data/` directory is created automatically next to `project.py` when a dataset is downloaded.

---

## Supported Datasets

The dataset is selected interactively when the program starts.

| Dataset | Input type | Classes | Default augmentation |
|---|---|---:|---|
| MNIST | Grayscale | 10 | Random crop, padding 2 |
| Fashion-MNIST | Grayscale | 10 | Random crop, padding 2 + horizontal flip |
| CIFAR-10 | RGB | 10 | Random crop, padding 4 + horizontal flip |
| CIFAR-100 | RGB | 100 | Random crop, padding 4 + horizontal flip |

The loader automatically adds a channel dimension to grayscale datasets so that all images follow the same internal format:

```text
(B, H, W, C)
```

where:

- `B` = batch size
- `H` = image height
- `W` = image width
- `C` = number of channels

---

## Model Architecture

The current training program builds the following CNN:

```text
Input
  │
  ├── Conv2D(32, 3×3, padding=1)
  ├── BatchNorm
  ├── ReLU
  │
  ├── Conv2D(32, 3×3, padding=1)
  ├── BatchNorm
  ├── ReLU
  │
  ├── MaxPooling2D(2×2, stride=2)
  │
  ├── Conv2D(64, 3×3, padding=1)
  ├── BatchNorm
  ├── ReLU
  │
  ├── Conv2D(64, 3×3, padding=1)
  ├── BatchNorm
  ├── ReLU
  │
  ├── MaxPooling2D(2×2, stride=2)
  │
  ├── Flatten
  ├── Dropout(p=0.5)
  │
  ├── Dense(256)
  ├── BatchNorm
  ├── ReLU
  │
  └── Dense(number_of_classes)
       │
       └── Logits
```

The final layer contains:

- 10 outputs for MNIST
- 10 outputs for Fashion-MNIST
- 10 outputs for CIFAR-10
- 100 outputs for CIFAR-100

The output layer returns **logits**. Softmax is applied internally by the loss implementation rather than being used as a separate model layer.

---

## Layers

### `Conv2D`

The convolution layer supports:

- configurable number of filters
- configurable kernel size
- padding
- stride
- He or Xavier initialization
- normal or uniform initialization
- `float32` by default

The forward pass uses:

```python
np.lib.stride_tricks.sliding_window_view
```

to obtain convolution windows and then performs the convolution using matrix multiplication.

Conceptually:

```text
windows @ weights.T
```

This avoids deeply nested Python loops over the batch, spatial positions, filters, and channels.

The backward pass manually computes:

```text
dL/dW
dL/db
dL/dX
```

Weight gradients are obtained by combining the upstream gradients with the convolution windows. Input gradients are first computed for every element of every window and then accumulated back into the corresponding input positions.

---

### `MaxPooling2D`

Max pooling also uses `sliding_window_view` during its forward pass.

For each pooling window, the layer stores the index of the maximum element. During backpropagation, the upstream gradient is routed only to the location that produced the maximum value.

---

### `Dense`

The fully connected layer implements:

```text
Z = XWᵀ + b
```

and manually computes:

```text
dL/dX = dL/dZ · W
dL/dW = (dL/dZ)ᵀ · X
dL/db = Σ dL/dZ
```

---

### `BatchNorm`

Batch Normalization supports both:

```text
(B, H, W, C)
```

and:

```text
(B, D)
```

inputs.

During training it computes batch statistics and updates running statistics using momentum. During evaluation it uses the stored running mean and running variance.

The implementation includes trainable:

```text
gamma
beta
```

parameters and manually computes their gradients together with the input gradient.

---

### `ReLU`

The activation is:

```text
ReLU(x) = max(0, x)
```

The forward pass stores a boolean mask indicating which inputs were positive. The backward pass multiplies the incoming gradient by this mask.

The convention used at zero is:

```text
ReLU'(0) = 0
```

---

### `Flatten`

`Flatten` converts convolutional feature maps from:

```text
(B, H, W, C)
```

to:

```text
(B, H·W·C)
```

and restores the original shape during the backward pass.

---

### `Dropout`

The project implements **inverted dropout**.

During training, activations are randomly masked and scaled by:

```text
1 / (1 - drop_probability)
```

During evaluation, Dropout becomes an identity operation, so no additional scaling is required.

The current model uses:

```text
drop_prob = 0.5
```

before the first dense layer.

---

## Loss Function

The project combines Softmax and Cross-Entropy into a single:

```python
SoftmaxCrossEntropyLoss
```

implementation.

For numerical stability, logits are shifted by the maximum logit of each sample before exponentiation.

The backward pass uses the standard result:

```text
dL/dZ = (P - Y) / B
```

where:

- `P` is the softmax probability matrix
- `Y` is the one-hot target matrix
- `B` is the batch size

Labels are stored as integer class indices, so a full one-hot matrix does not need to be explicitly constructed.

---

## Sequential Model

`Sequential` receives an ordered list of layers.

It is responsible for:

- building every layer using the previous layer's output shape
- executing the forward pass
- executing the backward pass in reverse order
- collecting trainable parameters and gradients
- collecting parameters eligible for weight decay
- saving and restoring model weights
- switching all layers between training and evaluation mode

Example:

```python
model = Sequential([
    Conv2D(filters=32),
    BatchNorm(epsilon=1e-3, momentum=0.99),
    ReLU(),
    MaxPooling2D(),
    Flatten(),
    Dense(neurons=10),
])

model.build(input_shape)
```

---

## Weight Initialization

Convolutional and Dense layers support:

### He initialization

Designed primarily for ReLU-based networks.

### Xavier initialization

Designed to keep activation variance more stable across layers.

Both can use either:

```text
normal
```

or:

```text
uniform
```

distributions.

When running `project.py`, the selected initialization and distribution are used for the convolutional layers. The current architecture explicitly uses He/normal initialization for the hidden Dense layer and Xavier/normal initialization for the output Dense layer.

---

## Optimizers

### SGD

Standard gradient descent:

```text
θ ← θ - ηg
```

---

### SGD with Momentum

Maintains a running update:

```text
uₜ = μuₜ₋₁ + gₜ
θ ← θ - ηuₜ
```

The training program uses:

```text
momentum = 0.9
```

---

### Adam

Implements:

- first-moment estimation
- second-moment estimation
- bias correction

The default values used by `project.py` are:

```text
β₁ = 0.9
β₂ = 0.999
ε  = 1e-8
```

---

### AdamW

AdamW uses the same adaptive Adam update together with **decoupled weight decay**.

Weight decay is applied only to parameters returned by:

```python
model.decayable_parameters()
```

In the current implementation, convolutional and dense **weight matrices** are decayable, while biases and BatchNorm parameters are not.

---

## Learning-Rate Scheduling

The project includes:

```python
ReduceLROnPlateau
```

which monitors validation loss.

The current training configuration uses:

```text
factor    = 0.5
patience  = 3
min_delta = 1e-3
min_lr    = 1e-5
```

If validation loss stops improving sufficiently, the learning rate is reduced while respecting the configured minimum.

---

## Early Stopping

Training also uses early stopping.

Current configuration:

```text
patience  = 7
min_delta = 1e-3
```

Whenever validation loss improves, a copy of the model weights is stored.

When training finishes, the program restores the weights from the best validation epoch instead of keeping the weights from the final epoch.

---

## Data Augmentation

Two augmentation operations are implemented.

### Random crop

The image is padded and a crop with the original image dimensions is sampled from a random position.

### Random horizontal flip

Each image is horizontally flipped with probability:

```text
0.5
```

Augmentation is applied **only to training batches**.

A new random transformation can therefore be seen on each pass through the training data, while validation data remains unchanged.

---

## Data Preprocessing

The training dataset is split into:

```text
90% training
10% validation
```

using a **stratified split**, preserving approximately the same percentage of samples from each class.

After the split, the mean and standard deviation are calculated from the **training set only**.

The same training statistics are then used to standardize:

- training data
- validation data
- test data

This avoids using validation or test-set statistics during preprocessing.

---

## Training Loop

Each epoch follows the standard neural-network training pipeline:

```text
Training mode
     │
     ├── shuffle training samples
     ├── create mini-batches
     ├── optionally augment each batch
     │
     ├── forward pass
     ├── compute loss
     ├── loss backward pass
     ├── model backward pass
     └── optimizer update

Validation mode
     │
     ├── forward pass
     ├── validation loss
     └── validation accuracy

Early stopping
Learning-rate scheduler
```

The current batch size is:

```text
32
```

A NumPy random generator initialized with seed `42` is used by the main training program for reproducibility.

---

## Installation

Clone the repository and install the required packages.

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

Create and activate a virtual environment if desired, then install:

```bash
pip install numpy torch torchvision pytest
```

or:

```bash
pip install -r requirements.txt
```

`torchvision` is used to provide the datasets. 

---

## Running the Program

Run:

```bash
python project.py
```

The program will ask for the training configuration interactively.

Example flow:

```text
Choose training dataset:
1) MNIST
2) Fashion-MNIST
3) CIFAR-10
4) CIFAR-100

Choose epochs:
Choose weights' initialization for Convolutional Layers (He or Xavier):
Choose weights' distribution for Convolutional Layers (Uniform or Normal):
Choose learning rate:

Choose optimizer:
1) SGD
2) SGD_momentum
3) Adam
4) AdamW

Use data augmentation? (yes/no):
```

If AdamW is selected, the program also asks for:

```text
weight decay
```

During training, each epoch prints values in the form:

```text
Epoch N:
train_acc: ...
train_loss: ...
val_acc: ...
val_loss: ...
learning_rate: ...
```

---

## Example Configuration

A reasonable example for CIFAR-10 is:

```text
Dataset:          CIFAR-10
Initialization:   He
Distribution:     Normal
Optimizer:        AdamW
Learning rate:    0.001
Weight decay:     0.0001
Data augmentation: yes
```

These values are only an example configuration; the best hyperparameters depend on the dataset and training run.

---

## Test-Set Evaluation

The official test split is kept separate from both training and model selection.

Hyperparameters, architecture choices, and regularization settings are selected using only the training and validation sets. After training is complete, Early Stopping restores the weights from the epoch with the best validation loss.

The restored model is then evaluated once on the held-out test set to provide a final estimate of its performance on unseen data.
## Testing

The project uses `pytest`.

Run:

```bash
pytest test_project.py
```

The current test suite checks:

- mini-batch creation
- stratified train/validation splitting
- optimizer construction
- BatchNorm backward gradients
- Dense backward gradients
- Conv2D backward gradients

For the differentiable layers, analytical gradients from the manually implemented backward passes are compared against **numerical finite-difference gradients**.

For a parameter \(x\), the numerical approximation I used is:

```text
dL/dx ≈ [L(x + ε) - L(x - ε)] / (2ε)
```

The tests use `float64` during gradient checking for greater numerical precision and compare the analytical and numerical results with NumPy's `assert_allclose`.

This is especially useful in a from-scratch neural-network implementation because a backward pass can look plausible while still containing subtle indexing, broadcasting, or summation errors.

---

## Design Notes

### Data type

The network uses `np.float32` by default for activations, parameters, and gradients.

Using `float32` reduces memory usage compared with `float64` and is sufficient for normal neural-network training. The dtype can still be changed when constructing the layers.

For numerical gradient checking, the tests explicitly use `float64` to reduce floating-point error and allow more accurate comparison between analytical and numerical gradients.

### Tensor layout

Convolutional tensors use the `(B, H, W, C)` layout throughout the framework, where the channel dimension is last.

Keeping a consistent representation across datasets and layers simplifies shape propagation and broadcasting, particularly in BatchNorm.

### No autograd

Every backward method is written explicitly. Gradients are stored directly inside the corresponding layer.

### Shape propagation

Layers contain a `build()` method. `Sequential.build()` passes each layer's output shape to the next layer so parameters can be initialized with the correct shape before training.

### Vectorization and memory trade-off

The convolution and pooling implementations favor NumPy vectorization over deeply nested Python loops.

For convolution, input windows are exposed using `sliding_window_view` and reshaped so that most of the computation can be expressed as matrix multiplication. This significantly reduces Python-level looping, at the cost of additional intermediate arrays and memory usage during forward and backward propagation.

### In-place parameter updates

Optimizers update the existing NumPy arrays in place. This keeps the parameter references collected by the optimizer valid throughout training.

### Training and evaluation modes

`model.train()` and `model.eval()` propagate the mode to all layers.

This matters particularly for:

- BatchNorm
- Dropout

### Weight-decay selection

Layers explicitly declare which parameters should receive weight decay. This keeps decay away from parameters such as BatchNorm scale/shift values and biases.

---

## Educational Focus

The project is intentionally more explicit than a production deep-learning library.

Many operations that frameworks normally perform automatically are visible here, including:

- tensor-shape management
- parameter initialization
- gradient storage
- forward caches
- backpropagation
- optimizer state
- train/eval behavior
- regularization
- numerical gradient validation

The purpose is not to replace optimized libraries, but to build a concrete understanding of the mechanics behind them.

---

## Possible Future Improvements

Potential extensions include:

- model checkpoint files
- saving/loading trained models from disk
- additional activation functions
- additional pooling layers
- more data-augmentation techniques
- configurable model architectures
- configurable batch size
- additional learning-rate schedulers
- more extensive unit and gradient tests
- training-history plots
- automatic test-set evaluation after final model selection
- command-line arguments instead of interactive prompts
- performance comparisons with an equivalent PyTorch or Keras implementation

---

## Acknowledgements

This project relies on:

- **NumPy** for numerical computation and vectorized array operations
- **torchvision** for downloading and loading MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100
- **pytest** for automated testing

All CNN layers, backward passes, loss computation, optimizers, training utilities, regularization logic, and model orchestration in this repository are implemented directly in the project code.
