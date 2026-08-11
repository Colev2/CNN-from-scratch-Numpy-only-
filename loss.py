# Cross-Entropy Loss

import numpy as np

class CrossEntropy:
    def __init__(self):
        self.correct_class_probs = None
        self.dtype = np.float32


    def forward(self, input: np.ndarray, labels: np.ndarray) -> float:
        self.labels = np.asarray(labels)
        input = np.asarray(input, dtype=self.dtype)

        if input.ndim != 2:
            raise ValueError("Cross Entropy: Input shape needs to be (B,M)")

        if self.labels.ndim != 1:
            raise ValueError("Cross Entropy: Batch labels shape needs to be (B,)")

        if input.shape[0] != self.labels.shape[0]:
            raise ValueError("Cross Entropy: Input and labels must have same rows (B)")

        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("All labels must be integers")

        low = self.labels < 0
        high = self.labels >= input.shape[1]

        if np.any(low) or np.any(high):
            raise ValueError("Labels must be: 0 <= y < M")

        self.batch_size = self.labels.shape[0]
        self.input_shape = input.shape

        batch_indexes = np.arange(self.batch_size)
        self.correct_class_probs = input[batch_indexes, self.labels]   # (B,)

        loss = -1 * np.log(self.correct_class_probs)     # (B,) : -log(p[b,labels[b]])
        batch_loss = np.mean(loss, dtype=np.float64)

        return batch_loss

    def backward(self) -> np.ndarray:
        if self.correct_class_probs is None:
            raise ValueError("Cross Entropy: correct_class_probs attribute wasn't initialized. Forward method needs to be called")
        
        gradients = np.zeros(self.input_shape, dtype=self.dtype)

        batch_indexes = np.arange(self.batch_size)

        gradients[batch_indexes, self.labels] = -1 / (self.batch_size * self.correct_class_probs)

        return gradients



