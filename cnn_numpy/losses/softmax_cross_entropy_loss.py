import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self, dtype=np.float32):
        self.probabilities = None
        self.dtype = np.dtype(dtype)


    def forward(self, input: np.ndarray, labels: np.ndarray) -> float:
        self.labels = np.asarray(labels)
        input = np.asarray(input, dtype=self.dtype)

        if input.ndim != 2:
            raise ValueError("Softmax Cross Entropy: Input shape needs to be (B,M)")

        if self.labels.ndim != 1:
            raise ValueError("Softmax Cross Entropy: Batch labels shape needs to be (B,)")

        if input.shape[0] != self.labels.shape[0]:
            raise ValueError("Softmax Cross Entropy: Input and labels must have same rows (B)")

        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("Each label must be an integer")

        low = self.labels < 0
        high = self.labels >= input.shape[1]

        if np.any(low) or np.any(high):
            raise ValueError("Each label must be: 0 <= y < M")

        self.input_shape = input.shape
        self.batch_size = self.labels.shape[0]
        batch_indexes = np.arange(self.batch_size)

        # Instead of the usual:
        # softmax = e^logit / Σe^logit,  
        # I'll use: 
        # softmax = e^(logit - max(input)) / Σe^(logit - max(input)), 
        # which is equivalent since e^max(input) can be moved out of the sum so:
        # softmax = (e^logit / e^max(input)) / (Σe^logit / (e^max(input)) = e^logit / Σe^logit. 
        # This way, if a logit is too big, we avoid e^too_big (overflow)
        max_logits = np.max(input, axis=1)  # (B,)
        exp_array = np.exp(input - max_logits[:, np.newaxis])    # Broadcasting: (B,M) - (B,1) -> (B,M)
        exp_sum = np.sum(exp_array, axis=1)    # (B,)
        self.probabilities = exp_array / exp_sum[:, np.newaxis]     # Broadcasting: (B,M) / (B,1) -> (B,M)

        correct_class_logits = input[batch_indexes, self.labels]    # (B,)
        loss = -correct_class_logits + max_logits + np.log(exp_sum)     # (B,)
        batch_loss = np.mean(loss, dtype=np.float64)    # scalar

        return batch_loss


    def backward(self) -> np.ndarray:
        if self.probabilities is None:
            raise ValueError("Cross Entropy: probabilities attribute wasn't initialized. Forward method needs to be called")
        
        din = np.empty(self.input_shape, dtype=self.dtype)

        batch_indexes = np.arange(self.batch_size)

        # uL_batch / u(z_bi) = {p_b,i / B   if i != k, 
        #                       (p_b,k - 1) / B   if i = k} = (P-Y) / B, 
        # where k is the correct class of image b, and Y the one hot vector [0 0...1...0 0]
        din[:, :] = self.probabilities[:, :] / self.batch_size
        din[batch_indexes, self.labels] += -1 / self.batch_size

        return din



