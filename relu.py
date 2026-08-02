import numpy as np
from Conv import Conv_layer

class ReLU_layer:
    def __init__(self):
        self.dtype = np.float32

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, input).astype(self.dtype)
        self.positive_input_mask = np.zeros_like(self.output)
        self.positive_input_mask[self.output > 0] = 1
        return self.output

    def backward(self):
        pass


