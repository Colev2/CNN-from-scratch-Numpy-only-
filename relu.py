import numpy as np

class ReLU_layer:
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self):
        pass

