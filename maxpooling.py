import numpy as np

class MaxPooling2D_layer:
    def __init__(self, pool_size=2, stride=2)
        self.pool_size = pool_size
        self.stride = stride
        self.dtype = np.float32

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        for i in range(input):
