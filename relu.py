import numpy as np

class ReLU_layer:
    def __init__(self):
        self.positive_input_mask = None
        self.dtype = np.float32

    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim != 3:
            raise ValueError("Input in ReLU layer must be 3-dimensional")
        
        input = np.asarray(input, dtype=self.dtype)
        output = np.maximum(0, input)
        self.positive_input_mask = input > 0

        return output   

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, self.dtype)
        if self.positive_input_mask is None:
            raise ValueError("ReLU forward method must be called first")
        din = dout * self.positive_input_mask

        return din


def main():
    layer = ReLU_layer()
    input = np.array([[-2, 3], [0, 5]])
    dout = np.array([[7, -4], [2, 6]])
    layer.forward(input)
    print(layer.backward(dout))



if __name__ == "__main__":
    main()