import numpy as np

class ReLU_layer:
    def __init__(self):
        self.positive_input_mask = None
        self.dtype = np.float32

    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim != 4:
            raise ValueError("Input in ReLU must be of shape (B,H,W,C)")
        
        input = np.asarray(input, dtype=self.dtype)

        output = np.maximum(0, input)
        self.positive_input_mask = input > 0        # Boolean ndarray that is basically the input's derivative, with the convention that ReLU'(0) = 0

        return output   

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, self.dtype)

        if dout.shape != self.positive_input_mask:
            raise ValueError("Dout must have same shape as positive_input_mask in ReLU")

        if self.positive_input_mask is None:
            raise ValueError("Positive_input_mask attribute isn't initialized. ReLU's forward method must be called first")
        
        din = dout * self.positive_input_mask

        return din


def main():
    layer = ReLU_layer()
    input = np.array([[-2, 3], [0, 5]])
    layer.forward(input)
    print(layer.positive_input_mask)



if __name__ == "__main__":
    main()