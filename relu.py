import numpy as np

class ReLU_layer:
    def __init__(self):
        self.positive_input_mask = None
        self.dtype = np.float32

    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=self.dtype)

        output = np.maximum(0, input)

        # Boolean ndarray [False, True,...] whose elements are True where condition is met, which esentially is input's derivative, with the convention that ReLU'(0) = 0
        self.positive_input_mask = input > 0 

        return output   

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, self.dtype)

        if self.positive_input_mask is None:
            raise ValueError("ReLU: Positive_input_mask attribute isn't initialized. Forward method needs to be called")

        if dout.shape != self.positive_input_mask.shape:
            raise ValueError("ReLU: Dout must have same shape as positive_input_mask")

        din = dout * self.positive_input_mask

        return din


def main():
    layer = ReLU_layer()
    input = np.array([[-2, 3], [0, 5]])
    layer.forward(input)
    print(layer.positive_input_mask)



if __name__ == "__main__":
    main()