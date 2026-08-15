import numpy as np

class Flatten_layer:
    def __init__(self):
        self.dtype = np.float32
        self.input_shape = None
        self.output_shape = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim != 4:
            raise ValueError("Flatten: Input must have shape (B,H,W,C)")
        
        input = np.asarray(input, self.dtype)
        
        self.input_shape = input.shape

        output = np.reshape(input, shape=(input.shape[0], -1))

        self.output_shape = output.shape

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise ValueError("Flatten: input_shape attribute wasn't initialized. Forward method needs to be called.")

        if self.output_shape is None:
            raise ValueError("Flatten: output_shape attribute wasn't initialized. Forward method needs to be called.")

        if dout.shape != self.output_shape:
            raise ValueError("Flatten: Dout shape must be same as forward's output shape")

        din = np.reshape(dout, shape=self.input_shape)

        return din
        