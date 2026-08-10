import numpy as np

class Flatten_layer:
    def __init__(self):
        self.dtype = np.float32
        self.input_shape = None
        self.output_shape = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim != 4:
            raise ValueError("Flatten layer input must have shape (B,H,W,C)")
        
        input = np.asarray(input, self.dtype)
        
        self.input_shape = input.shape

        output = np.reshape(input, shape=(input.shape[0], -1))

        self.output_shape = output.shape

        return output



    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise ValueError("Attribute input_shape wasn't initialized. Flatten_layer's forward method needs to be called.")

        if self.output_shape is None:
            raise ValueError("Attribute output_shape wasn't initialized. Flatten_layer's forward method needs to be called.")

        if dout.shape != self.output_shape:
            raise ValueError("Dout shape must be same as Flatten_layer's forward's output shape")

        din = np.reshape(dout, shape=self.input_shape)

        return din
        