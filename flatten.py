import numpy as np

class Flatten_layer:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.built = False
        self.training = True
        self.dtype = np.float32


    def build(self, input_shape):
        if self.built:
            raise RuntimeError("Flatten: Layer has already been built")

        if len(input_shape) != 3:
            raise ValueError("Flatten: Build expects input shape (H,W,C)")

        height, width, channels = input_shape

        if height <= 0 or width <= 0 or channels <= 0:
            raise ValueError("Flatten: Input dimensions must be greater than 0")

        self.built_shape = input_shape

        self.built = True

        return (height * width * channels,)

    
    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim != 4:
            raise ValueError("Flatten: Input must have shape (B,H,W,C)")
        
        input = np.asarray(input, self.dtype)
        self.input_shape = input.shape

        if self.built_shape != self.input_shape[1:]:
            raise ValueError("Flatten: Layer was built with different shape than forward's input")

        output = np.reshape(input, shape=(input.shape[0], -1))      # (B,H*W*C)
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


    def parameters(self):

        return []


    def get_weights(self):

        return []


    def set_weights(self, weights):
        pass


    def train(self):
        self.training = True


    def eval(self):
        self.training = False


    def regularizable_parameters(self):
        return []
        