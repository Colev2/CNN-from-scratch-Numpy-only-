import numpy as np

class ReLU:
    def __init__(self, dtype=np.float32):
        self.positive_input_mask = None
        self.built = False
        self.training = True
        self.dtype = np.dtype(dtype)


    def build(self, input_shape):
        if len(input_shape) < 1:
            raise ValueError("ReLU: Build expects a non-empty input shape")

        for dim in input_shape:
            if dim <= 0:
                raise ValueError("ReLU: Input dimensions must be greater than 0")

        self.built_shape = input_shape

        self.built = True

        return input_shape


    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=self.dtype)

        if self.built_shape != input.shape[1:]:
            raise ValueError("ReLU: Layer was built with different shape than forward's input")

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


    def parameters_grads(self):

        return []


    def get_weights(self):

        return []


    def set_weights(self, weights):
        pass


    def train(self):
        self.training = True


    def eval(self):
        self.training = False


    def decayable_parameters(self):
        return []


