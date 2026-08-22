import numpy as np

class Dropout:
    def __init__(self, drop_prob=0.2, rng=None):
        if not 0 <= drop_prob <= 1:
            raise ValueError("Dropout: Keep probability must be between 0 (exclusive) and 1 (inclusive)")
        
        self.drop_prob = drop_prob

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
        self.built = False
        self.training = True
        self.mask = None
        self.dtype = np.float32


    def build(self, input_shape):
        if len(input_shape) < 1:
            raise ValueError("Dropout: Build expects a non-empty input shape")

        for dim in input_shape:
            if dim <= 0:
                raise ValueError("Dropout: Input dimensions must be greater than 0")

        self.built_shape = input_shape

        self.built = True

        return input_shape


    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, self.dtype)

        if self.training:
            self.mask = self.rng.random(input.shape) < 1 - self.drop_prob
            output = input * self.mask / (1 - self.drop_prob)  
            return output
        else:
            self.mask = None
            return input



    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dout
        else:
            din = dout * self.mask / (1 - self.drop_prob)
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



