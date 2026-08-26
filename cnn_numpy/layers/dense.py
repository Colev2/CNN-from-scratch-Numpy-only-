import numpy as np

class Dense:
    def __init__(self, neurons=1, rng=None, initialization="he", distribution="normal", dtype=np.float32):
        if neurons < 1:
            raise ValueError("Dense: Neurons must be 1 or more")
        
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(initialization, str):
            raise ValueError("Dense: Initialization must be string")
        if initialization.strip().lower() not in ["he", "xavier"]:
            raise ValueError("Dense: initialization argument must be either 'he' or 'xavier'")

        if not isinstance(distribution, str):
            raise ValueError("Dense: Distribution must be string")
        if distribution.strip().lower() not in ["normal", "uniform"]:
            raise ValueError("Dense: distribution argument must be either 'normal' or 'uniform'")

        self.neurons = neurons
        self.rng = rng
        self.initialization = initialization.strip().lower()
        self.distribution = distribution
        self.output_shape = None
        self.in_features = None
        self.weights = None
        self.built = False
        self.training = True
        self.dtype = np.dtype(dtype)


    def _initialize_parameters(self):
        fan_in = self.in_features
        fan_out = self.neurons

        # Weights: He initialization
        if self.initialization == "he":
            if self.distribution == "normal":
                std = (2 / fan_in) ** 0.5
                self.weights = self.rng.normal(loc=0, scale=std, size=(self.neurons, self.in_features))   
                self.weights = np.asarray(self.weights, dtype=self.dtype)
            elif self.distribution == "uniform":
                limit = (6 / fan_in) ** 0.5
                self.weights = self.rng.uniform(low=-limit, high=limit, size=(self.neurons, self.in_features))
                self.weights = np.asarray(self.weights, dtype=self.dtype)

        # Weights: Xavier initialization
        elif self.initialization == "xavier":
            if self.distribution == "normal":
                std = (2 / (fan_in + fan_out)) ** 0.5
                self.weights = self.rng.normal(loc=0, scale=std, size=(self.neurons, self.in_features))
                self.weights = np.asarray(self.weights, dtype=self.dtype)
            elif self.distribution == "uniform":
                limit = (6 / (fan_in + fan_out)) ** 0.5
                self.weights = self.rng.uniform(low=-limit, high=limit, size=(self.neurons, self.in_features))
                self.weights = np.asarray(self.weights, dtype=self.dtype)

        self.bias = np.zeros((self.neurons), dtype=self.dtype)

        # Gradients
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)


    def build(self, input_shape: tuple):
        # input_shape = (features,)

        if len(input_shape) != 1:
            raise ValueError("Dense: Build expects input shape (features,)")

        if input_shape[0] < 1:
            raise ValueError("Dense: Build features must be greater than 0")

        self.in_features = input_shape[0]

        self._initialize_parameters()

        self.built = True

        return (self.neurons,)


    def forward(self, input: np.ndarray) -> np.ndarray:
        
        input = np.asarray(input, dtype=self.dtype)

        if input.ndim != 2:
            raise ValueError("Dense: Input must have shape (B,features)")
        
        if self.in_features != input.shape[1]:
            raise ValueError("Dense: Input features do not match attribute's in_features value")
        
        self.input = input

        # X: (B, D)
        # W: (N, D)
        # Z = X @ W.T + b    -> (B, N)
        output = self.input @ self.weights.T + self.bias    # (B,features) @ (features,neurons) + (neurons,) -> (B,neurons)
        self.output_shape = output.shape

        return output


    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, dtype=self.dtype)

        if self.output_shape is None:
            raise ValueError("Dense: Forward method needs to be called to get output shape")
        
        if dout.shape != self.output_shape:
            raise ValueError("Dense: Dout shape must be same as Dense's forward's output's shape")

        # X: (B, D)
        # W: (N, D)
        # Z = X @ W.T + b    -> (B, N)

        # θL/θx_b,d = Σ_n (θL/θz_b,n * w_n,d) = Σ_n (dout_b,n * w_n,d) = dout @ w
        din = dout @ self.weights     # (B,neurons) @ (neurons,features)  ->  (B,features)

        # θL/θb_n = Σ_b (θL/θz_b,d) = Σ_b (dout_b,d)
        self.dbias[...] = np.sum(dout, axis=0)

        # θL/θw_n,d = Σ_b (θL/θz_b,n * x_b,d) = Σ_b (dout_b,n * x_b,d) = dout.T @ x
        self.dweights[...] = dout.T @ self.input

        return din

    # Weights + Gradients
    def parameters_grads(self):
        parameters_grads = [(self.weights, self.dweights), (self.bias, self.dbias)]

        return parameters_grads


    def get_weights(self):

        return [self.weights.copy(), self.bias.copy()]


    def set_weights(self, weights:list):
        self.weights[...] = weights[0]
        self.bias[...] = weights[1]


    def train(self):
        self.training = True


    def eval(self):
        self.training = False


    def decayable_parameters(self):
        return [self.weights]