import numpy as np

class Dense_layer:
    def __init__(self, neurons=1, rng=None, initialization="he", distribution="normal"):
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
        self.dtype = np.float32

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


    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=self.dtype)

        if len(input.shape) != 2:
            raise ValueError("Dense: Input must have shape (B,features)")

        if input.shape[1] < 1:
            raise ValueError("Dense: Input features must be 1 or more")
        
        if self.in_features is None:
            self.in_features = input.shape[1]
        elif self.in_features != input.shape[1]:
            raise ValueError("Dense: Input features do not match attribute's in_features value")

        if self.weights is None:
            self._initialize_parameters()
        
        self.input = input

        output = self.input @ self.weights.T + self.bias    # (B,features) @ (features,neurons) + (neurons,) -> (B,neurons)
        self.output_shape = output.shape

        return output


    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, dtype=self.dtype)

        if self.output_shape is None:
            raise ValueError("Dense: Forward method needs to be called to get output shape")
        
        if dout.shape != self.output_shape:
            raise ValueError("Dense: Dout shape must be same as Dense's forward's output's shape")

        # z = Wx + b

        # θL/θx_j = Σ_i (θL/θz_i * w_ij) = Σ_i (dout_i * w_ij)
        dL_dx = dout @ self.weights     # (B,neurons) @ (neurons,features)  ->  (B,features)

        # θL/θb_i = θL/θz_i = dout_i
        dL_db = np.sum(dout, axis=0)

        # θL/θw_ij = θL/θz_i * x_j = dout_i * x_j
        dL_dw = dout.T @ self.input

        self.dweights = dL_dw
        self.dbias = dL_db
        din = dL_dx

        return din