import numpy as np

class Dense_layer:
    def __init__(self, in_features, neurons=1, rng=None, initialization="he", distribution="normal"):
        if in_features < 1:
            raise ValueError("Input features must be 1 or more")
        
        if neurons < 1:
            raise ValueError("Neurons must be 1 or more")
        
        if rng is None:
            rng = np.random.default_rng()

        if initialization not in ["he", "xavier"]:
            raise ValueError("initialization argument must be either 'he' or 'xavier'")

        if distribution not in ["normal", "uniform"]:
            raise ValueError("distribution argument must be either 'normal' or 'uniform'")

        self.input = None
        self.output_shape = None
        self.dweights = None
        self.dbias = None

        self.in_features = in_features
        self.dtype = np.float32

        fan_in = in_features
        fan_out = neurons

        # Weights: He initialization
        if initialization == "he":
            if distribution == "normal":
                std = (2 / fan_in) ** 0.5
                self.weights = rng.normal(loc=0, scale=std, size=(neurons, in_features))   
                self.weights = np.asarray(self.weights, dtype=self.dtype)
            elif distribution == "uniform":
                limit = (6 / fan_in) ** 0.5
                self.weights = rng.uniform(low=-limit, high=limit, size=(neurons, in_features))
                self.weights = np.asarray(self.weights, dtype=self.dtype)

        # Weights: Xavier initialization
        elif initialization == "xavier":
            if distribution == "normal":
                std = (2 / (fan_in + fan_out)) ** 0.5
                self.weights = rng.normal(loc=0, scale=std, size=(neurons, in_features))
                self.weights = np.asarray(self.weights, dtype=self.dtype)
            elif distribution == "uniform":
                limit = (6 / (fan_in + fan_out)) ** 0.5
                self.weights = rng.uniform(low=-limit, high=limit, size=(neurons, in_features))
                self.weights = np.asarray(self.weights, dtype=self.dtype)

        self.bias = np.zeros((neurons), dtype=self.dtype)


    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=self.dtype)
        
        if input.shape != (input.shape[0], self.in_features,):
            raise ValueError("Input shape must be (B,input_features). Flatten layer needs to be called")
        
        self.input = input

        output = self.input @ self.weights.T + self.bias    # (B,features) @ (features,neurons) + (neurons,) -> (B,neurons)
        self.output_shape = output.shape

        return output


    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, dtype=self.dtype)

        if self.input is None:
            raise ValueError("Dense's forward method needs to be called to get input")

        if self.output_shape is None:
            raise ValueError("Dense's forward method needs to be called to get output shape")
        
        if dout.shape != self.output_shape:
            raise ValueError("Dout shape must be same as Dense's forward's output's shape")

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