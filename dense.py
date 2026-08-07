import numpy as np

class Dense_layer:
    def __init__(self, in_features, out_features, neurons=1, rng=None, initialization="he", distribution="normal"):
        if neurons < 1:
            raise ValueError("Neurons must be 1 or more")
        
        if rng is None:
            rng = np.random.default_rng()

        if initialization not in ["he", "xavier"]:
            raise ValueError("initialization argument must be either 'he' or 'xavier'")

        if distribution not in ["normal", "uniform"]:
            raise ValueError("distribution argument must be either 'normal' or 'uniform'")

        self.in_features = in_features
        self.out_features = out_features
        self.dtype = np.float32

        fan_in = in_features
        fan_out = out_features

        # Weights: He initialization
        if initialization == "he":
            if distribution == "normal":
                std = (2 / fan_in) ** 0.5
                self.filter_weights = rng.normal(loc=0, scale=std, size=(neurons, in_features))   # (F,Kh,Kw,Cin)
                self.filter_weights = np.asarray(self.filter_weights, dtype=self.dtype)
            elif distribution == "uniform":
                limit = (6 / fan_in) ** 0.5
                self.filter_weights = rng.uniform(low=-limit, high=limit, size=(neurons, in_features))
                self.filter_weights = np.asarray(self.filter_weights, dtype=self.dtype)

        # Weights: Xavier initialization
        elif initialization == "xavier":
            if distribution == "normal":
                std = (2 / (fan_in + fan_out)) ** 0.5
                self.filter_weights = rng.normal(loc=0, scale=std, size=(neurons, in_features))
                self.filter_weights = np.asarray(self.filter_weights, dtype=self.dtype)
            elif distribution == "uniform":
                limit = (6 / (fan_in + fan_out)) ** 0.5
                self.filter_weights = rng.uniform(low=-limit, high=limit, size=(neurons, in_features))
                self.filter_weights = np.asarray(self.filter_weights, dtype=self.dtype)

        self.bias = np.zeros((neurons), dtype=self.dtype)


        def forward(self, input: np.ndarray) -> np.ndarray:
            input = np.asarray(input, dtype=self.dtype)
            self.input_shape = input.shape

            if self.input_shape != self.in_features:
                raise ValueError("input shape must be same as in_features argument")

            output = self.filter_weights @ input + self.bias
            self.output_shape = output.shape

            return output


        def backward(self, dout: np.ndarray) -> np.ndarray:
            if dout.shape != self.output_shape:
                raise ValueError("Dout shape must be same as Dense's forward's output's shape")

            return din