import numpy as np

class BatchNorm:
    def __init__(self, epsilon, momentum):
        if epsilon <= 0:
            raise ValueError("BatchNorm: Epsilon argument must be greater than 0")
        if not 0 <= momentum < 1:
            raise ValueError("BatchNorm: Momentum argument must be between 0 (inclusive) and 1 (exclusive)")
        
        self.epsilon = epsilon
        self.momentum = momentum
        self.dtype = np.float32

        self.built = False
        self.training = True

    def _initialize_parameters(self):
        self.gamma = np.ones(self.num_features, dtype=self.dtype)
        self.beta = np.zeros(self.num_features, dtype=self.dtype)

        self.dgamma = np.zeros_like(self.gamma, dtype=self.dtype)
        self.dbeta = np.zeros_like(self.beta, dtype=self.dtype)

        self.running_mean = np.zeros(self.num_features, dtype=self.dtype)
        self.running_variance = np.ones(self.num_features, dtype=self.dtype)


    def build(self, input_shape):
        if self.built:
            raise ValueError("BatchNorm: Layer is already built")
        
        if len(input_shape) != 3 and len(input_shape) != 1:
            raise ValueError("BatchNorm: Build method expects input shape (H,W,C) or (D,)")

        self.num_features = input_shape[-1]
        self.built_shape = input_shape

        self._initialize_parameters()

        self.built = True

        return input_shape
    

    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=self.dtype)

        if input.ndim != len(self.built_shape) + 1:
            raise ValueError("BatchNorm: Forward expects input shape (B,H,W,C) or (B,D) matching the built shape")

        if input.shape[-1] != self.num_features:
            raise ValueError(f"BatchNorm: Layer was built with {self.num_features} number of features but input has {input.shape[-1]}")

        self.reduction_axis = tuple(range(input.ndim - 1))
        self.N = np.prod(input.shape[:-1])

        if self.training:
            self.batch_mean_c = np.sum(input, axis=self.reduction_axis) / self.N    # (C,) 
            # If prev layer is conv: Broadcast: (B,H,W,C) - (C,) -> (B,H,W,C) -> sum over b,h,w -> (C,)
            # if dense: Broadcast: (B,D) - (D,) -> (B,D) -> sum over b -> (D,)
            self.batch_variance_c = np.sum((input - self.batch_mean_c) ** 2, axis=self.reduction_axis) / self.N  
            batch_std_c = np.sqrt(self.batch_variance_c + self.epsilon)
            self.batch_invstd_c = 1 / batch_std_c
            self.batch_x_cent = input - self.batch_mean_c

            self.norm_inp = self.batch_x_cent / batch_std_c   # (B,H,W,C)

            self.running_mean[...] = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean_c
            self.running_variance[...] = self.momentum * self.running_variance + (1 - self.momentum) * self.batch_variance_c

        else:
            self.norm_inp = (input - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)


        output = self.gamma * self.norm_inp + self.beta      # Broadcast: (C,) * (B,H,W,C) + (C,) -> (B,H,W,C) + (C,) -> (B,H,W,C)

        return output


    def backward(self, dout: np.ndarray) -> np.ndarray:
        if dout.shape != self.norm_inp.shape:
            raise ValueError("BatchNorm: Dout shape does not much forward's output shape")
        
        # dL/dβ
        self.dbeta[...] = np.sum(dout, axis=self.reduction_axis)

        # dL/dγ
        self.dgamma[...] = np.sum(dout * self.norm_inp, axis=self.reduction_axis)

        # dL/dx_hat
        dL_dx_hat = dout * self.gamma  # Broadcast: (B,H,W,C) * (C,) -> (B,H,W,C)

        # dL/dx_centered
        dL_dx_cent_direct = dL_dx_hat * self.batch_invstd_c
        dL_dinvstd = np.sum(dL_dx_hat * self.batch_x_cent, axis=self.reduction_axis)
        dL_dvar = (-0.5) * dL_dinvstd * self.batch_invstd_c ** 3
        dL_dx_cent_var = dL_dvar * 2 * self.batch_x_cent / self.N
        dL_dx_cent = dL_dx_cent_direct + dL_dx_cent_var

        # dL_dx
        dL_dmean = -np.sum(dL_dx_cent, axis=self.reduction_axis)
        dL_dx_mean = dL_dmean / self.N
        dL_dx_direct = dL_dx_cent

        dL_dx = dL_dx_direct + dL_dx_mean

        return dL_dx
    

    def parameters(self):
        parameters = [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

        return parameters


    def get_weights(self):

        return [self.gamma.copy(), self.beta.copy(), self.running_mean.copy(), self.running_variance.copy()]


    def set_weights(self, weights: list):
        self.gamma[...] = weights[0]
        self.beta[...] = weights[1]
        self.running_mean[...] = weights[2]
        self.running_variance[...] = weights[3]


    def train(self):
        self.training = True


    def eval(self):
        self.training = False


    def regularizable_parameters(self):
        return []



