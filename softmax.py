import numpy as np

class Softmax:
    def __init__(self):
        self.dtype = np.float32
        self.probabilities = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim != 2:
            raise ValueError("Softmax input shape must be (B,M) where M are the previous Dense layer's neurons")
        
        input = np.asarray(input, dtype= self.dtype)

        # Instead of the usual:
        # softmax = e^logit / Σe^logit,  
        # I'll use: 
        # softmax = e^(logit - max(input)) / Σe^(logit - max(input)), 
        # which is equivalent since e^max(input) can be moved out of the sum so:
        # softmax = (e^logit / e^max(input)) / (Σe^logit / (e^max(input)) = e^logit / Σe^logit. 
        # This way, if a logit is too big, we avoid e^too_big (overflow)
        exp_array = np.exp(input - np.max(input, axis=1)[:, np.newaxis])    # (B,M) 
        exp_sum = np.sum(exp_array, axis=1)[:, np.newaxis]     # (B,1)

        self.probabilities = exp_array / exp_sum    # Broadcasting (B,M) / (B,1) -> (B,M)

        return self.probabilities

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, dtype=self.dtype)

        if self.probabilities is None:
            raise ValueError("Softmax: Attribute probabilities isn't initialized. Forward method needs to be called")

        if dout.shape != self.probabilities.shape:
            raise ValueError("Softmax: Upstream gradient (dout) must have same shape as forward's output")

        dL_dz = self.probabilities * (dout - np.sum(dout * self.probabilities, axis=1)[:, np.newaxis])
        din = dL_dz

        return din


def main():
    x = np.array([12, 11, 9])
    layer = Softmax()
    print(layer.forward(x))




if __name__ == "__main__":
    main()