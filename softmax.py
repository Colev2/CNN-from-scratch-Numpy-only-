import numpy as np

class Softmax:
    def __init__(self):
        self.dtype = np.float32
        self.probabilities = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype= self.dtype)

        # Instead of the usual:
        # softmax = e^logit / Σe^logit,  
        # I'll use: 
        # softmax = e^(logit - max(input)) / Σe^(logit - max(input)), 
        # which is equivalent since e^max(input) can be moved out of the sum so:
        # softmax = (e^logit / e^max(input)) / (Σe^logit / (e^max(input)) = e^logit / Σe^logit. 
        # This way, if a logit is too big, we avoid e^too_big (overflow)
        exp_array = np.exp(input - np.max(input))
        exp_sum = np.sum(exp_array)

        self.probabilities = exp_array / exp_sum

        return self.probabilities

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, dtype=self.dtype)

        if self.probabilities is None:
            raise ValueError("Attribute probabilities isn't initialized. Softmax's forward method needs to be called")

        if dout.shape != self.probabilities.shape:
            raise ValueError("Upstream gradient (dout) must have same shape as Softmax's output")

        inner_prod = np.inner(dout, self.probabilities)

        dL_dz = self.probabilities * (dout - inner_prod)
        din = dL_dz

        return din


def main():
    x = np.array([12, 11, 9])
    layer = Softmax()
    print(layer.forward(x))




if __name__ == "__main__":
    main()