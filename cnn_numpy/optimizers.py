import numpy as np

class SGD:
    def __init__(self, parameters_grads: list[tuple[np.ndarray, np.ndarray]], learning_rate):
        if learning_rate <= 0:
            raise ValueError("SGD: Learning rate must be greater than 0")

        self.parameters_grads = parameters_grads

        self.learning_rate = learning_rate

    def update(self) -> None:
        for parameter, gradient in self.parameters_grads:
            parameter[...] -= self.learning_rate * gradient


class SGD_momentum:
    def __init__(self, parameters_grads: list[tuple[np.ndarray, np.ndarray]], learning_rate, momentum_coeff):
        if learning_rate <= 0:
            raise ValueError("SGD momentum: Learning rate must be greater than 0")

        if not 0 <= momentum_coeff < 1:
            raise ValueError("SGD momentum: Momentum coefficient must be between 0 (inclusive) and 1 (exclusive)")

        self.parameters_grads = parameters_grads

        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.state = {}

    def update(self) -> None:
        for parameter, gradient in self.parameters_grads:
            if id(parameter) not in self.state:
                self.state[id(parameter)] = np.zeros_like(parameter)

            u = self.momentum_coeff * self.state[id(parameter)] + gradient
            self.state[id(parameter)] = u

            parameter[...] -= self.learning_rate * u
            

class Adam:
    def __init__(self, parameters_grads: list[tuple[np.ndarray, np.ndarray]], learning_rate, b1, b2, epsilon):
        if learning_rate <= 0:
            raise ValueError("Adam: Learning rate must be greater than 0")

        if not 0 <= b1 < 1 or not 0 <= b2 < 1:
            raise ValueError("Adam: b1,b2 parameters must be between 0 (inclusive) and 1 (exclusive)")

        if epsilon <= 0:
            raise ValueError("Adam: epsilon parameter must be greater than 0")

        self.parameters_grads = parameters_grads

        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon

        self.t = 0
        self.state = {}


    def update(self) -> None:
        self.t += 1  

        for parameter, gradient in self.parameters_grads:
            if id(parameter) not in self.state:
                self.state[id(parameter)] = {"m": np.zeros_like(parameter), "u": np.zeros_like(parameter)}
            m = self.b1 * self.state[id(parameter)]["m"] + (1 - self.b1) * gradient
            u = self.b2 * self.state[id(parameter)]["u"] + (1 - self.b2) * (gradient ** 2)

            self.state[id(parameter)]["m"] = m
            self.state[id(parameter)]["u"] = u

            m_corrected = m / (1 - self.b1 ** self.t)
            u_corrected = u / (1 - self.b2 ** self.t)
            # In-place update
            parameter[...] -= self.learning_rate * m_corrected / (np.sqrt(u_corrected) + self.epsilon)


class AdamW:
    def __init__(self, parameters_grads: list[tuple[np.ndarray, np.ndarray]], decayable_parameters: list, learning_rate, b1, b2, epsilon, weight_decay):
        if learning_rate <= 0:
            raise ValueError("AdamW: Learning rate must be greater than 0")

        if not 0 <= b1 < 1 or not 0 <= b2 < 1:
            raise ValueError("AdamW: b1,b2 parameters must be between 0 (inclusive) and 1 (exclusive)")

        if epsilon <= 0:
            raise ValueError("AdamW: epsilon parameter must be greater than 0")

        if weight_decay < 0:
            raise ValueError("AdamW: Weight decay must be non-negative")

        self.parameters_grads = parameters_grads
        self.decayable_ids = {id(p) for p in decayable_parameters}

        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.t = 0
        self.state = {}


    def update(self) -> None:
        self.t += 1  

        for parameter, gradient in self.parameters_grads:
            if id(parameter) not in self.state:     # https://docs.python.org/3/library/functions.html#id
                self.state[id(parameter)] = {"m": np.zeros_like(parameter), "u": np.zeros_like(parameter)}
            m = self.b1 * self.state[id(parameter)]["m"] + (1 - self.b1) * gradient
            u = self.b2 * self.state[id(parameter)]["u"] + (1 - self.b2) * (gradient ** 2)

            self.state[id(parameter)]["m"] = m
            self.state[id(parameter)]["u"] = u

            m_corrected = m / (1 - self.b1 ** self.t)
            u_corrected = u / (1 - self.b2 ** self.t)

            adam_step = m_corrected / (np.sqrt(u_corrected) + self.epsilon)

            if id(parameter) in self.decayable_ids:
                parameter[...] *= (1 - self.learning_rate * self.weight_decay)  # θ = (1-λ*η)θ

            parameter[...] -= self.learning_rate * adam_step    # θ = θ - λ*η*θ - η * Adam_step | θ = θ - η * Adam_step