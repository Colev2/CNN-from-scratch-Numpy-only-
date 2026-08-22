import numpy as np

class SGD:
    def __init__(self, parameters: list[tuple[np.ndarray, np.ndarray]], regularizable_parameters: list[np.ndarray], learning_rate, l2_lambda=0.0):
        if learning_rate <= 0:
            raise ValueError("SGD optimizer: Learning rate must be greater than 0")

        if l2_lambda < 0:
            raise ValueError("SGD optimizer: l2_lambda parameter must be 0 or greater")

        self.parameters_grads = parameters
        self.regularizable_ids = {id(p) for p in regularizable_parameters}

        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda

    def update(self) -> None:
        for parameter, gradient in self.parameters_grads:
            if id(parameter) in self.regularizable_ids:
                gradient = gradient + 2 * self.l2_lambda * parameter
            parameter[...] -= self.learning_rate * gradient


class SGD_momentum:
    def __init__(self, parameters: list[tuple[np.ndarray, np.ndarray]], regularizable_parameters: list[np.ndarray], learning_rate, momentum_coeff, l2_lambda=0.0):
        if learning_rate <= 0:
            raise ValueError("SGD momentum optimizer: Learning rate must be greater than 0")

        if not 0 <= momentum_coeff < 1:
            raise ValueError("SGD momentum optimizer: Momentum coefficient must be between 0 (inclusive) and 1 (exclusive)")

        if l2_lambda < 0:
            raise ValueError("SGD momentum optimizer: l2_lambda parameter must be 0 or greater")

        self.parameters_grads = parameters
        self.regularizable_ids = {id(p) for p in regularizable_parameters}

        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.l2_lambda = l2_lambda
        self.state = {}

    def update(self) -> None:
        for parameter, gradient in self.parameters_grads:
            if id(parameter) in self.regularizable_ids:
                gradient = gradient + 2 * self.l2_lambda * parameter

            if id(parameter) not in self.state:
                self.state[id(parameter)] = np.zeros_like(parameter)

            u = self.momentum_coeff * self.state[id(parameter)] + gradient
            self.state[id(parameter)] = u

            parameter[...] -= self.learning_rate * u
            

class Adam:
    def __init__(self, parameters: list[tuple[np.ndarray, np.ndarray]], regularizable_parameters: list[np.ndarray], learning_rate, b1, b2, epsilon, l2_lambda=0.0):
        if learning_rate <= 0:
            raise ValueError("Adam optimizer: Learning rate must be greater than 0")

        if not 0 <= b1 < 1 or not 0 <= b2 < 1:
            raise ValueError("Adam optimizer: b1,b2 parameters must be between 0 (inclusive) and 1 (exclusive)")

        if epsilon <= 0:
            raise ValueError("Adam optimizer: epsilon parameter must be greater than 0")

        if l2_lambda < 0:
            raise ValueError("Adam optimizer: l2_lambda parameter must be 0 or greater")

        self.parameters_grads = parameters
        self.regularizable_ids = {id(p) for p in regularizable_parameters}

        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.l2_lambda = l2_lambda

        self.t = 0
        self.state = {}


    def update(self) -> None:
        self.t += 1  

        for parameter, gradient in self.parameters_grads:
            # L2 regularization gradient
            if id(parameter) in self.regularizable_ids:
                gradient = gradient + 2 * self.l2_lambda * parameter

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

