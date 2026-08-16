import numpy as np

class SGD:
    def __init__(self, learning_rate):
        if learning_rate <= 0:
            raise ValueError("SGD optimizer: Learning rate must be greater than 0")
        
        self.learning_rate = learning_rate

    def update(self, parameters: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for parameter, gradient in parameters:
            parameter -= self.learning_rate * gradient


class SGD_momentum:
    def __init__(self, learning_rate, momentum_coeff):
        if learning_rate <= 0:
            raise ValueError("SGD momentum optimizer: Learning rate must be greater than 0")

        if not 0 <= momentum_coeff < 1:
            raise ValueError("SGD momentum optimizer: Momentum coefficient must be between 0 (inclusive) and 1 (exclusive)")

        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.state = {}

    def update(self, parameters: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for parameter, gradient in parameters:
            if id(parameter) not in self.state:
                self.state[id(parameter)] = np.zeros_like(parameter)

            u = self.momentum_coeff * self.state[id(parameter)] + gradient
            self.state[id(parameter)] = u

            parameter -= self.learning_rate * u
            

class Adam:
    def __init__(self, learning_rate, b1, b2, epsilon):
        if learning_rate <= 0:
            raise ValueError("Adam optimizer: Learning rate must be greater than 0")

        if not 0 <= b1 < 1 or not 0 <= b2 < 1:
            raise ValueError("Adam optimizer: b1,b2 parameters must be between 0 (inclusive) and 1 (exclusive)")

        if epsilon <= 0:
            raise ValueError("Adam optimizer: epsilon parameter must be greater than 0")

        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.t = 0
        self.state = {}

    def update(self, parameters: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.t += 1  
        for parameter, gradient in parameters:
            if id(parameter) not in self.state:
                self.state[id(parameter)] = {"m": np.zeros_like(parameter), "u": np.zeros_like(parameter)}
            m = self.b1 * self.state[id(parameter)]["m"] + (1 - self.b1) * gradient
            u = self.b2 * self.state[id(parameter)]["u"] + (1 - self.b2) * (gradient ** 2)

            self.state[id(parameter)]["m"] = m
            self.state[id(parameter)]["u"] = u

            m_corrected = m / (1 - self.b1 ** self.t)
            u_corrected = u / (1 - self.b2 ** self.t)

            parameter -= self.learning_rate * m_corrected / (np.sqrt(u_corrected) + self.epsilon)

def main():
    parameter = [(np.array([1.0, 2.0]), np.array([0.5, -0.25]))]
    optimizer = Adam(learning_rate=1.0, b1=0.9, b2=0.999, epsilon=0.01)
    optimizer.update(parameter)
    m = optimizer.state[id(parameter[0][0])]['m']
    u = optimizer.state[id(parameter[0][0])]['u']
    m_corrected = m / (1 - optimizer.b1 ** optimizer.t)
    u_corrected = u / (1 - optimizer.b2 ** optimizer.t)
    print(f"m: {m}")
    print(f"u: {u}")
    print(f"m_corrected: {m_corrected}")
    print(f"u_corrected: {u_corrected}")
    print(f"parameter: {parameter[0][0]}")


if __name__ == "__main__":
    main()