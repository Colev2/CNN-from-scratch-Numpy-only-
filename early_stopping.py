import numpy as np

class EarlyStopping:
    def __init__(self, patience: int, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.counter = 0
        self.best_epoch = None
        self.best_parameters = []


    def step(self, val_loss: float, current_parameters: list[np.ndarray], epoch: int) -> tuple[bool, list[np.ndarray]]:
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_parameters = []
            for parameter in current_parameters:
                self.best_parameters.append(parameter.copy())
        else:
            self.counter += 1

        stop = False
        if self.counter >= self.patience:
            stop = True

        return stop, self.best_parameters

