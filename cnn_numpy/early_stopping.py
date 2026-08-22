import numpy as np

class EarlyStopping:
    def __init__(self, patience: int, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.counter = 0
        self.best_epoch = None
        self.best_weights = None



    def step(self, model: object, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_val_loss - self.min_delta:
            self.counter = 0
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_weights = model.get_weights()
        else:
            self.counter += 1

        stop = False
        if self.counter >= self.patience:
            stop = True

        return stop


    def restore_best_weights(self, model: object):
        model.set_weights(self.best_weights)

        print(f"Best weights are at epoch: {self.best_epoch + 1}")

