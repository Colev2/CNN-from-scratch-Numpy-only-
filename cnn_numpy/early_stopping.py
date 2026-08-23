import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        if not isinstance(patience, int):
            raise ValueError("Early Stopping: Patience must be integer")
        if patience < 0:
            raise ValueError("Early Stopping: Patience must be non-negative")

        if not isinstance(min_delta, float):
            raise ValueError("Early Stopping: Minimum delta must be float")
        if min_delta < 0:
            raise ValueError("Early Stopping: Minimum delta must be non-negative")
        
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.bad_epochs = 0
        self.best_epoch = None
        self.best_weights = None



    def step(self, model: object, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_val_loss - self.min_delta:
            self.bad_epochs = 0
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_weights = model.get_weights()
        else:
            self.bad_epochs += 1

        stop = False
        if self.bad_epochs > self.patience:
            stop = True

        return stop


    def restore_best_weights(self, model: object):
        model.set_weights(self.best_weights)

        print(f"Best weights are at epoch: {self.best_epoch + 1}")

