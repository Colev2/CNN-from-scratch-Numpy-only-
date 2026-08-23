import numpy as np

class ReduceLROnPlateau:
    def __init__(self, optimizer: object, factor=0.5, patience=5, min_delta=0.001, min_lr=1e-5):
        if not isinstance(factor, float):
            raise ValueError("LR scheduler: LR reduce factor must be float")
        if not 0 < factor < 1:
            raise ValueError("LR scheduler: LR reduce factor must be between 0 and 1 exclusively")

        if not isinstance(patience, int):
            raise ValueError("LR scheduler: Patience must be integer")
        if patience < 0:
            raise ValueError("LR scheduler: Patience must be non-negative")

        if not isinstance(min_delta, float):
            raise ValueError("LR scheduler: Minimum delta must be float")
        if min_delta < 0:
            raise ValueError("LR scheduler: Minimum delta must be non-negative")

        if not isinstance(min_lr, float):
            raise ValueError("LR scheduler: Minimum learning rate must be float")
        if not 0 < min_lr < optimizer.learning_rate:
            raise ValueError("LR scheduler: Minimum learning rate must be greater than 0")

        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr

        self.best_val_loss = np.inf
        self.bad_epochs = 0



    def step(self, val_loss: float) -> None:
        if val_loss < self.best_val_loss - self.min_delta:
            self.bad_epochs = 0
            self.best_val_loss = val_loss
        else:
            self.bad_epochs += 1

        # Reduce LR
        if self.bad_epochs > self.patience:
            self.optimizer.learning_rate = max(self.optimizer.learning_rate * self.factor, self.min_lr)
            self.bad_epochs = 0




