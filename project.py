import numpy as np
from cnn_numpy.layers.conv import Conv2D
from cnn_numpy.layers.relu import ReLU
from cnn_numpy.layers.maxpooling import MaxPooling2D
from cnn_numpy.layers.flatten import Flatten
from cnn_numpy.layers.dense import Dense
from cnn_numpy.layers.dropout import Dropout
from cnn_numpy.losses.softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
from cnn_numpy.layers.batchnorm import BatchNorm
from cnn_numpy.optimizers import SGD, SGD_momentum, Adam, AdamW
from cnn_numpy.sequential import Sequential
from cnn_numpy.early_stopping import EarlyStopping
from cnn_numpy.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from pathlib import Path


def main():
    dataset_class = get_dataset_class()
    train_data, labels, X_test, y_test = load_dataset(dataset_class)
    epochs = get_epochs()
    initialization = get_initialization()
    distribution = get_distribution()
    learning_rate = get_learning_rate()
    optimizer_class = get_optimizer_class()

    weight_decay = None

    if optimizer_class is AdamW:
        weight_decay = get_weight_decay()

    rng = np.random.default_rng(42)

    # Train/Validation split
    X_train, y_train, X_val, y_val = train_test_split(train_data, labels, validation_size=0.1, rng=rng)

    # In-place standardization of the dataset (new data mean=0, new data std=1)
    mean = np.mean(X_train, axis=(0,1,2), dtype=np.float64, keepdims=True)
    std = np.std(X_train, axis=(0,1,2), dtype=np.float64, keepdims=True)

    # Train set standardization (m=0, σ=1)
    X_train -= mean
    X_train /= std

    # Validation set standardization
    X_val -= mean
    X_val /= std

    # Test set standardization
    X_test -= mean
    X_test /= std

    model = Sequential([
        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Flatten(),

        Dense(neurons=256, rng=rng, initialization="he", distribution="normal"),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        Dropout(drop_prob=0.5, rng=rng),

        Dense(neurons=len(np.unique(labels)), rng=rng, initialization="xavier", distribution="normal"),
            ])

    model.build(X_train.shape[1:])
    optimizer = create_optimizer_object(model, optimizer_class, learning_rate, weight_decay=weight_decay)

    # Early Stopping
    early_stopping = EarlyStopping(patience=7, min_delta=1e-3)

    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, min_delta=1e-3, min_lr=1e-5)

    # Training
    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}...")
        train_loss, train_accuracy = train_epoch(model, X_train, y_train, optimizer, rng)

        val_loss, val_accuracy = evaluate(model, X_val, y_val)

        print(
            f"Epoch {epoch + 1}: "
            f"train_acc: {train_accuracy:.4f}% , "
            f"train_loss: {train_loss:.4f} | "
            f"val_acc: {val_accuracy:.4f}% , "
            f"val_loss: {val_loss:.4f} | "
            f"learning_rate: {optimizer.learning_rate:.4f}"
            )

        stop = early_stopping.step(model=model, val_loss=val_loss, epoch=epoch)
        if stop:
            break

        lr_scheduler.step(val_loss=val_loss)


    # Restore best weights
    early_stopping.restore_best_weights(model)

    # Test evaluation
    # test_loss, test_accuracy = evaluate(model, X_test, y_test)

    # print(f"Test accuracy = {test_accuracy:.2f}% | "
    #    f"Test loss = {test_loss:.3f}"
    #   )




def get_dataset_class():
    dataset_choice = input("Choose training dataset:\n" 
    "1) MNIST\n" 
    "2) Fashion-MNIST\n" 
    "3) CIFAR-10\n" 
    "4) CIFAR-100\n").strip().lower()

    datasets = {
        "1": MNIST,
        "mnist": MNIST,

        "2": FashionMNIST,
        "fashion-mnist": FashionMNIST,
        "fashionmnist": FashionMNIST,

        "3": CIFAR10,
        "cifar-10": CIFAR10,
        "cifar10": CIFAR10,

        "4": CIFAR100,
        "cifar-100": CIFAR100,
        "cifar100": CIFAR100
    }

    try:
        dataset_class = datasets[dataset_choice]
    except KeyError:
        raise ValueError("Dataset choice was invalid")

    return dataset_class


def load_dataset(dataset_class: type):
    DATA_ROOT = Path(__file__).resolve().parent / "data"    # Path to the data subfolder next to this file
    DATA_ROOT.mkdir(parents=True, exist_ok=True)    # Create the folder if it doesn't exist
    
    data_obj = dataset_class(root=str(DATA_ROOT), train=True, download=True)

    test_set_obj = dataset_class(root=str(DATA_ROOT), train=False, download=True)

    data = np.asarray(data_obj.data, dtype=np.float32)
    X_test = np.asarray(test_set_obj.data, dtype=np.float32)

    labels = np.asarray(data_obj.targets, dtype=np.int64)
    y_test = np.asarray(test_set_obj.targets, dtype=np.int64)

    if data.ndim == 3:
        data = data[:, :, :, np.newaxis]
    elif data.ndim != 4:
        raise ValueError("Dataset needs to have shape (B,H,W,C)")

    if X_test.ndim == 3:
        X_test = X_test[:, :, :, np.newaxis]
    elif X_test.ndim != 4:
        raise ValueError("Test set needs to have shape (B,H,W,C)")

    return data, labels, X_test, y_test


def get_epochs():
    try:
        epochs = int(input("Choose epochs: "))
        if epochs <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("Epochs must be an integer greater than 0")

    return epochs


def get_initialization():
    initialization = input("Choose weights' initialization for Convolutional Layers (He or Xavier): ").strip().lower()
    if initialization not in ["he", "xavier"]:
        raise ValueError("Initialization must be either He or Xavier")

    return initialization


def get_distribution():
    distribution = input("Choose weights' distribution for Convolutional Layers (Uniform or Normal): ").strip().lower()
    if distribution not in ["uniform", "normal"]:
        raise ValueError("Distribution must be either Uniform or Normal")

    return distribution


def get_learning_rate():
    try:
        learning_rate = float(input("Choose learning rate: "))
        if learning_rate <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("Learning rate must be a float greater than 0")

    return learning_rate


def get_optimizer_class():
    optimizer_choice = input("Choose optimizer:\n" 
        "1) SGD\n" 
        "2) SGD_momentum\n" 
        "3) Adam\n" 
        "4) AdamW\n").strip().lower()
    
    optimizers = {
            "1": SGD,
            "sgd": SGD,
    
            "2": SGD_momentum,
            "sgd_momentum": SGD_momentum,
    
            "3": Adam,
            "adam": Adam,

            "4": AdamW,
            "adamw": AdamW
        }
    
    try:
        optimizer_class = optimizers[optimizer_choice]
    except KeyError:
        raise ValueError("Optimizer choice was invalid")

    return optimizer_class


def get_weight_decay():
    try:
        weight_decay = float(input("Give weight decay: "))
        if weight_decay < 0 :
            raise ValueError
    except ValueError:
        raise ValueError("Weight decay must be a non-negative float")

    return weight_decay


def create_optimizer_object(model, optimizer_class, learning_rate, weight_decay=None):
    parameters_grads = model.parameters_grads()
    if optimizer_class is SGD:
        optimizer = SGD(parameters_grads, learning_rate)
    elif optimizer_class is SGD_momentum:
        optimizer = SGD_momentum(parameters_grads, learning_rate, momentum_coeff=0.9)
    elif optimizer_class is Adam:
        optimizer = Adam(parameters_grads, learning_rate, b1=0.9, b2=0.999, epsilon=1e-8)
    elif optimizer_class is AdamW:
        optimizer = AdamW(parameters_grads, model.decayable_parameters(), learning_rate, b1=0.9, b2=0.999, epsilon=1e-8, weight_decay=weight_decay)

    return optimizer


def train_test_split(X, y, validation_size: float, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError("Train_test_split: X sample size must be same as y sample size")
    
    if not 0 < validation_size < 1:
        raise ValueError("Train_test_split: split size must be between 0 and 1 exclusively")
    
    if rng is None:
        rng = np.random.default_rng()

    # Stratification: Get same percentage of images from each class on split set
    classes = np.unique(y)
    data_idxs_per_class = {}
    train_class_indexes = []
    val_class_indexes = []

    for class_label in classes:
        data_idxs_per_class[class_label] = np.flatnonzero(y == class_label)     # e.g: data_idxs_per_class = {0: array([3, 700, 250, ...]), 1: array([...]), ...}
        # Shuffle each class's samples, so that we don't take the first 500 (for example) samples of that class in the dataset  
        # for training and the rest for validation
        rng.shuffle(data_idxs_per_class[class_label])
        class_samples = len(data_idxs_per_class[class_label])
        validation_class_samples = int(np.round(validation_size * class_samples))
        train_class_indexes.append(data_idxs_per_class[class_label][:class_samples - validation_class_samples])
        val_class_indexes.append(data_idxs_per_class[class_label][class_samples - validation_class_samples:])

    train_indexes = np.concatenate(train_class_indexes)     # Concatenate all training class samples
    val_indexes = np.concatenate(val_class_indexes)

    X_train = X[train_indexes]
    X_val = X[val_indexes]
    y_train = y[train_indexes]
    y_val = y[val_indexes]

    return X_train, y_train, X_val, y_val


def create_batches(X, y, batch_size, rng=None, shuffle=False):
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of labels must be same as number of images")

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")

    indexes = np.arange(y.shape[0])

    # Shuffle
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indexes)   # e.g: [2,0,5,3,10,...] -> random sample indexes

    for b in range(0, X.shape[0], batch_size):
        batch_indexes = indexes[b:b + batch_size]

        X_batch = X[batch_indexes]
        y_batch = y[batch_indexes]

        yield X_batch, y_batch      # Return one batch at a time


def train_epoch(model, X_train, y_train, optimizer, rng):
    model.train()

    correct_predictions_train = 0
    sample_loss_sum_train = 0

    loss = SoftmaxCrossEntropyLoss()

    train_batches = create_batches(X_train, y_train, batch_size=32, rng=rng, shuffle=True)    # generator object
    
    for X_train_batch, y_train_batch in train_batches:  # Extract tuple (X_train_batch_0, y_train_batch_0), ... from generator
        # ----- Forward -----

        batched_logits_train = model.forward(X_train_batch)
        batch_loss_train = loss.forward(batched_logits_train, y_train_batch)

        # ----- Backward -----

        dlogits = loss.backward()
        model.backward(dlogits)

        # ----- Update -----

        optimizer.update()

        # ----- Sum of sample losses -----
        # (Σ_b (L_b) += batch_loss * B 

        sample_loss_sum_train += batch_loss_train * len(y_train_batch)

        # ----- Accuracy -----

        predicted_class_idx = np.argmax(batched_logits_train, axis=1)
        correct_predictions_train += np.count_nonzero(predicted_class_idx == y_train_batch)

    train_accuracy = (correct_predictions_train / len(y_train)) * 100

    train_loss = sample_loss_sum_train / len(y_train)

    return train_loss, train_accuracy


def evaluate(model, X, y):
    model.eval()

    correct_predictions = 0
    sample_loss_sum = 0

    loss = SoftmaxCrossEntropyLoss()

    # ----- Evaluate -----

    batches = create_batches(X, y, batch_size=32, shuffle=False)

    for X_batch, y_batch in batches:
        logits = model.forward(X_batch)
        batch_loss = loss.forward(logits, y_batch)

        # ----- Sum of sample losses -----
        # (Σ_b (L_b) += batch_loss * B 

        sample_loss_sum += batch_loss * len(y_batch)

        # ----- Accuracy ------

        predicted_class_idx = np.argmax(logits, axis=1)
        correct_predictions += np.count_nonzero(predicted_class_idx == y_batch)

    accuracy = (correct_predictions / X.shape[0]) * 100
    average_loss = sample_loss_sum / len(y)

    return average_loss, accuracy

    
if __name__ == "__main__":
    main()