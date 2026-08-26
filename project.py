import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from cnn_numpy.layers.conv import Conv2D
from cnn_numpy.layers.relu import ReLU
from cnn_numpy.layers.maxpooling import MaxPooling2D
from cnn_numpy.layers.flatten import Flatten
from cnn_numpy.layers.dense import Dense
from cnn_numpy.layers.dropout import Dropout
from cnn_numpy.layers.batchnorm import BatchNorm
from cnn_numpy.losses.softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
from cnn_numpy.optimizers import SGD, SGD_momentum, Adam, AdamW
from cnn_numpy.sequential import Sequential
from cnn_numpy.early_stopping import EarlyStopping
from cnn_numpy.lr_scheduler import ReduceLROnPlateau
from cnn_numpy.data_augmentation import augment_batch
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100


def main():
    dataset_class = get_dataset_class()
    dataset_info = get_dataset_info(dataset_class)

    print(f"\nSelected dataset: {dataset_info['name']}")

    X_train_full, y_train_full, X_test, y_test, class_names = load_dataset(dataset_class)

    train_model_choice = get_yes_no("\nTrain model? (yes/no): ")

    if train_model_choice:
        model, mean, std = train_model(dataset_class, X_train_full, y_train_full)

        save = get_yes_no("\nSave best weights after training? (yes/no): ")

        if save:
            save_trained_model(model, dataset_class, mean, std, class_names)

    else:
        loaded_model = load_trained_model(dataset_class)

        if loaded_model is None:
            print("\nProgram terminated")
            return

        model, mean, std, class_names = loaded_model

    test_eval = get_yes_no("\nEvaluate best model on the test set? (yes/no): ")

    if test_eval:
        evaluate_test_model(model, X_test, y_test, mean, std)

    make_predictions(model, dataset_class, mean, std, class_names)

    print("\nProgram terminated")


# User input

def get_yes_no(question: str):
    while True:
        choice = input(question).strip().lower()

        if choice in ("yes", "y"):
            return True

        if choice in ("no", "n"):
            return False

        print("Please answer yes or no.")


def get_dataset_class():
    dataset_choice = input("Choose dataset:\n"
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

        if weight_decay < 0:
            raise ValueError

    except ValueError:
        raise ValueError("Weight decay must be a non-negative float")

    return weight_decay


def get_data_augmentation():
    augmentation_choice = input("Use data augmentation? (yes/no): ").strip().lower()

    if augmentation_choice in ("yes", "y"):
        return True

    if augmentation_choice in ("no", "n"):
        return False

    raise ValueError("Data augmentation choice must be yes or no")


# Dataset information

def get_dataset_info(dataset_class):
    if dataset_class is MNIST:
        return {
            "name": "MNIST",
            "filename": "mnist",
            "input_shape": (28, 28, 1),
            "image_mode": "L"
            }

    if dataset_class is FashionMNIST:
        return {
            "name": "Fashion-MNIST",
            "filename": "fashion_mnist",
            "input_shape": (28, 28, 1),
            "image_mode": "L"
        }

    if dataset_class is CIFAR10:
        return {
            "name": "CIFAR-10",
            "filename": "cifar10",
            "input_shape": (32, 32, 3),
            "image_mode": "RGB"
        }

    if dataset_class is CIFAR100:
        return {
            "name": "CIFAR-100",
            "filename": "cifar100",
            "input_shape": (32, 32, 3),
            "image_mode": "RGB"
        }

    raise ValueError("Unsupported dataset")


# Dataset loading

def get_data_root():
    #  Get the path of the data file. Path(__file__).resolve().parent ensures it will be in the same folder
    # as the one project.py is in
    data_root = Path(__file__).resolve().parent / "data"    
    data_root.mkdir(parents=True, exist_ok=True)

    return data_root


def load_dataset(dataset_class: type):
    data_root = get_data_root()

    train_dataset = dataset_class(root=str(data_root), train=True, download=True)
    test_dataset = dataset_class(root=str(data_root), train=False, download=True)

    X_train = np.asarray(train_dataset.data, dtype=np.float32)
    y_train = np.asarray(train_dataset.targets, dtype=np.int64)

    X_test = np.asarray(test_dataset.data, dtype=np.float32)
    y_test = np.asarray(test_dataset.targets, dtype=np.int64)

    class_names = list(train_dataset.classes)

    if X_train.ndim == 3:
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]

    elif X_train.ndim != 4:
        raise ValueError("Dataset needs to have shape (B,H,W,C)")

    return X_train, y_train, X_test, y_test, class_names


# Model creation

def create_model(dataset_class, num_classes, rng, initialization="he", distribution="normal"):
    dataset_info = get_dataset_info(dataset_class)

    model = Sequential([
        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Flatten(),
        Dropout(drop_prob=0.5, rng=rng),

        Dense(neurons=256, rng=rng, initialization="he", distribution="normal"),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        Dense(neurons=num_classes, rng=rng, initialization="xavier", distribution="normal")
    ])

    model.build(dataset_info["input_shape"])

    return model


# Training

def train_model(dataset_class, X_train_full, y_train_full):
    epochs = get_epochs()
    initialization = get_initialization()
    distribution = get_distribution()
    learning_rate = get_learning_rate()
    optimizer_class = get_optimizer_class()

    weight_decay = None

    if optimizer_class is AdamW:
        weight_decay = get_weight_decay()

    use_data_augm = get_data_augmentation()

    rng = np.random.default_rng(42)

    X_train, y_train, X_val, y_val = train_test_split(X_train_full, y_train_full, validation_size=0.1, rng=rng)

    mean = np.mean(X_train, axis=(0,1,2), dtype=np.float64)
    std = np.std(X_train, axis=(0,1,2), dtype=np.float64)

    X_train -= mean
    X_train /= std

    X_val -= mean
    X_val /= std

    model = create_model(dataset_class=dataset_class, num_classes=len(np.unique(y_train_full)), rng=rng,
                initialization=initialization, distribution=distribution)

    optimizer = create_optimizer_object(model, optimizer_class, learning_rate, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=7, min_delta=1e-3)

    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, min_delta=1e-3, min_lr=1e-5)

    crop_padding = None
    horizontal_flip = None

    if use_data_augm:
        crop_padding, horizontal_flip = get_augmentation_config(dataset_class)

    for epoch in range(epochs):
        print(f"\nTraining epoch {epoch + 1}...")

        train_loss, train_accuracy = train_epoch(model, X_train, y_train, optimizer, rng, use_data_augm, 
                                            crop_padding, horizontal_flip)

        val_loss, val_accuracy = evaluate(model, X_val, y_val)

        print(
            f"Epoch {epoch + 1}: "
            f"train_acc: {train_accuracy:.4f}% , "
            f"train_loss: {train_loss:.4f} | "
            f"val_acc: {val_accuracy:.4f}% , "
            f"val_loss: {val_loss:.4f} | "
            f"learning_rate: {optimizer.learning_rate:.5f}"
        )

        stop = early_stopping.step(model=model, val_loss=val_loss, epoch=epoch)

        if stop:
            break

        lr_scheduler.step(val_loss=val_loss)

    print("\nTraining finished")

    early_stopping.restore_best_weights(model)

    print("Best model weights were restored.")

    return model, mean, std


# Save / Load

def get_saved_model_paths(dataset_class):
    dataset_info = get_dataset_info(dataset_class)

    save_dir = Path(__file__).resolve().parent / "saved_models"

    filename = dataset_info["filename"]

    # Paths for weights and 
    weights_path = save_dir / f"{filename}_weights.npz"
    preprocessing_path = save_dir / f"{filename}_preprocessing.npz"

    return weights_path, preprocessing_path


def save_trained_model(model, dataset_class, mean, std, class_names):
    dataset_info = get_dataset_info(dataset_class)

    weights_path, preprocessing_path = get_saved_model_paths(dataset_class)

    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Save weights e.g inside .../saved_models/cifar-10_weights.npz
    model.save_weights(weights_path)

    # Save mean, std, class_names ndarrays inside e.g: .../saved_models/cifar-10_preprocessing.npz
    np.savez(preprocessing_path, mean=mean, std=std, class_names=np.asarray(class_names, dtype=str))

    print(f"\nBest {dataset_info['name']} model was saved.")
    print(f"Weights: {weights_path}")
    print(f"Preprocessing statistics: {preprocessing_path}")


def load_trained_model(dataset_class):
    dataset_info = get_dataset_info(dataset_class)

    weights_path, preprocessing_path = get_saved_model_paths(dataset_class)

    if not weights_path.is_file() or not preprocessing_path.is_file():
        print(f"\nNo trained weights for {dataset_info['name']} were found.")
        print("Train a model first.")

        return None

    with np.load(preprocessing_path, allow_pickle=False) as preprocessing_data:
        mean = preprocessing_data["mean"]
        std = preprocessing_data["std"]
        class_names = preprocessing_data["class_names"].tolist()

    rng = np.random.default_rng(42)

    # Build a model with same architecture and set its weights to the saved ones
    model = create_model(dataset_class=dataset_class, num_classes=len(class_names), rng=rng,
                initialization="he", distribution="normal")

    model.load_weights(weights_path)
    model.eval()    # Set evaluation mode on

    print(f"\nBest {dataset_info['name']} model was loaded.")

    return model, mean, std, class_names



# Test evaluation

def evaluate_test_model(model, X_test, y_test, mean, std):
    X_test -= mean
    X_test /= std

    test_loss, test_accuracy = evaluate(model, X_test, y_test)

    print(f"\nTest accuracy = {test_accuracy:.2f}%")
    print(f"Test loss = {test_loss:.3f}")


# Single image prediction

def print_available_classes(class_names):
    print("\nAvailable classes:")

    classes_per_line = 10

    for start in range(0, len(class_names), classes_per_line):
        print("  " + ", ".join(class_names[start:start + classes_per_line]))


def load_and_preprocess_image(image_path, dataset_class, mean, std):
    dataset_info = get_dataset_info(dataset_class)

    image_path = str(image_path).strip().strip('"').strip("'")
    image_path = Path(image_path).expanduser()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file '{image_path}' was not found.")

    with Image.open(image_path) as image:
        image = image.convert(dataset_info["image_mode"])   # Convert image type to corresponding dataset img type

        height, width, _ = dataset_info["input_shape"]

        # Resize to the dimensions of the dataset the model trained on
        image = image.resize((width, height))
        image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]     # add channel axis

    expected_shape = dataset_info["input_shape"]

    if image.shape != expected_shape:
        raise ValueError(f"Processed image has shape {image.shape}, but model expects {expected_shape}")

    image = (image - mean) / std
    image = np.asarray(image, dtype=np.float32)

    image = image[np.newaxis, ...]  # add batch axis

    return image


def predict_image(model, image, class_names):
    model.eval()

    logits = model.forward(image)[0]    # shape: (num_classes,)

    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(exp_logits)

    predicted_class_idx = np.argmax(probabilities)

    predicted_class = class_names[predicted_class_idx]
    confidence_probab = probabilities[predicted_class_idx]

    return predicted_class, confidence_probab


def show_prediction(image_path, predicted_class, confidence):
    with Image.open(image_path) as image:
        image = image.copy()

    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence * 100:.2f}%")
    plt.axis("off")
    plt.show()


def make_predictions(model, dataset_class, mean, std, class_names):
    prediction = get_yes_no("\nMake a prediction? (yes/no): ")

    if not prediction:
        return

    print_available_classes(class_names)

    print("\nGive an image that belongs to one of the classes shown above.")

    while True:
        image_path = input("\nEnter image path: ")

        try:
            image = load_and_preprocess_image(image_path, dataset_class, mean, std)

        except FileNotFoundError as error:
            print(error)
            continue

        except OSError:
            print("The selected file could not be opened as an image.")
            continue

        predicted_class, confidence = predict_image(model, image, class_names)

        show_prediction(image_path, predicted_class, confidence)

        another_prediction = get_yes_no("\nMake another prediction? (yes/no): ")

        if not another_prediction:
            break


# Optimizer

def create_optimizer_object(model, optimizer_class, learning_rate, weight_decay=None):
    parameters_grads = model.parameters_grads()

    if optimizer_class is SGD:
        optimizer = SGD(parameters_grads, learning_rate)

    elif optimizer_class is SGD_momentum:
        optimizer = SGD_momentum(parameters_grads, learning_rate, momentum_coeff=0.9)

    elif optimizer_class is Adam:
        optimizer = Adam(parameters_grads, learning_rate, b1=0.9, b2=0.999, epsilon=1e-8)

    elif optimizer_class is AdamW:
        optimizer = AdamW(parameters_grads, model.decayable_parameters(), learning_rate, b1=0.9, b2=0.999,
                        epsilon=1e-8, weight_decay=weight_decay)

    return optimizer


# Train / Validation split

def train_test_split(X, y, validation_size: float, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError("Train_test_split: X sample size must be same as y sample size")

    if not 0 < validation_size < 1:
        raise ValueError("Train_test_split: split size must be between 0 and 1 exclusively")

    if rng is None:
        rng = np.random.default_rng()

    # Stratification: Get same percentage of images from each class on split set
    classes = np.unique(y)  # https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    data_idxs_per_class = {}
    train_class_indexes = []
    val_class_indexes = []

    for class_label in classes:
        # https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html
        # e.g: data_idxs_per_class = {0: array([3, 700, 250, ...]), 1: array([...]), ...}
        data_idxs_per_class[class_label] = np.flatnonzero(y == class_label) 
        # Shuffle each class's samples, so that we don't take the first 500 (for example) samples 
        # of that class in the dataset for training and the rest for validation
        rng.shuffle(data_idxs_per_class[class_label])

        class_samples = len(data_idxs_per_class[class_label])
        validation_class_samples = int(np.round(validation_size * class_samples))

        train_class_indexes.append(data_idxs_per_class[class_label][:class_samples - validation_class_samples])
        val_class_indexes.append(data_idxs_per_class[class_label][class_samples - validation_class_samples:])

    train_indexes = np.concatenate(train_class_indexes)     # Concatenate all training class samples
    val_indexes = np.concatenate(val_class_indexes)    # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html

    X_train = X[train_indexes]
    X_val = X[val_indexes]

    y_train = y[train_indexes]
    y_val = y[val_indexes]

    return X_train, y_train, X_val, y_val


# Mini-batches

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

        rng.shuffle(indexes)    # e.g: [2,0,5,3,10,...] -> random sample indexes

    for b in range(0, X.shape[0], batch_size):
        batch_indexes = indexes[b:b + batch_size]

        X_batch = X[batch_indexes]
        y_batch = y[batch_indexes]

        yield X_batch, y_batch      # Return one batch at a time


# Data augmentation configuration

def get_augmentation_config(dataset_class):
    if dataset_class is MNIST:
        return 2, False

    if dataset_class is FashionMNIST:
        return 2, True

    if dataset_class in (CIFAR10, CIFAR100):
        return 4, True

    raise ValueError("Unsupported dataset")


# One training epoch

def train_epoch(model, X_train, y_train, optimizer, rng, use_data_augm, crop_padding, horizontal_flip):
    model.train()

    correct_predictions_train = 0
    sample_loss_sum_train = 0

    loss = SoftmaxCrossEntropyLoss()

    train_batches = create_batches(X_train, y_train, batch_size=32, rng=rng, shuffle=True)  # generator object

    for X_train_batch, y_train_batch in train_batches:  # Extract tuple (X_train_batch_0, y_train_batch_0), ... from generator
        if use_data_augm:
            X_train_batch = augment_batch(X_train_batch, rng=rng, crop_padding=crop_padding, horizontal_flip=horizontal_flip)

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
        # https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html
        predicted_class_idx = np.argmax(batched_logits_train, axis=1)
        correct_predictions_train += np.count_nonzero(predicted_class_idx == y_train_batch)

    train_accuracy = (correct_predictions_train / len(y_train)) * 100
    train_loss = sample_loss_sum_train / len(y_train)

    return train_loss, train_accuracy


# Evaluation

def evaluate(model, X, y):
    model.eval()

    correct_predictions = 0
    sample_loss_sum = 0

    loss = SoftmaxCrossEntropyLoss()

    batches = create_batches(X, y, batch_size=32, shuffle=False)

    for X_batch, y_batch in batches:
        logits = model.forward(X_batch)
        batch_loss = loss.forward(logits, y_batch)

        sample_loss_sum += batch_loss * len(y_batch)

        predicted_class_idx = np.argmax(logits, axis=1)
        correct_predictions += np.count_nonzero(predicted_class_idx == y_batch)

    accuracy = (correct_predictions / X.shape[0]) * 100
    average_loss = sample_loss_sum / len(y)

    return average_loss, accuracy


if __name__ == "__main__":
    main()