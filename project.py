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
        predictions = evaluate_test_model(model, X_test, y_test, mean, std)

        show_matrix = get_yes_no("\nShow confusion matrix? (yes/no): ")

        if show_matrix:
            matrix = create_confusion_matrix(y_test, predictions, len(class_names))

            print_per_class_accuracy(matrix, class_names)
            show_confusion_matrix(matrix, class_names)

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

def create_default_model(dataset_class, num_classes, rng, initialization="he", distribution="normal"):
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



def create_cifar100_model(num_classes, rng, initialization="he", distribution="normal"):
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

        Conv2D(filters=128, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        Conv2D(filters=128, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Flatten(),
        Dropout(drop_prob=0.25, rng=rng),

        Dense(neurons=256, rng=rng, initialization="he", distribution="normal"),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        Dense(neurons=num_classes, rng=rng, initialization="xavier", distribution="normal")
    ])

    model.build((32, 32, 3))

    return model



def create_model(dataset_class, num_classes, rng, initialization="he", distribution="normal"):
    if dataset_class is CIFAR100:
        return create_cifar100_model(num_classes, rng, initialization, distribution)

    return create_default_model(dataset_class, num_classes, rng, initialization, distribution)




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
        mean = preprocessing_data["mean"].reshape(-1)
        std = preprocessing_data["std"].reshape(-1)
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
    X_test = X_test.copy()

    X_test -= mean
    X_test /= std

    test_loss, test_accuracy, predictions = evaluate(model, X_test, y_test, return_predictions=True)

    print(f"\nTest accuracy = {test_accuracy:.2f}%")
    print(f"Test loss = {test_loss:.3f}")

    return predictions


# Single image prediction

def print_available_classes(class_names):
    print("\nAvailable classes:")

    classes_per_line = 10

    for start in range(0, len(class_names), classes_per_line):
        print("  " + ", ".join(class_names[start:start + classes_per_line]))


def center_crop_and_resize(image, target_width, target_height):
    source_width, source_height = image.size

    source_aspect_ratio = source_width / source_height
    target_aspect_ratio = target_width / target_height

    if source_aspect_ratio > target_aspect_ratio:
        # Image is too wide -> crop left and right
        new_width = round(source_height * target_aspect_ratio)
        left = (source_width - new_width) // 2

        image = image.crop((left, 0, left + new_width, source_height))

    elif source_aspect_ratio < target_aspect_ratio:
        # Image is too tall -> crop top and bottom
        new_height = round(source_width / target_aspect_ratio)
        top = (source_height - new_height) // 2

        image = image.crop((0, top, source_width, top + new_height))

    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return image


def load_and_preprocess_image(image_path, dataset_class, mean, std):
    dataset_info = get_dataset_info(dataset_class)

    image_path = str(image_path).strip().strip('"').strip("'")
    image_path = Path(image_path).expanduser()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file '{image_path}' was not found.")

    with Image.open(image_path) as image:
        image = image.convert(dataset_info["image_mode"])

        height, width, _ = dataset_info["input_shape"]

        image = center_crop_and_resize(image, width, height)

        # Keep a copy to display what the model receives spatially
        resized_image = image.copy()

        image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    expected_shape = dataset_info["input_shape"]

    if image.shape != expected_shape:
        raise ValueError(f"Processed image has shape {image.shape}, but model expects {expected_shape}")

    image = (image - mean) / std
    image = np.asarray(image, dtype=np.float32)

    image = image[np.newaxis, ...]

    return image, resized_image


def predict_image(model, image, class_names):
    model.eval()

    logits = model.forward(image)[0]    # shape: (num_classes,)

    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(exp_logits)

    # Sort class indices from highest to lowest probability and get the top 3
    top3_indices = np.argsort(probabilities)[::-1][:3]  # https://numpy.org/doc/stable/reference/generated/numpy.argsort.html

    top3_predictions = []

    for class_idx in top3_indices:
        predicted_class = class_names[class_idx]
        confidence = probabilities[class_idx]

        top3_predictions.append((predicted_class, confidence))  # e.g: [(cat, 80%), (dog, 10%), (deer, 5%)]

    return top3_predictions


def show_prediction(image_path, resized_image, predicted_class, confidence):
    with Image.open(image_path) as image:
        original_image = image.copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].set_title(f"Original\n{original_image.width} x {original_image.height}")
    axes[0].axis("off")

    axes[1].imshow(resized_image, interpolation="nearest")
    axes[1].set_title(f"Model input\n{resized_image.width} x {resized_image.height}")
    axes[1].axis("off")

    fig.suptitle(f"Prediction: {predicted_class} | Confidence: {confidence * 100:.2f}%")

    plt.tight_layout()
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
            image, resized_image = load_and_preprocess_image(image_path, dataset_class, mean, std)

        except FileNotFoundError as error:
            print(error)
            continue

        except OSError:
            print("The selected file could not be opened as an image.")
            continue

        top3_predictions = predict_image(model, image, class_names)

        print("\nTop 3 predictions:")

        for rank, (predicted_class, confidence) in enumerate(top3_predictions, start=1):
            print(f"{rank}. {predicted_class}: {confidence * 100:.2f}%")

        predicted_class, confidence = top3_predictions[0]

        show_prediction(image_path, resized_image, predicted_class, confidence)

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

def evaluate(model, X, y, return_predictions=False):
    model.eval()

    correct_predictions = 0
    sample_loss_sum = 0

    loss = SoftmaxCrossEntropyLoss()

    if return_predictions:
        all_predictions = []

    batches = create_batches(X, y, batch_size=32, shuffle=False)

    for X_batch, y_batch in batches:
        logits = model.forward(X_batch)
        batch_loss = loss.forward(logits, y_batch)

        sample_loss_sum += batch_loss * len(y_batch)

        predicted_class_idx = np.argmax(logits, axis=1)
        correct_predictions += np.count_nonzero(predicted_class_idx == y_batch)

        if return_predictions:
            all_predictions.append(predicted_class_idx)

    accuracy = (correct_predictions / X.shape[0]) * 100
    average_loss = sample_loss_sum / len(y)

    if return_predictions:
        all_predictions = np.concatenate(all_predictions)
        return average_loss, accuracy, all_predictions

    return average_loss, accuracy


def create_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_class, predicted_class in zip(y_true, y_pred):
        matrix[true_class, predicted_class] += 1

    return matrix


def show_confusion_matrix(matrix, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(matrix)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center")

    fig.colorbar(image, ax=ax)

    plt.tight_layout()
    plt.show()


def print_per_class_accuracy(matrix, class_names):
    print("\nPer-class test accuracy:")

    for class_idx, class_name in enumerate(class_names):
        correct = matrix[class_idx, class_idx]
        total = np.sum(matrix[class_idx])
        accuracy = correct / total * 100

        print(f"{class_name}: {accuracy:.2f}%")
















if __name__ == "__main__":
    main()