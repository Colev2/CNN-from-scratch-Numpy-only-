import numpy as np
from conv import Conv_layer
from relu import ReLU_layer
from maxpooling import MaxPooling2D_layer
from flatten import Flatten_layer
from dense import Dense_layer
from softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100


def main():
    dataset_choice = input("Choose training dataset:\n" 
    "1) MNIST\n"
    "2) Fashion-MNIST\n"
    "3) CIFAR-10\n"
    "4) CIFAR-100\n").strip().lower()

    datasets = {
        "1": MNIST,
        "mnist": MNIST,

        "2": FashionMNIST,
        "fashionmnist": FashionMNIST,

        "3": CIFAR10,
        "cifar-10": CIFAR10,

        "4": CIFAR100,
        "cifar-100": CIFAR100
    }

    try:
        dataset_class = datasets[dataset_choice]
    except KeyError:
        raise KeyError("Dataset choice was invalid")

    dataset_obj = dataset_class(root="C:/Users/STAMATIS/Documents/CNN_numpy", train=True, download=True)
    data = np.array(dataset_obj.data, dtype=np.float32)
    labels = np.array(dataset_obj.targets, dtype=np.int64)

    if data.ndim == 3:
        data = data[:, :, :, np.newaxis]
    elif data.ndim != 4:
        raise ValueError("Dataset needs to have shape (B,H,W,C)")

    mean = np.mean(data, axis=(0,1,2), dtype=np.float64, keepdims=True)
    std = np.std(data, axis=(0,1,2), dtype=np.float64, keepdims=True)
    # In-place standardization of the dataset (new data mean=0, new data std=1)
    data -= mean
    data /= std

    indexes = np.arange(data.shape[0])
    rng = np.random.default_rng(seed=42)
    subset_indexes = rng.choice(indexes, size=32, replace=False)
    subset_data = data[subset_indexes]
    subset_labels = labels[subset_indexes]
    subset_classes, counts = np.unique(subset_labels, return_counts=True)

    print(subset_indexes)
    print(subset_data.shape)
    print(subset_labels)
    print(subset_classes, counts)

    batches = create_batches(subset_data, subset_labels, batch_size=32, rng=rng)
    print(batches[0][0].shape)
    print(batches[0][1].shape)
    print(batches[0][1])

    classes, counts = np.unique(batches[0][1], return_counts=True)
    print(classes, counts)

def create_batches(X, y, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of labels must be same as number of images")

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")

    indexes = np.arange(y.shape[0])
    rng.shuffle(indexes)   # [2,0,5,3,10,...] -> random image indexes
    y_shuffled = y[indexes]
    X_shuffled = X[indexes, :, :, :]  

    batches = []

    for b in range(0, X.shape[0], batch_size):
        X_batch = X_shuffled[b:b + batch_size, :, :, :]
        y_batch = y_shuffled[b:b + batch_size]
        batches.append((X_batch, y_batch))

    return batches      # List of ndarrays: [(X_batch_1, y_batch_1), (X_batch2, y_batch_2), ...]





if __name__ == "__main__":
    main()