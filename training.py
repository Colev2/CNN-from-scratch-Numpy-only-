import numpy as np
from conv import Conv_layer
from relu import ReLU_layer
from maxpooling import MaxPooling2D_layer
from flatten import Flatten_layer
from dense import Dense_layer
from softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
from optimizer import SGD, SGD_momentum, Adam
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100


def main():
    dataset_class = get_dataset_class()
    train_data, labels, X_test, y_test = load_dataset(dataset_class)
    epochs = get_epochs()
    initialization = input("Choose weights' initialization for Convolutional Layers (He or Xavier): ")
    distribution = input("Choose weights' distribution for Convolutional Layers (Uniform or Normal): ")
    learning_rate = get_learning_rate()
    optimizer_class = get_optimizer_class()
    optimizer = create_optimizer_object(optimizer_class, learning_rate)

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

    conv1 = Conv_layer(in_channels=X_train.shape[3], filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, 
                      initialization=initialization, distribution=distribution)
    feature_maps1 = conv1.forward(X_train[0:1])

    relu1 = ReLU_layer()
    activation1 = relu1.forward(feature_maps1)

    conv2 = Conv_layer(in_channels=feature_maps1.shape[3], filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng,
                        initialization=initialization, distribution=distribution)
    feature_maps2 = conv2.forward(activation1)

    relu2 = ReLU_layer()
    activation2 = relu2.forward(feature_maps2)

    maxpool1 = MaxPooling2D_layer(in_channels=activation2.shape[3], pool_size=(2,2), stride=2)
    maxpool1_output = maxpool1.forward(activation2)

    conv3 = Conv_layer(in_channels=maxpool1_output.shape[3], filters=64, filter_shape=(3,3), padding=1, stride=2, rng=rng,
                       initialization=initialization, distribution=distribution)
    feature_maps3 = conv3.forward(maxpool1_output)

    relu3 = ReLU_layer()
    activation3 = relu3.forward(feature_maps3)

    conv4 = Conv_layer(in_channels=activation3.shape[3], filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng,
                       initialization=initialization, distribution=distribution)
    feature_maps4 = conv4.forward(activation3)

    relu4 = ReLU_layer()
    activation4 = relu4.forward(feature_maps4)

    maxpool2 = MaxPooling2D_layer(in_channels=activation4.shape[3], pool_size=(2,2), stride=2)
    maxpool2_output = maxpool2.forward(activation4)

    flatten = Flatten_layer()
    flattened = flatten.forward(maxpool2_output)

    dense1 = Dense_layer(in_features=flattened.shape[1], neurons=256, rng=rng, initialization="he", distribution="normal")

    relu5 = ReLU_layer()

    dense2 = Dense_layer(in_features=256, neurons=len(np.unique(labels)), rng=rng, initialization="xavier", distribution="normal")

    loss = SoftmaxCrossEntropyLoss()

    # Create validation batches outside of for loop since they don't need shuffling every epoch
    validation_batches = create_batches(X_val, y_val, batch_size=32, rng=rng)

    # Training
    for epoch in range(epochs):
        # Create batches that shuffle each epoch
        train_batches = create_batches(X_train, y_train, batch_size=32, rng=rng)    # [(X_batch_0, y_batch_0), (X_batch_1, y_batch_1), ...]

        correct_predictions_train = 0
        sample_loss_sum_train = 0
        correct_predictions_val = 0
        sample_loss_sum_val = 0
        
        for X_train_batch, y_train_batch in train_batches:

            # ----- Forward -----

            feature_maps1 = conv1.forward(X_train_batch)
            activation1 = relu1.forward(feature_maps1)

            feature_maps2 = conv2.forward(activation1)
            activation2 = relu2.forward(feature_maps2)

            maxpool1_output = maxpool1.forward(activation2)

            feature_maps3 = conv3.forward(maxpool1_output)
            activation3 = relu3.forward(feature_maps3)

            feature_maps4 = conv4.forward(activation3)
            activation4 = relu4.forward(feature_maps4)

            maxpool2_output = maxpool2.forward(activation4)

            flattened = flatten.forward(maxpool2_output)

            dense1_output = dense1.forward(flattened)

            activation5 = relu5.forward(dense1_output)

            logits_train = dense2.forward(activation5)
            batch_loss_train = loss.forward(logits_train, y_train_batch)

            # ----- Backward -----

            loss_gradient = loss.backward()

            dense2_gradient = dense2.backward(loss_gradient)

            relu5_gradient = relu5.backward(dense2_gradient)

            dense1_gradient = dense1.backward(relu5_gradient)

            flatten_gradient = flatten.backward(dense1_gradient)

            maxpool2_gradient = maxpool2.backward(flatten_gradient)

            relu4_gradient = relu4.backward(maxpool2_gradient)
            conv4_gradient = conv4.backward(relu4_gradient)

            relu3_gradient = relu3.backward(conv4_gradient)
            conv3_gradient = conv3.backward(relu3_gradient)

            maxpool1_gradient = maxpool1.backward(conv3_gradient)

            relu2_gradient = relu2.backward(maxpool1_gradient)
            conv2_gradient = conv2.backward(relu2_gradient)

            relu1_gradient = relu1.backward(conv2_gradient)
            _ = conv1.backward(relu1_gradient)

            # ----- Trainable parameters -----

            parameters = [(conv1.weights, conv1.dweights), (conv1.bias, conv1.dbias), (conv2.weights, conv2.dweights), (conv2.bias, conv2.dbias),
                    (conv3.weights, conv3.dweights), (conv3.bias, conv3.dbias), (conv4.weights, conv4.dweights), (conv4.bias, conv4.dbias),
                    (dense1.weights, dense1.dweights), (dense1.bias, dense1.dbias), (dense2.weights, dense2.dweights), (dense2.bias, dense2.dbias)]


            # ----- Update -----

            optimizer.update(parameters)

            # ----- Sum of sample losses -----

            sample_loss_sum_train += batch_loss_train * len(y_train_batch)

            # ----- Accuracy -----

            highest_prob_idx_train = np.argmax(logits_train, axis=1)
            correct_predictions_train += np.count_nonzero(highest_prob_idx_train == y_train_batch)

        train_accuracy = (correct_predictions_train / X_train.shape[0]) * 100

        # ----- Validation Pass -----
            
        for X_val_batch, y_val_batch in validation_batches:
            feature_maps1 = conv1.forward(X_val_batch)
            activation1 = relu1.forward(feature_maps1)

            feature_maps2 = conv2.forward(activation1)
            activation2 = relu2.forward(feature_maps2)

            maxpool1_output = maxpool1.forward(activation2)

            feature_maps3 = conv3.forward(maxpool1_output)
            activation3 = relu3.forward(feature_maps3)

            feature_maps4 = conv4.forward(activation3)
            activation4 = relu4.forward(feature_maps4)

            maxpool2_output = maxpool2.forward(activation4)

            flattened = flatten.forward(maxpool2_output)

            dense1_output = dense1.forward(flattened)

            activation5 = relu5.forward(dense1_output)

            logits_val = dense2.forward(activation5)
            batch_loss_val = loss.forward(logits_val, y_val_batch)

            # ----- Sum of sample losses -----

            sample_loss_sum_val += batch_loss_val * len(y_val_batch)

            # ----- Accuracy ------

            highest_prob_idx_val = np.argmax(logits_val, axis=1)
            correct_predictions_val += np.count_nonzero(highest_prob_idx_val == y_val_batch)

        val_accuracy = (correct_predictions_val / X_val.shape[0]) * 100

        print(f"Epoch {epoch + 1}: train_accuracy = {train_accuracy}%, loss = {sample_loss_sum_train / len(y_train)}, \
              val_accuracy = {val_accuracy}%, val_loss = {sample_loss_sum_val / len(y_val)}")


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
    data_obj = dataset_class(root="C:/Users/STAMATIS/Documents/CNN_numpy", train=True, download=True)

    test_set_obj = dataset_class(root="C:/Users/STAMATIS/Documents/CNN_numpy", train=False, download=True)

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
        "3) Adam\n").strip().lower()
    
    optimizers = {
            "1": SGD,
            "sgd": SGD,
    
            "2": SGD_momentum,
            "sgd_momentum": SGD_momentum,
    
            "3": Adam,
            "adam": Adam
        }
    
    try:
        optimizer_class = optimizers[optimizer_choice]
    except KeyError:
        raise ValueError("Optimizer choice was invalid")

    return optimizer_class


def create_optimizer_object(optimizer_class, learning_rate):
    if optimizer_class == SGD:
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_class == SGD_momentum:
        optimizer = SGD_momentum(learning_rate=learning_rate, momentum_coeff=0.9)
    elif optimizer_class == Adam:
        optimizer = Adam(learning_rate=learning_rate, b1=0.9, b2=0.999, epsilon=0.001)

    return optimizer


def create_batches(X, y, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of labels must be same as number of images")

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")

    # Shuffle
    indexes = np.arange(y.shape[0])
    rng.shuffle(indexes)   # [2,0,5,3,10,...] -> random image indexes
    y_shuffled = y[indexes]
    X_shuffled = X[indexes]  

    batches = []

    for b in range(0, X.shape[0], batch_size):
        X_batch = X_shuffled[b:b + batch_size, :, :, :]
        y_batch = y_shuffled[b:b + batch_size]
        batches.append((X_batch, y_batch))

    return batches      # List of ndarrays: [(X_batch_1, y_batch_1), (X_batch2, y_batch_2), ...]



def train_test_split(X, y, validation_size: float, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError("Train_test_split: X sample size must be same as y sample size")
    
    if not 0 <= validation_size <= 1:
        raise ValueError("Train_test_split: split size must be between 0 and 1 inclusively")
    
    if rng is None:
        rng = np.random.default_rng()

    # Stratification:
    classes = np.unique(y)
    data_idxs_per_class = {}
    train_class_indexes = []
    val_class_indexes = []

    for class_label in classes:
        data_idxs_per_class[class_label] = np.flatnonzero(y == class_label)     # e.g: data_idxs_per_class = {0: array([3, 700, 250, ...]), 1: array([...]), ...} 
        rng.shuffle(data_idxs_per_class[class_label])
        class_samples = len(data_idxs_per_class[class_label])
        validation_class_samples = int(np.round(validation_size * class_samples))
        train_class_indexes.append(data_idxs_per_class[class_label][:class_samples - validation_class_samples])
        val_class_indexes.append(data_idxs_per_class[class_label][class_samples - validation_class_samples:])

    train_indexes = np.concatenate(train_class_indexes)
    val_indexes = np.concatenate(val_class_indexes)

    rng.shuffle(train_indexes)
    rng.shuffle(val_indexes)

    X_train = X[train_indexes]
    X_val = X[val_indexes]
    y_train = y[train_indexes]
    y_val = y[val_indexes]

    return X_train, y_train, X_val, y_val



    
if __name__ == "__main__":
    main()