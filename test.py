import numpy as np
from cnn_numpy.layers.conv import Conv2D
from cnn_numpy.layers.relu import ReLU
from cnn_numpy.layers.dense import Dense
from cnn_numpy.layers.maxpooling import MaxPooling2D
from cnn_numpy.layers.flatten import Flatten
from project import create_batches, train_test_split, create_optimizer_object

X = np.arange(20).reshape(10, 2)
y = np.arange(10)

print(X)
print(y)
batches = create_batches(X, y, batch_size=4, shuffle=False)
print(np.concatenate([y_batch for _, y_batch in batches], axis=1))