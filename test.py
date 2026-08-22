import numpy as np
from conv import Conv_layer
from relu import ReLU_layer
from dense import Dense_layer
from maxpooling import MaxPooling2D_layer
from flatten import Flatten_layer

conv1 = Conv_layer(filters=3)
conv2 = Conv_layer(filters=3)
conv1.weights = np.array([0, 1])
conv1.bias = np.array([2,3])
conv2.weights = np.array([4,5])
conv2.bias = np.array([6,7])
x = [[conv1.weights.copy(), conv1.bias.copy()], [conv2.weights.copy(), conv2.bias.copy()]]
conv1.weights[...] = np.array([0, 2])
conv2.bias[...] = np.array([8,9])
print(x)
