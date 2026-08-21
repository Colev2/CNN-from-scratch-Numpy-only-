import numpy as np
from conv import Conv_layer
from relu import ReLU_layer
from dense import Dense_layer
from maxpooling import MaxPooling2D_layer
from flatten import Flatten_layer

rng = np.random.default_rng(42)

layers = [Conv_layer(filters=32, stride=1), ReLU_layer(), Conv_layer(filters=32), MaxPooling2D_layer(), Flatten_layer(), Dense_layer(neurons=256)]

shape = (32,32,3)

for layer in layers:
    shape = layer.build(shape)
    print(shape)

parameters = []

