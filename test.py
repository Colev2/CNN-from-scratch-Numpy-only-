import numpy as np
from flatten import Flatten_layer

input = np.array([[1, 2], 
                  [3, 4],
                  [5, 6]])
layer = Flatten_layer()
output = layer.forward(input)
print(output.shape)