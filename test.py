import numpy as np
from flatten import Flatten_layer

input = np.array([[1, 2], 
                  [3, 4],
                  [5, 6]])

for i in range(input.flatten().shape[0]):
    print(input.flat[i])