import numpy as np
from conv import Conv_layer
from relu import ReLU_layer

x = np.array([1,2])
b = x.copy()
x = np.append(x, 3)
print(x)
print(b)