import numpy as np
from flatten import Flatten_layer

indexes = np.arange(10)
np.random.shuffle(indexes)
print(indexes)
y = np.array([2, 5, 3, 1, 0, 6, 8, 4, 9, 2])

y = y[indexes]

print(y)

