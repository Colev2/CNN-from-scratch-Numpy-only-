import numpy as np
import copy

a = np.array([1, 2, 3])

x = [a]
y = copy.deepcopy(x)
a[0] = 99
print(a)
print(x)
print(y)

