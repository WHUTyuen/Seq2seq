import numpy as np

a = np.arange(6).reshape(2,3)
print(np.tile(a,(2,1)).shape)
print(a.shape)