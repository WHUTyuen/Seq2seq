import numpy as np

a = np.arange(24).reshape(2,3,4)
out = a[:,-1,:]
print(out.shape)