import numpy as np

x = np.random.random((100, 100))
p = np.array(np.where(x<0.28))
p = p.T
print(p[1])