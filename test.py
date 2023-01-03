import numpy as np
import matplotlib.pyplot as plt
x = np.random.gamma(2, 2, 10000)
y = np.random.gamma(4, 4, 10000)
plt.scatter(x, y)
plt.show()