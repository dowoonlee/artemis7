import MyPlot as mp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


x = np.random.normal(size= 100, loc = 2)
y = np.random.normal(size= 100)
w = np.random.normal(size= 100, scale=10)

fig = plt.figure(figsize=(5, 5))
ax = plt.subplot()
mp.bubble_diagram(ax, x, y, w, label="bubble")
mp.legend(ax)
mp.set_minorticks(ax)
plt.show()

