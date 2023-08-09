import MyPlot as mp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


x = np.random.normal(size= 1000, loc = 2)
y = np.random.normal(size= 1000)
w = np.random.normal(size= 1000, scale=10)


fig = plt.figure(figsize=(10, 10))
gs = GridSpec(nrows=2, ncols=2)
axes = [plt.subplot(gs[i]) for i in range(4)]
mp.colored_scatter(axes[0], x, y, np.arange(len(x)))
mp.number_density(axes[1], x, y)
mp.bubble_diagram(axes[2], x, y, w)
mp.plot(axes[3], [(np.sort(x), y)])
# mp.legend(ax)
for ax in axes:
    mp.set_minorticks(ax)
    mp.set_background(ax)
plt.show()

