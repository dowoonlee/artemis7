import MyPlot as mp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


x = np.random.normal(size= 1000, loc = 2)
y = np.random.normal(size= 1000)

fig = plt.figure(figsize=(5, 5))
gs = GridSpec(nrows=3, ncols=3, hspace=0, wspace=0)
axc = plt.subplot(gs[1:, :2])
axu = plt.subplot(gs[0, :2])
axr = plt.subplot(gs[1:, 2])
mp.hist_density(axc, axu, axr, x, y)
plt.show()

