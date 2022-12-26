import numpy as np
import matplotlib.pyplot as plt

from dwlib.plot.myfigure import myFigure

n = 10
mf = myFigure(figsize=(12, 12))
# for i in range(2):
#     mf.set_cmap(ax_idx= i, cmap='viridis', vrange=(0, 4))

mf.plot(np.sort(np.random.rand(n)), np.random.rand(n), label='hh', ax_idx=0)
mf.scatter(np.random.rand(n), np.random.rand(n),  label=1, s=50, marker='+', ax_idx=0)
mf.scatter(np.random.rand(n), np.random.rand(n),  label=2, s=50, marker='x', ax_idx=0)
mf.scatter(np.random.rand(n), np.random.rand(n),  label='1-', s=50, marker='+', ax_idx=0)

mf.set_lim(ax_idx= 0, xr=(0, 1))
mf.legend(ax_idx = 0, loc=1)
mf.legend(ax_idx = 0, loc=2)
mf.update_layout(hspace=0, wspace=0)
mf.set_axis(xticks=[0.2, 0.4], xlabel=['a', 'b'])

mf.set_grid(0)
plt.show()

