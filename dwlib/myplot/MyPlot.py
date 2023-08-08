import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import numpy as np
import sys
sys.path.append(sys.path[0]+"/..")
from stats.binning import *

def number_density(ax, x, y, **kwargs):
    bin = sturges(x)
    sigma_ratio = {0.954:r"$2\sigma$", 0.866:r"$1.5\sigma$", 0.683:r"$1\sigma$"}
    img, xe, ye = np.histogram2d(x, y, bins=bin)
    img_sorted_1d = np.sort(img.reshape(-1))[::-1]
    img_cs = np.cumsum(img_sorted_1d)/np.sum(img_sorted_1d)
    ext = [xe[0], xe[-1], ye[0], ye[-1]]

    lv = np.array([img_sorted_1d[np.where(img_cs > level)[0][0]] for level in sigma_ratio.keys()])
    
    ax.contourf(img.T, extent = ext, cmap=plt.get_cmap("cubehelix_r"), alpha=0.7, vmin = np.percentile(img_sorted_1d, 68), vmax= img_sorted_1d.max())
    cs = ax.contour(img.T, extent = ext, levels = lv, colors='k')
    plt.clabel(cs, levels=lv, fontsize=10, fmt={l:v for (l, v) in zip(lv, sigma_ratio.values())})
    ax.scatter(np.mean(x), np.mean(y), marker="x", color="w", s= 50)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.grid()
    return

def hist_density(axc, axu, axr, x, y):
    bin = sturges(x)
    axc = number_density(axc, x, y)
    
    axu.hist(x, bins= bin, range=(np.min(x), np.max(x)), histtype="step", color="k")
    axu.set_xlim(np.min(x), np.max(x))
    axu.grid(axis="x")
    axu.xaxis.tick_top()
    axu.xaxis.set_label_position('top')

    axr.hist(y, bins= bin, range=(np.min(y), np.max(y)), histtype="step", orientation="horizontal", color="k")
    axr.set_ylim(np.min(y), np.max(y))
    axr.grid(axis="y")
    axr.yaxis.tick_right()
    axr.yaxis.set_label_position('right')
    return








