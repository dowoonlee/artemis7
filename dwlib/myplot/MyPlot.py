import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

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

def number_density_histxy(axc, axu, axr, x, y):
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


def bubble_diagram(ax, x, y, w, **kwargs):
    minS, maxS = 5, 200
    scale = (w - np.min(w))/(np.max(w)-np.min(w))
    scale *= (maxS- minS)
    scale += minS

    ax.scatter(x, y, s= scale, facecolor="None", edgecolor="k", **kwargs)
    ax.grid()
    return

def legend(ax, **kwargs):
    legend_kwargs = {
                'fontsize' : 12,
                'framealpha' : 0.9,
                'facecolor' : "w",
                'edgecolor' : 'k',
                'shadow' : False,
                'labelspacing' : 0.1,
            }
    font = fm.FontProperties(family='monospace', weight="light", style="normal", size=12)
    ax.legend(**kwargs, **legend_kwargs, prop=font)


def set_minorticks(ax, axis="both", **kwargs):
    def xaxis(ax):
        majorticks = ax.get_xticks()
        minorbin = abs(majorticks[1]-majorticks[0])/5
        ax.xaxis.set_minor_locator(MultipleLocator(minorbin))
        return
    def yaxis(ax):
        majorticks = ax.get_yticks()
        minorbin = abs(majorticks[1]-majorticks[0])/5
        ax.yaxis.set_minor_locator(MultipleLocator(minorbin))
    if axis == "x":
        xaxis(ax)
    elif axis == "y":
        yaxis(ax)
    else:
        xaxis(ax)
        yaxis(ax)
    return
    

