import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import sys
sys.path.append(sys.path[0]+"/..")
from artemis7.stats._binning import *

## init ##
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

__all__ = [
    "plot", "colored_scatter", "number_density", "number_density_histxy", "bubble_diagram",
    "legend", "set_background", "set_minorticks", "adjust_tickpostion"
]


def plot(ax, xy, **kwargs):
    m = cm.ScalarMappable(norm = Normalize(vmin = 0, vmax = len(xy)), cmap=plt.get_cmap("viridis"))
    xr, yr = (np.inf, -np.inf), (np.inf, -np.inf)
    for i, line in enumerate(xy):
        args = {k:v[i] for k, v in kwargs.items()}
        ax.plot(*line, color="w", linewidth=3)
        ax.plot(*line, **args, color=m.to_rgba(i))
        xr = (np.min([xr[0], line[0].min()]), np.max([xr[1], line[0].max()]))
        yr = (np.min([yr[0], line[1].min()]), np.max([yr[1], line[1].max()]))
    ax.grid()
    ax.set_xlim(xr[0], xr[1])
    ax.set_ylim(yr[0], yr[1])
    return

def colored_scatter(ax, x, y, c):
    m = cm.ScalarMappable(norm = Normalize(vmin = np.min(c), vmax = np.max(c)), cmap=plt.get_cmap("viridis"))
    ax.scatter(x, y, facecolor=m.to_rgba(c), edgecolor="k", alpha=0.5)
    axins = inset_axes(
        ax,
        width = "20%",
        height= "3%",
        loc=3
    )
    plt.colorbar(m, cax = axins, orientation = "horizontal", ticks=[np.min(c), np.max(c)])
    axins.xaxis.set_ticks_position("top")

    ax.grid()
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    return


def number_density(ax, x, y, **kwargs):
    bin = sturges(x)
    sigma_ratio = {0.954:r"$2\sigma$", 0.866:r"$1.5\sigma$", 0.683:r"$1\sigma$"}
    img, xe, ye = np.histogram2d(x, y, bins=bin)
    img_sorted_1d = np.sort(img.reshape(-1))[::-1]
    img_cs = np.cumsum(img_sorted_1d)/np.sum(img_sorted_1d)
    ext = [xe[0], xe[-1], ye[0], ye[-1]]

    lv = np.array([img_sorted_1d[np.where(img_cs > level)[0][0]] for level in sigma_ratio.keys()])
    
    ax.contourf(img.T, extent = ext, cmap=plt.get_cmap("cubehelix_r"), alpha=0.7, vmin = np.percentile(img_sorted_1d, 68), vmax= img_sorted_1d.max(), origin="lower")
    cs = ax.contour(img.T, extent = ext, levels = lv, colors='k', origin="lower")
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
    ax.legend(**kwargs, **legend_kwargs)

def set_background(ax, **kwargs):
    ax.patch.set_facecolor('skyblue')
    ax.patch.set_alpha(0.05)
    return


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

def adjust_tickposition(ax, loc, **kwargs):
    """
        261
        795
        384
    """
    if loc == 1 or loc == 2 or loc == 6:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    if loc == 1 or loc == 5 or loc == 4:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    return