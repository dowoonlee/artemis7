import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import namedtuple

import numpy as np
CmapInfo = namedtuple('CmapInfo', [
    'cmap', 'vr'
])
AxisInfo = namedtuple('AxisInfo', [
    'ax', 'nplt', 'nsct', 'xr', 'yr', 'cmapinfo'
])

class myFigure(object):
    def __init__(self, figsize = (8, 8), nrows=1, ncols=1):
        self._fontsize = 14

        self._fig = plt.figure(figsize = figsize)
        if not isinstance(nrows, int) or not isinstance(ncols, int):
            raise TypeError("N_Row and N_Column should be integer")

        cmap_basis = CmapInfo(cm.get_cmap('jet'), (0, 1))
        self.gs = GridSpec(nrows=nrows, ncols=ncols)
        self._axis_info = [
            AxisInfo(
                ax = plt.subplot(self.gs[i]),
                nplt = 0,
                nsct = 0,
                xr = (),
                yr = (),
                cmapinfo =cmap_basis) for i in range(nrows*ncols)
        ]
    @staticmethod
    def _return_default(kw, k, default_value):
        if k in kw.keys():
            return kw[k]
        return default_value

    def update_layout(self, **kwargs):
        """
        Update the space between axes
        hspace : Vertical space
        wspace : Horizontal space
        """
        self.gs.update(**kwargs)

    def set_cmap(self, **kwargs):
        """
        Choose the colormap for the axis
        ax_idx : Int. the index of the axis. default is 0
        vrange : Tuple. (Vmin, Vmax). The range for the colormap. default is (0, 1)
        cmap : String. the name of colormap default is 'jet'
        """
        ax_idxs = self._return_default(kwargs, 'ax_idx', [0])
        vranges = self._return_default(kwargs, 'vrange', [(0, 1)])
        for ax_idx, vrange in zip(ax_idxs, vranges):
            ai = self._axis_info[ax_idx]
            cmap = self._return_default(kwargs, 'cmap', 'viridis')
            cmapinfo = CmapInfo(cm.get_cmap(cmap), vrange)
            self._axis_info[ax_idx] = ai._replace(cmapinfo=cmapinfo)
        return

    def set_lim(self, **kwargs):
        """
        Set the x,y limit for the axis
        ax_idx : Int. the index of the axis. default is 0
        xr : Tuple. the range of the x-axis
        yr : Tuple. the range of teh y-axis
        """
        ax_idxs = self._return_default(kwargs, 'ax_idx', [0])
        for ax_idx in ax_idxs:
            axis_info = self._axis_info[ax_idx]
            ax = axis_info.ax

            xr = self._return_default(kwargs, 'xr', ax.set_xlim())
            yr = self._return_default(kwargs, 'yr', ax.set_ylim())
            ax.set_xlim(xr[0], xr[1])
            ax.set_ylim(yr[0], yr[1])
        return 
    
    def set_axis(self, **kwargs):
        """
        ax_idx : Int. the index of the axis. default is 0
        x/y ticks : array-like. the list of xticks locations
        x/y labels : array-like. the labels to place at the give ticks locations
        x/y position : String. the position of axis. default is bottom/left
        """
        ax_idxs = self._return_default(kwargs, 'ax_idx', [0])
        for ax_idx in ax_idxs:
            axis_info = self._axis_info[ax_idx]
            ax = axis_info.ax
            
            keys = kwargs.keys()
            
            if 'xticks' in keys:
                xticks = kwargs['xticks']
                xlabel = self._return_default(kwargs, 'xlabel', xticks)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabel)
                if 'xposition' in keys:
                    ax.xaxis.ticks_top()
                    ax.axis.set_label_position(kwargs['xposition'])
            
            if 'yticks' in keys:
                yticks = kwargs['yticks']
                ylabel = self._return_default(kwargs, 'ylabel', yticks)
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabel)
                if 'yposition' in keys:
                    ax.yaxis.ticks_right()
                    ax.axis.set_label_position(kwargs['yposition'])
        return

    def set_grid(self, ax_idx = 0, which='major', axis='both', **kwargs):
        """
        ax_idx : Int. the index of the axis. default is 0
        which : {'major', 'minor', 'both'}. the gird lines to apply. default is 'major'
        axis : {'both', 'x', 'y'}. the axis to apply.
        """
        axis_info = self._axis_info[ax_idx]
        ax = axis_info.ax
        ax.grid(which=which, axis=axis, **kwargs)
        return
    
    def set_title(self, label, loc='center', ax_idx=0, **kwargs):
        """
        label : String. Text for title
        loc : {'center', 'left', 'right'}. location of the titles
        ax_idx : Int. the index of the axis. default is 0
        """
        axis_info = self._axis_info[ax_idx]
        ax = axis_info.ax
        ax.set_title(label, loc=loc)
        return

    def legend(self, **kwargs):
        """
        ax_idx : Int. the index of the axis. default is 0
        loc : Int/String. the location of the legend. default is 'best'
        """
        ax_idxs = self._return_default(kwargs, 'ax_idx', [0])
        for ax_idx in ax_idxs:
            axis_info = self._axis_info[ax_idx]
            ax = axis_info.ax
            loc = self._return_default(kwargs, 'loc', 'best')
            legend_kwargs = {
                'loc' : loc,
                'fontsize' : self._fontsize,
                'framealpha' : 0.9,
                'edgecolor' : 'w',
                'shadow' : True,
                'labelspacing' : 0.1,
            }
            ax.legend(**legend_kwargs)
        return
        
    def plot(self, *args, **kwargs):
        """
        ax_idx : Int. the index of the axis. default is 0
        cidx : Int/List. the value for color
        """
        keys = kwargs.keys()
        ax_idx = kwargs['ax_idx']
        kwargs.pop('ax_idx')

        ai = self._axis_info[ax_idx]
        ax = ai.ax
        ci = ai.cmapinfo

        if 'cidx' in keys:
            cidx = kwargs['cidx']
            kwargs.pop('cidx')
        else:
            cidx = ai.nplt

        norm = Normalize(vmin=ci.vr[0] , vmax = ci.vr[1])
        m = cm.ScalarMappable(norm=norm, cmap=ci.cmap)
        ax.plot(*args, c= m.to_rgba(cidx), **kwargs)
        self._axis_info[ax_idx] = ai._replace(nplt = ai.nplt+1)
        return 

    def scatter(self, *args, **kwargs):
        """
        ax_idx : Int. the index of the axis. default is 0
        cidx : Int/List. the value for color
        """
        keys = kwargs.keys()
        ax_idx = kwargs['ax_idx']
        kwargs.pop('ax_idx')

        ai = self._axis_info[ax_idx]
        ax = ai.ax
        ci = ai.cmapinfo

        if 'cidx' in keys:
            cidx = kwargs['cidx']
            kwargs.pop('cidx')
        else:
            cidx = [ai.nsct for _ in range(len(args[0]))]

        norm = Normalize(vmin=ci.vr[0] , vmax = ci.vr[1])
        m = cm.ScalarMappable(norm=norm, cmap=ci.cmap)

        #marker = kwargs['marker'] if 'marker' in keys else 
        ax.scatter(*args, c= m.to_rgba(cidx), **kwargs)
        self._axis_info[ax_idx] = ai._replace(nsct = ai.nsct+1)
    
