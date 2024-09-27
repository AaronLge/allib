# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:03:31 2024

@author: aaron.lange
"""

import os
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import sys
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd

path = r"C:\\temp\\python_self_crated\\packages"
sys.path.insert(0, path)
from allib import general as gl


# %% classes

class Line:
    def __init__(self, x, y, label=None, color=None, linewidth=None, linestyle=None, yy_side='left', spinecolor=None):
        self.x = x
        self.y = y
        self.label = label
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.yy_side = yy_side
        self.spinecolor = spinecolor


class Bar:
    def __init__(self, x, y, label=None, color=None, yy_side='left', width=1, bottom=None, align='center', zorder=None, alpha=1.0, spinecolor=None):
        if color is None:
            color = [0, 0, 0]
        self.x = x
        self.y = y
        self.label = label
        self.color = color
        self.yy_side = yy_side
        self.width = width
        self.bottom = bottom
        self.align = align
        self.zorder = zorder
        self.alpha = alpha
        self.spinecolor = spinecolor


class Scatter:
    def __init__(self, x, y, cmap=None, color='black', cmap_mode='density', c=None, alpha=1, size=None, label=None, cbar=None, cbar_label=None, yy_side='left', spinecolor=None, cmap_norm=None, cbar_label_fontsize=None):
        self.x = x
        self.y = y
        self.color = color
        self.cmap = cmap
        self.cmap_mode = cmap_mode
        self.c = c
        self.alpha = alpha
        self.size = size
        self.label = label
        self.cbar = cbar
        self.cbar_label = cbar_label
        self.yy_side = yy_side
        self.spinecolor = spinecolor
        self.cmap_norm = cmap_norm
        self.cbar_label_fontsize = cbar_label_fontsize


class Tile:
    def __init__(self, num, errorbar=None, bar=None, textbox=None, lines=None, scatter=None, title=None, x_label=None, y_label=None, grid=None, x_lim=(None, None),
                 y_lim=(None, None), x_norm='lin', y_norm='lin', spinecolor_left=None, spinecolor_right=None, y_label_right=None, legend='auto'):
        if lines is None:
            lines = []
        if scatter is None:
            scatter = []
        if textbox is None:
            textbox = []
        if errorbar is None:
            errorbar = []
        if bar is None:
            bar = []
        self.errorbar = errorbar
        self.lines = lines
        self.scatter = scatter
        self.num = num
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.grid = grid
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.textbox = textbox
        self.spinecolor_right = spinecolor_right
        self.spinecolor_left = spinecolor_left
        self.bar = bar
        self.y_label_right = y_label_right
        self.legend = legend

    def add_line(self, line):
        if self.lines is None:
            self.lines = []
        self.lines.append(line)

    def add_scatter(self, scatter):
        if self.scatter is None:
            self.scatter = []
        self.scatter.append(scatter)

    def add_textbox(self, textbox):
        if self.textbox is None:
            self.textbox = []
        self.textbox.append(textbox)

    def add_errorbar(self, errorbar):
        if self.errorbar is None:
            self.errorbar = []
        self.errorbar.append(errorbar)

    def add_bar(self, bar):
        if self.bar is None:
            self.bar = []
        self.bar.append(bar)


class PolarTile:
    def __init__(self, num, RoseBar=None, grid=None, title=None):
        self.num = num
        self.RoseBar = RoseBar
        if RoseBar is None:
            self.RoseBar = []
        self.title = title
        self.grid = grid

    def add_RoseBar(self, RoseBar):
        if self.RoseBar is None:
            self.RoseBar = []
        self.RoseBar.append(RoseBar)


class RoseBar:
    def __init__(self, angles=None, radial=None, cmap='cool', r_bins=None,radial_data=None, radial_mode='summed', radial_datatype='percent', cbar=True, cbar_label=None, r_max=None):
        self.angles = angles
        self.radial = radial
        self.cmap = cmap
        self.r_bins = r_bins
        self.radial_data = radial_data
        self.radial_mode = radial_mode
        self.radial_datatype = radial_datatype
        self.cbar = cbar
        self.cbar_label = cbar_label
        self.r_max = r_max


class Textbox:
    def __init__(self, data=None, fontsize=None, fontweight='bold', corner1=None, corner2=None, colors=None, orientation_v='center', orientation_h='center', header=True):
        self.data = data
        self.fontsize = fontsize
        self.fontweight = fontweight
        self.corner1 = corner1
        self.corner2 = corner2
        self.colors = colors
        self.orientation_v = orientation_v
        self.orientation_h = orientation_h
        self.header = header


class ErrorBar:
    def __init__(self, x=None, y=None, errorlims=None, color=None, fmt=None, yy_side='left'):
        self.x = x
        self.y = y
        self.errorlims = errorlims
        self.color = color
        self.fmt = fmt
        self.yy_side = yy_side

# %% functions
def plot_tiled(Tiles, **kwargs):

    plt.rc('text', usetex=True)
    figsize = kwargs.get('figsize', (8.268, 11.693))
    global_max = kwargs.get('global_max', ['auto', 'auto'])
    global_min = kwargs.get('global_min', ['auto', 'auto'])
    fontsize_title = kwargs.get('fontsize_title', 10)
    fontsize_legend = kwargs.get('fontsize_legend', 7)
    fontsize_label = kwargs.get('fontsize_label', 8)
    fontsize_ticks = kwargs.get('fontsize_ticks', 6)
    grid = kwargs.get('grid', [3, 2])
    scatter_max = kwargs.get('scatter_max', 'auto')
    scatter_min = kwargs.get('scatter_min', 'auto')

    FIG = []
    i = 0

    N_exp = len(Tiles)
    tiles_page = grid[0] * grid[1]
    pages = int(np.ceil(N_exp / tiles_page))

    isPolar = [type(Tile).__name__ == 'PolarTile' for Tile in Tiles]
    isKartasian = [type(Tile).__name__ == 'Tile' for Tile in Tiles]

    def find_Tile_lims(Tiles):
        x_min = None
        x_max = None
        y_min = None
        y_max = None
        c_min = None
        c_max = None

        for n, Tile in enumerate(Tiles):
            if isKartasian[n]:

                for scatter in Tile.scatter:
                    if scatter.c is not None:
                        if gl.compare_values(np.nanmax(scatter.c), c_max, '>'):
                            c_max = np.nanmax(scatter.c)
                        if gl.compare_values(np.nanmax(scatter.c), c_min, '<'):
                            c_min = np.nanmin(scatter.c)

                    if gl.compare_values(np.nanmax(scatter.x), x_max, '>'):
                        x_max = np.nanmax(scatter.x)
                    if gl.compare_values(np.nanmin(scatter.x), x_min, '<'):
                        x_min = np.nanmin(scatter.x)
                    if gl.compare_values(np.nanmax(scatter.y), y_max, '>'):
                        y_max = np.nanmax(scatter.y)
                    if gl.compare_values(np.nanmin(scatter.y), y_min, '<'):
                        y_min = np.nanmin(scatter.y)

                for line in Tile.lines:
                    if line.x is not None:
                        if gl.compare_values(np.nanmax(line.x), x_max, '>'):
                            x_max = np.nanmax(line.x)
                        if gl.compare_values(np.nanmin(line.x), x_min, '<'):
                            x_min = np.nanmin(line.x)

                    if line.y is not None:
                        if gl.compare_values(np.nanmax(line.y), y_max, '>'):
                            y_max = np.nanmax(line.y)
                        if gl.compare_values(np.nanmin(line.y), y_min, '<'):
                            y_min = np.nanmin(line.y)

                for bar in Tile.bar:
                    if bar.x is not None:
                        if gl.compare_values(np.nanmax(bar.x), x_max, '>'):
                            x_max = np.nanmax(bar.x)
                        if gl.compare_values(np.nanmin(bar.x), x_min, '<'):
                            x_min = np.nanmin(bar.x)

                    if bar.y is not None:
                        if gl.compare_values(np.nanmax(bar.y), y_max, '>'):
                            y_max = np.nanmax(bar.y)
                        if gl.compare_values(np.nanmin(bar.y), y_min, '<'):
                            y_min = np.nanmin(bar.y)


        return (x_min, x_max), (y_min, y_max), (c_min, c_max)

    (x_min_auto, x_max_auto), (y_min_auto, y_max_auto), (c_min_auto, c_max_auto) = find_Tile_lims(Tiles)

    if global_max[0] == 'auto':
        x_max = x_max_auto
    elif global_max[0] is None:
        x_max = None
    else:
        x_max = global_max[0]

    if global_min[0] == 'auto':
        x_min = x_min_auto
    elif global_min[0] is None:
        x_min = None
    else:
        x_min = global_min[0]

    if global_max[1] == 'auto':
        if y_max_auto is not None:
            y_max = 1.1*y_max_auto
        else:
            y_max = None
    elif global_max[1] is None:
        y_max = None
    else:
        y_max = global_max[1]

    if global_min[1] == 'auto':
        y_min = y_min_auto
    elif global_min[1] is None:
        y_min = None
    else:
        y_min = global_min[1]

    if scatter_min == 'auto':
        c_min = c_min_auto
    elif scatter_min is None:
        c_min = None
    else:
        c_min = scatter_min

    if scatter_max == 'auto':
        c_max = c_max_auto
    elif scatter_max is None:
        c_max = None
    else:
        c_max = scatter_max

    # iteration pages
    for page, _ in enumerate(range(pages)):

        fig, ax = plt.subplots(grid[0], grid[1], figsize=figsize)

        if tiles_page != 1:
            ax_flat = ax.flatten()
        else:
            ax_flat = [ax]

        # iteration tile
        for i_page, _ in enumerate(range(tiles_page)):

            axis = ax_flat[i_page]

            # if section exists
            if i < N_exp:

                Tile = Tiles[i]

                ## KARTASIAN:
                if isKartasian[i]:

                    # JBO-Logo
                    with matplotlib.cbook.get_sample_data(path + '\\allib\\' + 'JBO_logo.png') as file:
                        image_bgr = plt.imread(file, format='png')

                    if Tile.x_norm != 'lin':
                        axis.set_xscale(Tile.x_norm)
                    if Tile.y_norm != 'lin':
                        axis.set_yscale(Tile.y_norm)

                    axin = axis.inset_axes([0.05, 0.7, 0.5, 0.3], zorder=-1)
                    axin.imshow(image_bgr, zorder=-1)
                    axin.axis('off')

                    # grid
                    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)
                    # SCATTER
                    for scatter in Tile.scatter:

                        ax_yy = get_yyaxis(axis, side=scatter.yy_side, color=scatter.spinecolor)

                        alpha = scatter.alpha
                        if scatter.cmap is not None:

                            if scatter.cmap_mode == 'density':
                                c = gl.c_scatterplot(scatter.x, scatter.y)
                                alpha = 0.5

                                if scatter.cmap_norm == 'sqrt':
                                    c = np.sqrt(c)

                            if scatter.cmap_mode == 'manual':
                                c = scatter.c

                            cmap = matplotlib.colormaps[scatter.cmap]
                            cmap.set_bad(color="grey")

                            ax_yy.scatter(scatter.x,
                                          scatter.y,
                                          cmap=cmap,
                                          c=c,
                                          alpha=alpha,
                                          s=scatter.size,
                                          label=scatter.label,
                                          vmax=c_max,
                                          vmin=c_min,
                                          plotnonfinite=True)

                            # colorbar
                            if scatter.cbar is not None:

                                if c_max is None:
                                    cbar_max = np.nanmax(c)
                                else:
                                    cbar_max = c_max

                                if c_min is None:
                                    cbar_min = np.nanmin(c)
                                else:
                                    cbar_min = c_min

                                cmappable = ScalarMappable(
                                    Normalize(cbar_min, cbar_max), cmap=cmap)

                                cbar = plt.colorbar(mappable=cmappable, ax=ax_yy)

                                for tick_label in cbar.ax.yaxis.get_ticklabels():
                                    tick_label.set_fontsize(fontsize_ticks)

                                offset_text = cbar.ax.yaxis.get_offset_text()
                                if offset_text.get_text():  # If there is an offset (i.e., text is not empty)
                                    offset_text.set_fontsize(fontsize_ticks)

                                if scatter.cbar_label is not None:
                                    if scatter.cbar_label_fontsize is None:
                                        cbar_label_fontsize = fontsize_label
                                    else:
                                        cbar_label_fontsize = scatter.cbar_label_fontsize

                                    cbar.set_label(scatter.cbar_label, fontsize=cbar_label_fontsize)

                        else:
                            color = scatter.color

                            ax_yy.scatter(scatter.x,
                                          scatter.y,
                                          color=color,
                                          label=scatter.label)

                    # BAR
                    for bar in Tile.bar:

                        ax_yy = get_yyaxis(axis, side=bar.yy_side, color=bar.spinecolor)

                        ax_yy.bar(bar.x,
                                  bar.y,
                                  width=bar.width,
                                  bottom=bar.bottom,
                                  align=bar.align,
                                  color=bar.color,
                                  zorder=bar.zorder,
                                  alpha=bar.alpha,
                                  label=bar.label)

                    # LINES
                    for line in Tile.lines:

                        ax_yy = get_yyaxis(axis, side=line.yy_side, color=line.spinecolor)

                        if line.x is None:
                            ax_yy.axhline(line.y[0],
                                          linestyle=line.linestyle,
                                          linewidth=line.linewidth,
                                          color=line.color,
                                          label=line.label)

                        elif line.y is None:
                            ax_yy.axvline(line.x[0],
                                          linestyle=line.linestyle,
                                          linewidth=line.linewidth,
                                          color=line.color,
                                          label=line.label)

                        else:
                            ax_yy.plot(line.x,
                                       line.y,
                                       linestyle=line.linestyle,
                                       linewidth=line.linewidth,
                                       color=line.color,
                                       label=line.label)

                    # ERRORBAR
                    for errorbar in Tile.errorbar:

                        ax_yy = get_yyaxis(axis, side=errorbar.yy_side)

                        ax_yy.errorbar(errorbar.x, errorbar.y, errorbar.errorlims, fmt=errorbar.fmt,
                                       ecolor=errorbar.color)

                    # TEXTBOX
                    for text in Tile.textbox:
                        axis = get_yyaxis(axis, side='left')
                        fig, axis = plot_dataframe(text.data,
                                                   plot=[fig, axis],
                                                   corner1=text.corner1,
                                                   corner2=text.corner2,
                                                   fontsize=text.fontsize,
                                                   colors=text.colors,
                                                   orientation_v=text.orientation_v,
                                                   orientation_h=text.orientation_h,
                                                   header=text.header
                                                   )

                    if x_min is not None: axis.set_xlim(left=x_min)
                    if x_max is not None: axis.set_xlim(right=x_max)
                    if y_min is not None: axis.set_ylim(bottom=y_min)
                    if y_max is not None: axis.set_ylim(top=y_max)

                    if x_min is None and x_max is None:
                        axis.autoscale(enable=True, axis='x', tight=True)

                    if y_min is None and y_max is None:
                        axis.autoscale(enable=True, axis='y', tight=True)

                    if Tile.x_label is not None:
                        axis.set_xlabel(Tile.x_label, fontsize=fontsize_label)
                    if Tile.y_label is not None:
                        axis.set_ylabel(Tile.y_label, fontsize=fontsize_label)

                    # change global axis colors if specified
                    update_axis_colors_ticksize_axlabels(axis,
                                                         color_left=Tile.spinecolor_left,
                                                         color_right=Tile.spinecolor_right,
                                                         ticksize_left=fontsize_ticks,
                                                         ticksize_right=fontsize_ticks,
                                                         label_left=Tile.y_label,
                                                         label_right=Tile.y_label_right,
                                                         label_size_left=fontsize_label,
                                                         label_size_right=fontsize_label,
                                                         sci_label_size_left=fontsize_ticks,
                                                         sci_label_size_right=fontsize_ticks)

                    # legend
                    _, labels = axis.get_legend_handles_labels()
                    if Tile.legend == 'auto' and labels:
                        axis.legend(loc="lower right", fontsize=fontsize_legend)

                    if type(Tile.legend) is list:
                        dummy = []
                        for dummy_legend in Tile.legend:
                            temp, = plt.plot([], [], color=dummy_legend["color"], label=dummy_legend["label"], linestyle=dummy_legend["linestyle"])
                            dummy.append(temp)

                        axis.legend(handles=dummy, fontsize=fontsize_legend, loc="lower right")

                ## POLAR
                if isPolar[i]:

                    axis.remove()  # Remove the Cartesian axis
                    axis = fig.add_subplot(grid[0], grid[1], i_page+1, polar=True)

                    # JBO-Logo    
                    with matplotlib.cbook.get_sample_data(path + '\\allib\\' + 'JBO_logo.png') as file:
                        image_bgr = plt.imread(file, format='png')

                    axin = axis.inset_axes([0.2, 0.6, 0.6, 0.4], zorder=-1)
                    axin.imshow(image_bgr, zorder=-1)
                    axin.axis('off')

                    for RoseBar in Tile.RoseBar:
                        fig, axis = plot_rosebar(RoseBar.radial_data,
                                                 RoseBar.r_bins,
                                                 RoseBar.angles,
                                                 r_max=RoseBar.r_max,
                                                 plot=[fig, axis],
                                                 figsize=None,
                                                 cmap=RoseBar.cmap,
                                                 radial_mode=RoseBar.radial_mode,
                                                 radial_datatype=RoseBar.radial_datatype,
                                                 cbar_label=RoseBar.cbar_label,
                                                 cbar=RoseBar.cbar)

                # title
                if Tile.title is not None: axis.set_title(Tile.title, fontsize=fontsize_title)

            # if no section exists, crate dummy plot for correct spacing and padding
            else:

                axis.spines['top'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.spines['right'].set_visible(False)

                axis.tick_params(axis=u'both', which=u'both', length=0)
                axis.set_xticklabels({})
                axis.set_yticklabels({})
                axis.set_title(' \n ')
                axis.set_xlabel(' ')
                axis.set_ylabel(' ')

            i = i + 1

        fig.tight_layout()

        FIG.append(fig)

    return FIG


def plot_rosebar(radial_data, r_bins, angles, r_max=None, plot=None, figsize=None, cmap='cool', radial_mode='summed', radial_datatype='percent', cbar_label=None, cbar=True):
    """plots roseplot to handels given in plot = [fig, ax], if in standalone mode, creates a new figure and figsize is required

    inputs:
    radial_data: N_angles lists of lists of length len(r_bins)-1, containing the data to be plotted in each bar, for the relevant angle segment, last entry is plotted first
                    if radial_mode='summed' (default), each bar bar[i] will be displayed as the value from sum(bar[0:i])
    r_bins: list, edges of the radial bin, are displayed as ticks in colorbar
    angles: list, midpoints of the angle segents, hast to be the same length as radial_data

    optional:
    r_max: float, maximum the radial data, displayed as green line in barchart
    plot: [fig, ax]: handles to plot to,
    figsize: (width, height): if plot argument, obiligatory
    cmap: colormap to use
    radial_mode: string, 'summed' (default) or None, than values are takes as is
    radial_datatype: string, 'percent' (default), % symbol is displayed ar radial ticks
    cbar_label: string, label for colorbar
    cbar: bool, toogle cbar

    returns:
    fig, ax
    """

    if plot is None:

        fig, axis = plt.subplots(grid[0], grid[1], figsize=figsize, polar=True)
        # JBO-Logo
        with matplotlib.cbook.get_sample_data(path + '\\allib\\' + 'JBO_logo.png') as file:
            image_bgr = plt.imread(file, format='png')

        axin = axis.inset_axes([0.2, 0.6, 0.6, 0.4], zorder=-1)
        axin.imshow(image_bgr, zorder=-1)
        axin.axis('off')

    else:
        fig = plot[0]
        axis = plot[1]

    N_r_bins = len(r_bins) - 1
    cmap = matplotlib.pyplot.get_cmap(cmap, N_r_bins)
    angle_width = 2 * np.pi / len(angles)
    theta = [2 * np.pi * float(curr) / 360 for curr in angles]

    for j in range(N_r_bins):

        if radial_mode == 'summed':
            r = [np.sum(r_angle_segment[0:N_r_bins - j])
                 for r_angle_segment in radial_data]

        plt.bar(theta, r, width=angle_width, color=cmap(N_r_bins - j), alpha=1)

    # theta ticks
    axis.set_xticks(theta)
    x_ticklabels = [str(round(label * 360 / (2 * np.pi), 2)
                        ) + ' °' for label in theta]
    axis.set_xticklabels(x_ticklabels, fontsize=7)
    # cbar
    if cbar:
        bar_ticks = r_bins

        norm = BoundaryNorm(bar_ticks, cmap.N)

        cmappable = ScalarMappable(norm, cmap=cmap)
        cmappable.set_array([])
        cbar = plt.colorbar(cmappable,
                            ax=axis,
                            pad=0.15,
                            shrink=0.9,
                            spacing='proportional')

        cbar.set_ticks(ticks=bar_ticks, labels=[f"{tick:.15g}" for tick in r_bins], fontsize=7)

        # add max value label
        if r_max is not None:
            cbar.ax.axhline(y=r_max, color=[0, 1, 0], linewidth=2)
            current_ticks = cbar.get_ticks()
            current_labels = [label.get_text() for label in cbar.ax.get_yticklabels()]

            # Add new tick
            new_ticks = np.append(current_ticks, r_max)
            new_labels = current_labels + ['max']

            # Sort the ticks and labels
            sorted_indices = np.argsort(new_ticks)
            sorted_ticks = new_ticks[sorted_indices]
            sorted_labels = [new_labels[i] for i in sorted_indices]

            # Set the new ticks and labels
            cbar.set_ticks(sorted_ticks)
            cbar.set_ticklabels(sorted_labels)

            for label, tick in zip(cbar.ax.get_yticklabels(), cbar.get_ticks()):
                if tick == r_max:
                    label.set_color([0, 1, 0])

        for tick_label in cbar.ax.yaxis.get_ticklabels():
            tick_label.set_fontsize(7)

        # r ticks
        y_ticklables = axis.get_yticklabels()
        y_tickpostions = axis.get_yticks()
        if radial_datatype == 'percent':
            y_ticklables = [label.get_text()[:-1] + r' \%' + '$' for label in y_ticklables]
        else:
            y_ticklables = [label.get_text() for label in y_ticklables]

        axis.set_yticks(y_tickpostions)
        axis.set_yticklabels(y_ticklables, fontsize=6)

        cbar.set_label(cbar_label, fontsize=8)
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)

    return fig, axis


def table(data, **kwargs):
    """The table function creates a table with the JBO-Design

    Parameters:
    data: A 2D array-like structure (list of lists, numpy array, etc.) containing the data to be displayed in the table.

    optional:

    collabels: List of strings for column headers.
    rowlabels: List of strings for row headers.
    grey: A boolean 2D array with the same shape as data, used to set text color to grey for specific cells.
    titel: A string to be displayed as the table's title.
    row_label_name: A string or list to be displayed as the title for the row labels.
    formater: A function or lambda used to format the data before displaying it.
    fontsize: Integer specifying the font size of the table text.
    figsize: List of two numbers specifying the figure size in inches.

    return:
    Figure
    """

    col_labels = kwargs.get('collabels', None)
    row_labels = kwargs.get('rowlabels', None)
    auto_width = kwargs.get('auto_width', True)
    grey = kwargs.get('grey', None)
    titel = kwargs.get('titel', None)
    row_label_name = kwargs.get('row_label_name', None)
    formater = kwargs.get('formater', None)
    fontsize = kwargs.get('fontsize', 8)
    data = np.array(data)
    nans = kwargs.get('nan', '')
    null = kwargs.get('null', '0')
    heatmap = kwargs.get('heatmap', False)
    cmap_heatmap = kwargs.get('cmap_heatmap', 'Blues')
    figsize = kwargs.get('figsize', (8.268, 11.693))
    datatype = kwargs.get('datatype', None)


    try:
        max_data = np.max(data)
    except:
        max_data = None

    if formater is not None:
        data = gl.format_array(data, formater)

    data = np.array(data)

    # Cast the array to float
    if datatype is 'float':
        data = data.astype(float)
    if datatype is 'str':
        data = data.astype(str)

    CELLS = data

    # convert row labels and add offset row_offset
    row_offset = 0
    if row_labels is not None:
        row_labels = [str(i) for i in row_labels]
        row_labels = np.array(row_labels)
        row_labels = row_labels.reshape(-1, 1)
        row_offset = 1

    # convert col labels and add col_offset
    col_offset = 0
    if col_labels is not None:
        col_labels = [str(i) for i in col_labels]
        col_labels = np.array(col_labels)
        col_labels = col_labels.reshape(1, -1)
        col_offset = 1

    # crate full CELL matix with optional col_labels und row_labels
    if col_labels is not None:

        if row_labels is not None:

            if row_label_name is not None:
                col_labels = np.hstack((np.array(row_label_name).reshape(1, -1), col_labels))
            else:
                col_labels = np.hstack((np.array('').reshape(1, -1), col_labels))

            CELLS = np.hstack((row_labels, CELLS))

        CELLS = np.vstack((col_labels, CELLS))

    else:
        # gibt es row labels?
        if row_labels is not None:
            CELLS = np.hstack((row_labels, CELLS))

    # A4 size in inches (landscape)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    table = ax.table(cellText=CELLS, bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)

    table.set_fontsize(fontsize)

    table.auto_set_column_width(col=0)

    def is_convertible_to_float(value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    # turn font color grey if grey==True and remove 'nans'
    for i in range(row_offset, CELLS.shape[0]):
        for j in range(col_offset, CELLS.shape[1]):
            if grey is not None:
                if grey[i-row_offset, j-col_offset]:
                    table[(i, j)].set_text_props(color=[0.5, 0.5, 0.5])
            if is_convertible_to_float(CELLS[i, j]):
                if np.isnan(float(CELLS[i, j])):
                    if nans is not None:
                        table[(i, j)].get_text().set_text(nans)
                if float(CELLS[i, j]) == 0:
                    if null is not None:
                        table[(i, j)].get_text().set_text(null)

    # Alternate background color for every second row
    for i, cell in enumerate(table.get_celld().values()):
        if i % (2 * CELLS.shape[1]) >= CELLS.shape[1]:  # The row index within each pair of rows
            cell.set_facecolor('#f0f0f0')  # Grey background color

    # Apply heatmap to the cells
    cmap = plt.cm.get_cmap(cmap_heatmap)
    if heatmap:
        for i in range(row_offset, CELLS.shape[0]):
            for j in range(col_offset, CELLS.shape[1]):
                norm = plt.Normalize(0, max_data * 1.2)
                value = float(table[i, j].get_text().get_text())
                table[(i, j)].set_facecolor(cmap(norm(value)))

    # Center the text vertically and horizontally in each cell
    for key, cell in table.get_celld().items():
        cell.set_text_props(verticalalignment='center',
                            horizontalalignment='center')

    if row_labels is not None:
        for j in range(row_labels.shape[0]):
            cell = table[(j+row_offset, 0)]
           # cell.set_text_props(fontweight='bold')

    if col_labels is not None:
        for j in range(col_labels.shape[1]):
            cell = table[(0, j)]
            cell.get_text().set_text(col_labels[0, j])
            cell.set_text_props(color='white')
            cell.set_facecolor('#008f85')  # Blue background color

    if titel is not None:
        plt.rc('text', usetex=True)
        y_range = ax.get_ylim()[1]
        ax.text(0.5, 1.05 * y_range, titel, horizontalalignment='center', verticalalignment='center')

    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_dataframe(data, header=True, plot=None, corner1=(0.1, 0.9), corner2=(0.9, 0.1), fontsize=12, colors=None, orientation_v='center', orientation_h='center'):
    """
    Plots a DataFrame or list of lists on a Matplotlib axis using ax.text() with coordinates specified by two corners,
    relative to the current axis limits, and handles logarithmic scaling.

    Parameters:
    - data (pd.DataFrame or list of lists): The data to plot.
    - ax (matplotlib.axes._subplots.AxesSubplot, optional): The Matplotlib axis to plot on.
                                                            If None, a new one will be created.
    - corner1 (tuple): The relative coordinates (x1, y1) of the first corner (top-left) in axis-relative coordinates.
    - corner2 (tuple): The relative coordinates (x2, y2) of the opposite corner (bottom-right) in axis-relative coordinates.
    - fontsize (int, optional): Font size of the text. Default is 12.

    Returns:
    - ax: The Matplotlib axis with the data plotted on it.
    """

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig = plot[0]
        ax = plot[1]

    x1 = corner1[0]
    y1 = corner1[1]
    x2 = corner2[0]
    y2 = corner2[1]

    # Calculate plot width and height based on actual coordinates
    plot_width = x2 - x1
    plot_height = y1 - y2

    #toggle_orienation offset
    if orientation_h == 'left':
        toggle_offset = 0
    elif orientation_h == 'right':
        toggle_offset = 2
    else:
        toggle_offset = 1

    # Convert list of lists to DataFrame if necessary
    if isinstance(data, list):
        if header:
        # Assume the first row is the header
            header = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=header)
        else:
            df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("Data must be a Pandas DataFrame or a list of lists.")

    # Calculate column width and row height in data units
    col_width = plot_width / df.shape[1]
    row_height = plot_height / df.shape[0]

    if header:
    # Draw the column headers
        for i, col in enumerate(df.columns):
            ax.text(x1 + i * col_width + toggle_offset* col_width / 2, y1, col, weight='bold', fontsize=fontsize, ha=orientation_h, va=orientation_v, transform=ax.transAxes)

    # Draw the data cells
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if colors is not None:
                ax.text(x1 + col * col_width + toggle_offset * col_width / 2, y1 - (row + 1) * row_height, str(df.iat[row, col]),
                        fontsize=fontsize, ha=orientation_h, va=orientation_v, transform=ax.transAxes, color=colors.iat[row, col])
            else:
                ax.text(x1 + col * col_width + toggle_offset * col_width / 2, y1 - (row + 1) * row_height, str(df.iat[row, col]),
                        fontsize=fontsize, ha=orientation_h, va=orientation_v, transform=ax.transAxes)
    return fig, ax


def get_yyaxis(ax, side='right', color=None):
    """
    Returns the y-axis on the specified side ('left' or 'right') associated with the given axis (ax).
    If the specified axis does not exist, it will create one using twinx() for the 'right' side or
    use the primary axis for the 'left' side. Optionally changes the color of the y-axis on the selected side.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The primary axis for which to find or create a y-axis on the specified side.
    side : str, optional
        The side of the y-axis to return ('left' or 'right'). Default is 'right'.
    color : str or None, optional
        The color to apply to the specified y-axis spine, ticks, and labels. If None, no color is applied. Default is None.

    Returns:
    --------
    side_ax : matplotlib.axes.Axes
        The axis on the specified side corresponding to the given axis.
    """
    if side not in ['left', 'right']:
        raise ValueError("Invalid value for 'side'. Expected 'left' or 'right'.")

    if side == 'left':
        # If 'left', return the original axis (primary axis)
        side_ax = ax
        if color:
            side_ax.spines['left'].set_visible(True)
            side_ax.spines['left'].set_color(color)
            side_ax.yaxis.label.set_color(color)
            side_ax.tick_params(axis='y', colors=color)
    else:
        # For 'right', check if a secondary axis already exists
        for child in ax.figure.get_axes():
            if child != ax and ax.get_shared_x_axes().joined(ax, child):
                side_ax = child
                break
        else:
            # If no secondary axis exists on the right, create it
            side_ax = ax.twinx()

        if color:
            side_ax.spines['right'].set_color(color)
            side_ax.yaxis.label.set_color(color)
            side_ax.tick_params(axis='y', colors=color)

    return side_ax


def update_axis_colors_ticksize_axlabels(
    ax,
    color_left=None,
    color_right=None,
    ticksize_left=None,
    ticksize_right=None,
    label_left=None,
    label_right=None,
    label_size_left=None,
    label_size_right=None,
    sci_label_size_left=None,  # New parameter for scientific notation label size
    sci_label_size_right=None   # New parameter for scientific notation label size
):
    """
    Updates the colors of the left and/or right y-axis spines, ticks, and labels.
    If the axis on one side does not exist, it will not modify or create a new one.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The primary axis whose y-axis colors will be updated.
    color_left : str or None, optional
        The color to apply to the left y-axis spine, ticks, and labels. If None, the left axis is not modified. Default is None.
    color_right : str or None, optional
        The color to apply to the right y-axis spine, ticks, and labels. If None, the right axis is not modified. Default is None.
    ticksize_left : int or None, optional
        The ticksize to apply to the left y-axis spine, ticks, and labels. If None, the left axis is not modified. Default is None.
    ticksize_right : int or None, optional
        The ticksize to apply to the right y-axis spine, ticks, and labels. If None, the right axis is not modified. Default is None.
    label_left : str or None, optional
        The label for the left y-axis. Default is None (no label).
    label_right : str or None, optional
        The label for the right y-axis. Default is None (no label).
    label_size_left : int or None, optional
        Font size for the left y-axis label. Default is None (use current size).
    label_size_right : int or None, optional
        Font size for the right y-axis label. Default is None (use current size).
    sci_label_size_left : int or None, optional
        Font size for scientific notation labels on the left y-axis. Default is None (use current size).
    sci_label_size_right : int or None, optional
        Font size for scientific notation labels on the right y-axis. Default is None (use current size).

    Returns:
    --------
    None
    """
    # Update left axis color if specified
    if color_left:
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color(color_left)
        ax.yaxis.label.set_color(color_left)
        ax.tick_params(axis='y', colors=color_left)

    if label_left:
        # Set the y-axis label with appropriate parameters
        if color_left:
            ax.set_ylabel(label_left, fontsize=label_size_left, color=color_left)
        else:
            ax.set_ylabel(label_left, fontsize=label_size_left)  # Omit color parameter if None

    if ticksize_left:
        ax.tick_params(axis='both', which='major', labelsize=ticksize_left)
        offset_text = ax.yaxis.get_offset_text()
        if offset_text.get_text():
            offset_text.set_fontsize(ticksize_left)

    # Check for scientific notation on the left axis by looking at the tick labels
    if any('e' in tick.get_text() or 'E' in tick.get_text() for tick in ax.get_yticklabels()):
        # Set the font size for the scientific notation labels
        if sci_label_size_left is not None:
            for tick in ax.get_yticklabels():
                tick.set_fontsize(sci_label_size_left)
            offset_text = ax.yaxis.get_offset_text()
            if offset_text.get_text():
                offset_text.set_fontsize(sci_label_size_left)

    # Check for the existence of a right-side secondary axis
    for child in ax.figure.get_axes():
        if child != ax and ax.get_shared_x_axes().joined(ax, child):
            # Update right axis color if specified and if the secondary axis exists
            if color_right:
                child.spines['right'].set_color(color_right)
                child.yaxis.label.set_color(color_right)
                child.tick_params(axis='y', colors=color_right)

            if ticksize_right:
                child.tick_params(axis='both', which='major', labelsize=ticksize_right)

            if label_right:
                # Set the y-axis label for the right axis
                if color_right:
                    child.set_ylabel(label_right, fontsize=label_size_right, color=color_right)
                else:
                    child.set_ylabel(label_right, fontsize=label_size_right)  # Omit color parameter if None

            offset_text = child.yaxis.get_offset_text()
            if offset_text.get_text():
                offset_text.set_fontsize(ticksize_right)

            # Check for scientific notation on the right axis by looking at the tick labels
            if any('e' in tick.get_text() or 'E' in tick.get_text() for tick in child.get_yticklabels()):
                # Set the font size for the scientific notation labels
                if sci_label_size_right is not None:
                    for tick in child.get_yticklabels():
                        tick.set_fontsize(sci_label_size_right)
                    offset_text = child.yaxis.get_offset_text()
                    if offset_text.get_text():
                        offset_text.set_fontsize(sci_label_size_right)

            break  # Exit the loop if the right axis is found

# def update_axis_colors_ticksize_axlabels(ax, color_left=None, color_right=None, ticksize_left=None, ticksize_right=None, label_left=None, label_right=None, label_size_left=None, label_size_right=None):
#     """
#     Updates the colors of the left and/or right y-axis spines, ticks, and labels.
#     If the axis on one side does not exist, it will not modify or create a new one.
#
#     Parameters:
#     -----------
#     ax : matplotlib.axes.Axes
#         The primary axis whose y-axis colors will be updated.
#     color_left : str or None, optional
#         The color to apply to the left y-axis spine, ticks, and labels. If None, the left axis is not modified. Default is None.
#     color_right : str or None, optional
#         The color to apply to the right y-axis spine, ticks, and labels. If None, the right axis is not modified. Default is None.
#     ticksize_left : int or None, optional
#         The ticksize to apply to the left y-axis spine, ticks, and labels. If None, the right axis is not modified. Default is None.
#      ticksize_right : int or None, optional
#         The ticksize to apply to the right y-axis spine, ticks, and labels. If None, the right axis is not modified. Default is None.
#
#     Returns:
#     --------
#     None
#     """
#     # Update left axis color if specified
#     if color_left:
#         ax.spines['left'].set_visible(True)
#         ax.spines['left'].set_color(color_left)
#         ax.yaxis.label.set_color(color_left)
#         ax.tick_params(axis='y', colors=color_left)
#
#     if label_left:
#         if color_left:
#             ax.set_ylabel(label_left, fontsize=label_size_left, color=color_left)
#         else:
#             ax.set_ylabel(label_left, fontsize=label_size_left)
#
#     if ticksize_left:
#         ax.tick_params(axis='both', which='major', labelsize=ticksize_left)
#         offset_text = ax.yaxis.get_offset_text()
#         if offset_text.get_text():
#             offset_text.set_fontsize(ticksize_left)
#
#     # Check for the existence of a right-side secondary axis
#     for child in ax.figure.get_axes():
#         if child != ax and ax.get_shared_x_axes().joined(ax, child):
#             # Update right axis color if specified and if the secondary axis exists
#             if color_right:
#                 child.spines['right'].set_color(color_right)
#                 child.yaxis.label.set_color(color_right)
#                 child.tick_params(axis='y', colors=color_right)
#             if ticksize_right:
#                 child.tick_params(axis='both', which='major', labelsize=ticksize_right)
#             if label_right:
#                 if color_right:
#                     child.set_ylabel(label_right, fontsize=label_size_right, color=color_right)
#                 else:
#                     child.set_ylabel(label_right, fontsize=label_size_right)
#
#             offset_text = ax.yaxis.get_offset_text()
#             if offset_text.get_text():
#                 offset_text.set_fontsize(ticksize_right)
#
#
#             break  # Exit the loop if the right axis is found

#%% macros:

def plot_table_condesation(Segments, figsize=(8.268, 11.693), titel=None):
    """"wrapper to plot condesation data in table from list of angle segments calculated before"""

    if titel is None:
        name_angle = Segments.result[1].angle_name.replace('\\', '')
        titel = f"'{Segments.result[0].colnames['y']}'" + "\n " + f"in '{name_angle}' directional sections" + "\n" + r"\small " + f"with v_m = '{Segments.result[0].colnames['x']}'"

    data = []
    grey = []
    columns = []

    for i, Seg in enumerate(Segments.result):

        data.append(Seg.result["value"].values)
        is_condensation = Seg.result["iscondensation"]

        grey_curr = [not curr for curr in is_condensation]

        grey.append(grey_curr)

        if Seg.angles_mod is not None:
            mean_angle = gl.angle_midpoints([Seg.angles_mod[0]], [Seg.angles_mod[1]])[0]
            columns.append(f"{mean_angle:3.0f}°")
        else:
            columns.append(f"omni")

    num_row = np.max([len(data_curr) for data_curr in data])
    num_col = len(data)

    DATA = np.empty((num_row, num_col))
    DATA[:] = float('nan')

    DATA = np.array(data).T
    GREY = np.array(grey).T

    vm = Segments.result[0].result["vm"].values

    FIG = table(DATA, collabels=columns, rowlabels=vm, titel=titel, grey=GREY, row_label_name='v_m in [m/s]', formater=".2f", figsize=figsize)

    return FIG


def RoseBarPlot(direction, magnitude, segments):

    r_ticks, r_edges = gl.auto_ticks(0, max(magnitude), fix_end=True, edges=True)

    counts = {}

    for angle_segment in segments:

        df = pd.concat([direction, magnitude], axis=1)
        df_filt = gl.filter_dataframe(df, angle.name, angle_segment[0], angle_segment[1])

        count, _, _ = sc.stats.binned_statistic(df_filt[magnitude.name], df_filt[magnitude.name], statistic='count', bins=r_edges)

        angle_midpoint = gl.angle_midpoints([angle_segment[0]], [angle_segment[1]])

        counts[angle_midpoint] = count

    fig, ax_polar = plt.subplots(
        subplot_kw={'projection': 'polar'}, figsize=size_1mal1)

    i = 0

    cmap = matplotlib.pyplot.get_cmap('cool', n_r_seg)

    for _ in r_segments:

        r = [np.sum(x["grid"]["prob"][0:n_r_seg - i])
             for x in Data_Out.values()]

        if INPUT_SELECT["include_360"]:
            r.pop(0)

        ax_polar.bar(theta, r, width=width_seg,
                     color=cmap(n_r_seg - i), alpha=1)
        i = i + 1

    if Input["label_by_section"]:
        ax_polar.set_xticks(theta)
        x_ticklabels = [str(round(label * 360 / (2 * np.pi), 2)
                            ) + ' °' for label in theta]
        ax_polar.set_xticklabels(x_ticklabels)

    ax_polar.tick_params(pad=5)
    ax_polar.tick_params(axis='x')

    ax_polar.set_yticks(ax_polar.get_yticks())

    y_ticklables = ax_polar.get_yticklabels()
    y_ticklables = [label.get_text() + ' %' for label in y_ticklables]
    ax_polar.set_yticklabels(y_ticklables)

    labels = list(r_segments)
    labels.insert(0, 0)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ax_polar.grid(Input["grid"], zorder=-1)

    cmappable = ScalarMappable(Normalize(0, 1), cmap=cmap)
    cbar = fig.colorbar(cmappable,
                        ax=ax_polar, pad=0.1
                        )

    cbar.set_ticks(ticks=np.linspace(0, 1, n_r_seg + 1), labels=labels)

    for tick_label in cbar.ax.yaxis.get_ticklabels():
        tick_label.set_fontsize(6)

    if Input["name_colorbar"] is None:
        cbar.set_label(Input["col_name_r"])
    else:
        cbar.set_label(Input["name_colorbar"])

    ax_polar.set_rlabel_position(Input["rot_r_label"])
    # Set 0 degrees to point upward (North)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)  # Rotate clockwise

    if Input["title"] is None:
        fig.suptitle(
            f"Roseplot '{Input['col_name_r']}' over '{Input['col_name_angle']}'" +
            "\n" + f"samples: {len(DATA):.1e}"
                   f" | {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                   f"| sample rate: {INPUT_SELECT['dt_sample']}",

            fontsize=7, y=0.93)
    else:
        fig.suptitle(Input["title"], fontsize=7, y=0.93)

    r_max = ax_polar.get_rmax()
    ax_polar.set_rmax(np.ceil(r_max))
    fig.tight_layout()

    return [fig]


# %% Save figure

def save_figs_as_png(FIG, filename, **kwargs):
    dpi = kwargs.get('dpi', 600)

    i = 1
    for fig in FIG:
        fig.savefig(filename + f"_page_{i}.png", dpi=dpi)
        i = i + 1
        plt.close(fig)
    return


def save_figs_as_pdf(FIG, filename, **kwargs):
    dpi = kwargs.get('dpi', 600)

    with PdfPages(filename + ".pdf") as pdf:
        for fig in FIG:
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
    return


# %% archive

def plot_single(Data_Out, Input, LIMS, num_plot):
    FIG = []

    fig, axis = plt.subplots(1, 1, figsize=size_1mal2)
    cd = Data_Out[list(Data_Out)[num_plot]]
    ck = list(Data_Out)[num_plot]

    x = DATA_SEC[ck][Input['col_name_x']]
    y = DATA_SEC[ck][Input['col_name_y']]

    x_grid = cd['grid']['x']

    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0.01, 0.8, 0.25, 0.15], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)

    # filter results out of grid-dataframe
    col_key = [
        col for col in cd['grid'].columns if 'plot' in col]

    # filter and plot mean
    col_mean = [col for col in col_key if 'mean' in col][0]

    plot_mean = axis.plot(x_grid, cd['grid'].loc[:, col_mean], linewidth=0.9,
                          color='black', zorder=6, label='condensed average')

    # filter and plot percentiles, if given
    col_perc = [col for col in col_key if 'percentile' in col]

    cmap_lines = LinearSegmentedColormap.from_list(
        'custom_colormap', ['#ff0000', '#00ff00'])
    range_grey = np.linspace(0, 1, len(col_perc))

    plot_perc = []
    for i_name, name in enumerate(col_perc):
        temp = axis.plot(x_grid, cd['grid'].loc[:, name], linewidth=0.9, color=cmap_lines(
            range_grey[i_name]), zorder=6, label=name.replace('result', '').replace('plot', ''))
        plot_perc.append(temp)

    # Create a scatter plot colored by density
    cmap = 'cool'
    c = gl.c_scatterplot(x, y)

    axis.scatter(x, y, cmap=cmap, c=c, s=0.3, zorder=1, alpha=0.5,
                 norm=colors.PowerNorm(gamma=Input["colormap_gradient"]))

    # plotlims and scaling

    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]
    y_lim_erg = axis.set_ylim(y_lims)
    axis.set_xlim(x_lims)
    axis.set_ylim(y_lims)

    range_y = y_lim_erg[1] - y_lim_erg[0]

    # filter and plot quantiles
    col_quant = [
        col for col in cd['grid'].columns if 'quantile' in col]

    if len(col_quant) > 0:
        col_quant = col_quant[0]
        axis.plot(x_grid, cd['grid'].loc[:, col_quant], linewidth=1,
                  color='red', zorder=7, label='selected correlation')

        plot_mean[0].set_color('black')
        plot_perc[0][0].set_color('black')
        plot_perc[1][0].set_color('black')

        plot_mean[0].set_linestyle('-')
        plot_perc[0][0].set_linestyle('--')
        plot_perc[1][0].set_linestyle('-.')

        if "quant_low" in Input:
            if Input["quant_low"] is not None:
                axis.axhline(1 / Input["quant_low"], linewidth=1,
                             color=[0, 1, 0])
                axis.text(x_lims[1] * Input["quant_label_x"], 1 / Input["quant_low"] + (0.01 * range_y),
                          f"f_low: {Input['quant_low']} Hz", fontsize=6)

        if "quant_up" in Input:
            if Input["quant_up"] is not None:
                axis.axhline(1 / Input["quant_up"], linewidth=1,
                             color=[0, 1, 0])
                axis.text(x_lims[1] * Input["quant_label_x"], 1 / Input["quant_up"] + (0.01 * range_y),
                          f"f_up: {Input['quant_up']} Hz", fontsize=6)

    # Plot title
    if Input["name_angle"] is not None:
        name_angle = Input["name_angle"]
    else:
        name_angle = ck

    name_angle += f": {round(segments[num_plot][0], 2)}° to {round(segments[num_plot][1], 2)}°" + '\n' + \
                  (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                   f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                   f"| sample rate: {INPUT_SELECT['dt_sample']}")

    axis.legend(loc="lower right")

    axis.set_title(name_angle)

    # Name
    if Input["name_x_axis"] is not None:
        name_x_axis = Input["name_x_axis"]
    else:
        name_x_axis = x.name

    if Input["name_y_axis"] is not None:
        name_y_axis = Input["name_y_axis"]
    else:
        name_y_axis = y.name

    plt.xlabel(name_x_axis)
    plt.ylabel(name_y_axis)
    fig.tight_layout()

    FIG.append(fig)

    return FIG


def plot_tiled_RWI(Data_Out, Input, LIMS):
    N_exp = len(Data_Out)

    RWI_max = max([max(cd) for _, cd in Data_Out.items()])

    # Name
    if Input["name_x_axis"] is not None:
        name_x_axis = Input["name_x_axis"]
    else:
        name_x_axis = COLNAMES["Hs"]

    if Input["name_y_axis"] is not None:
        name_y_axis = Input["name_y_axis"]
    else:
        name_y_axis = COLNAMES["Tp"]

    i = 0

    FIG = []
    pages = int(np.ceil(N_exp / 6))

    for page, _ in enumerate(range(pages)):

        fig, ax = plt.subplots(3, 2, figsize=size_3mal2)

        ax_flat = ax.flatten()
        # plotlims and scaling

        x_lims = LIMS["x_lims"]
        y_lims = LIMS["y_lims"]

        for i_page, _ in enumerate(range(6)):

            axis = ax_flat[i_page]

            # if section exists
            if i < N_exp:
                # delete unnesseary axis
                if i == N_exp:
                    for delete in ax_flat[np.mod(N_exp, 6):]:
                        delete.set_axis_off()
                    break

                cd = Data_Out[list(Data_Out)[i]]
                ck = list(Data_Out)[i]

                x = DATA_SEC[ck][COLNAMES["Hs"]]
                y = DATA_SEC[ck][COLNAMES["Tp"]]

                # plotlims and scaling
                x_lims = LIMS["x_lims"]
                y_lims = LIMS["y_lims"]

                axis.set_xlim(x_lims[0], x_lims[1])
                axis.set_ylim(y_lims[0], y_lims[1])

                # JBO-Logo
                with matplotlib.cbook.get_sample_data(path + r'\JBO_logo.png') as file:
                    image_bgr = plt.imread(file, format='png')

                axin = axis.inset_axes([0.05, 0.7, 0.5, 0.3], zorder=-1)
                axin.imshow(image_bgr, zorder=-1)
                axin.axis('off')

                # grid
                axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)

                x_lim_erg = axis.set_xlim(x_lims)
                y_lim_erg = axis.set_ylim(y_lims)

                range_x = x_lim_erg[1] - x_lim_erg[0]
                range_y = y_lim_erg[1] - y_lim_erg[0]

                axis.set_aspect(aspect=range_x / range_y, adjustable=None)

                # skatterplot
                axis.scatter(
                    x, y, c=cd, cmap='cool', s=0.5, zorder=1)

                axis.hlines(1 / Input["f_0"],
                            xmin=x_lim_erg[0], xmax=x_lim_erg[1], linewidth=1, color=[0, 1, 0])
                axis.text(range_x * 0.7, 1 / Input["f_0"] + 0.015 * range_y,
                          f'$f_0 = {Input["f_0"]}$ Hz', fontsize=6)

                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize

                cmappable = ScalarMappable(
                    Normalize(0, RWI_max), cmap='cool')
                cbar = plt.colorbar(mappable=cmappable, ax=axis, shrink=0.85)

                for tick_label in cbar.ax.yaxis.get_ticklabels():
                    tick_label.set_fontsize(7)

                if Input["name_colorbar"] is not None:
                    name_cbar = Input["name_colorbar"]
                else:
                    name_cbar = "RWI = $\\sqrt{S(f_0)}$ (Resonance Wave Intesity) $[\\sqrt{m^2/Hz}]$"

                cbar.set_label(name_cbar)
                # Plot title
                if Input["name_angle"] is not None:
                    name_angle = Input["name_angle"]
                else:
                    name_angle = ck

                # name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + \
                #              (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                #               f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                #               f"| sample rate: {INPUT_SELECT['dt_sample']}")

                # zwischenlösung für report, dannach löschen!
                name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + f"samples: {len(x):.4e} ({round(len(x) / len(DATA) * 100, 1)}%) "

                axis.set_title(name_angle)

                axis.set_xlabel(name_x_axis)
                axis.set_ylabel(name_y_axis)

            else:

                x_lim_erg = axis.set_xlim(x_lims)
                y_lim_erg = axis.set_ylim(y_lims)

                range_x = x_lim_erg[1] - x_lim_erg[0]
                range_y = y_lim_erg[1] - y_lim_erg[0]

                axis.set_aspect(aspect=range_x / range_y, adjustable=None)

                axis.spines['top'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.spines['right'].set_visible(False)

                axis.tick_params(axis=u'both', which=u'both', length=0)
                axis.set_xticklabels({})
                axis.set_yticklabels({})
                axis.set_title(' \n ')
                axis.set_xlabel(' ')
                axis.set_ylabel(' ')

            i = i + 1

        fig.tight_layout()

        FIG.append(fig)

    return FIG


def plot_single_RWI(Data_Out, Input, LIMS, num_plot):
    FIG = []
    fig, axis = plt.subplots(1, 1, figsize=size_1mal2)
    cd = Data_Out[list(Data_Out)[num_plot]]
    ck = list(Data_Out)[num_plot]
    RWI_max = max(Data_Out[ck])

    x = DATA_SEC[ck][COLNAMES["Hs"]]
    y = DATA_SEC[ck][COLNAMES["Tp"]]

    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0.01, 0.8, 0.25, 0.15], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)

    # plotlims and scaling
    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]

    x_lim_erg = axis.set_xlim(x_lims)
    y_lim_erg = axis.set_ylim(y_lims)

    range_x = x_lim_erg[1] - x_lim_erg[0]
    range_y = y_lim_erg[1] - y_lim_erg[0]

    # skatterplot
    axis.scatter(
        x, y, c=cd, cmap='cool', s=0.3, zorder=1)

    axis.hlines(1 / Input["f_0"], xmin=x_lim_erg[0],
                xmax=x_lim_erg[1], linewidth=1, color=[0, 1, 0])
    axis.text(range_x * 0.7, 1 / Input["f_0"] + 0.015 * range_y,
              f'$f_0 = {Input["f_0"]}$ Hz', fontsize=6)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cmappable = ScalarMappable(
        Normalize(0, RWI_max), cmap='cool')
    cbar = plt.colorbar(mappable=cmappable, ax=axis)

    for tick_label in cbar.ax.yaxis.get_ticklabels():
        tick_label.set_fontsize(7)

    if Input["name_colorbar"] is not None:
        name_cbar = Input["name_colorbar"]
    else:
        name_cbar = "RWI = $\\sqrt{S(f_0)}$ (Resonance Wave Intesity) $[\\sqrt{m^2/Hz}]$"

    cbar.set_label(name_cbar)
    # Plot title
    if Input["name_angle"] is not None:
        name_angle = Input["name_angle"]
    else:
        name_angle = ck

    # name_angle += f": {round(segments[num_plot][0], 2)}° to {round(segments[num_plot][1], 2)}°" + '\n' + \
    #              (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
    #               f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
    #               f"| sample rate: {INPUT_SELECT['dt_sample']}")

    axis.set_title(name_angle)

    # Name
    if Input["name_x_axis"] is not None:
        name_x_axis = Input["name_x_axis"]
    else:
        name_x_axis = x.name

    if Input["name_y_axis"] is not None:
        name_y_axis = Input["name_y_axis"]
    else:
        name_y_axis = y.name

    plt.xlabel(name_x_axis)
    plt.ylabel(name_y_axis)
    fig.tight_layout()

    FIG.append(fig)

    return FIG


def plot_tiled_DEL_Scatter(Data_Out, Input, LIMS):
    N_exp = len((Data_Out["set_points"]["segments"]))

    DEL_max = max(Data_Out["set_points"]["table"][Input["config_scatter"]])

    i = 0

    FIG = []
    pages = int(np.ceil(N_exp / 6))

    # Name
    if Input["name_x_axis"] is not None:
        name_x_axis = Input["name_x_axis"]
    else:
        name_x_axis = COLNAMES["Hs"]

    if Input["name_y_axis"] is not None:
        name_y_axis = Input["name_y_axis"]
    else:
        name_y_axis = COLNAMES["Tp"]
    # plotlims and scaling
    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]
    for page, _ in enumerate(range(pages)):

        fig, ax = plt.subplots(3, 2, figsize=size_3mal2)

        ax_flat = ax.flatten()

        for i_page, _ in enumerate(range(6)):

            axis = ax_flat[i_page]

            # if section exists
            if i < N_exp:

                cd = Data_Out["set_points"]["segments"][list(
                    Data_Out["set_points"]["segments"])[i]]
                ck = list(Data_Out["set_points"]["segments"])[i]

                x = Data_Out["set_points"]["segments"][ck]["Hs"]
                y = Data_Out["set_points"]["segments"][ck]["Tp"]

                # plotlims and scaling
                x_lims = LIMS["x_lims"]
                y_lims = LIMS["y_lims"]

                axis.set_xlim(x_lims[0], x_lims[1])
                axis.set_ylim(y_lims[0], y_lims[1])

                # JBO-Logo
                with matplotlib.cbook.get_sample_data(path + r'\JBO_logo.png') as file:
                    image_bgr = plt.imread(file, format='png')

                axin = axis.inset_axes([0.05, 0.7, 0.5, 0.3], zorder=-1)
                axin.imshow(image_bgr, zorder=-1)
                axin.axis('off')

                # grid
                axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)

                x_lim_erg = axis.set_xlim(x_lims)
                y_lim_erg = axis.set_ylim(y_lims)

                range_x = x_lim_erg[1] - x_lim_erg[0]
                range_y = y_lim_erg[1] - y_lim_erg[0]

                axis.set_aspect(aspect=range_x / range_y, adjustable=None)

                # skatterplot
                axis.scatter(
                    x, y, c=cd[Input["config_scatter"]], cmap='cool', s=0.5, zorder=1)

                axis.hlines(1 / Input["f_0"],
                            xmin=x_lim_erg[0], xmax=x_lim_erg[1], color=[0, 1, 0])
                axis.text(range_x * 0.7, 1 / Input["f_0"] + 0.015 * range_y,
                          f'$f_0 = {Input["f_0"]}$ Hz', fontsize=6)

                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize

                cmappable = ScalarMappable(
                    Normalize(0, DEL_max), cmap='cool')
                cbar = plt.colorbar(mappable=cmappable, ax=axis, shrink=0.85)

                for tick_labels in cbar.ax.yaxis.get_ticklabels():
                    tick_labels.set_fontsize(7)

                if Input["name_colorbar"] is not None:
                    name_cbar = Input["name_colorbar"]
                else:
                    name_cbar = (f"Bending DEL [Nm] | m = {Input['SN_slope']} " + "\n"
                                                                                  f"| N_ref = {Input['N_ref']:.2e} "
                                                                                  f"| lifetime = {Input['design_life']}y")

                cbar.set_label(name_cbar)

                # Plot title
                if Input["name_angle"] is not None:
                    name_angle = Input["name_angle"]
                else:
                    name_angle = ck

                # name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + \
                #              (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                #               f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                #               f"| sample rate: {INPUT_SELECT['dt_sample']}")

                # zwischenlösung für report, dannach löschen!
                name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + \
                              f"samples: {len(x):.4e} ({round(len(x) / len(DATA) * 100, 1)}%) "

                axis.set_title(name_angle)

                axis.set_xlabel(name_x_axis)
                axis.set_ylabel(name_y_axis)

            else:

                x_lim_erg = axis.set_xlim(x_lims)
                y_lim_erg = axis.set_ylim(y_lims)

                range_x = x_lim_erg[1] - x_lim_erg[0]
                range_y = y_lim_erg[1] - y_lim_erg[0]

                axis.set_aspect(aspect=range_x / range_y, adjustable=None)

                axis.spines['top'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.spines['right'].set_visible(False)

                axis.tick_params(axis=u'both', which=u'both', length=0)
                axis.set_xticklabels({})
                axis.set_yticklabels({})
                axis.set_title(' \n ')
                axis.set_xlabel(' ')
                axis.set_ylabel(' ')
            i = i + 1

        fig.tight_layout()

        FIG.append(fig)

    return FIG


def plot_single_DEL_Scatter(Data_Out, Input, LIMS, num_plot):
    fig, axis = plt.subplots(1, 1, figsize=size_1mal2)

    DEL_max = max(Data_Out["set_points"]["table"][Input["config_scatter"]])

    i = 0

    FIG = []

    # Name
    if Input["name_x_axis_scatter"] is not None:
        name_x_axis = Input["name_x_axis_scatter"]
    else:
        name_x_axis = COLNAMES["Hs"]

    if Input["name_y_axis_scatter"] is not None:
        name_y_axis = Input["name_y_axis_scatter"]
    else:
        name_y_axis = COLNAMES["Tp"]

    cd = Data_Out["set_points"]["segments"][list(
        Data_Out["set_points"]["segments"])[num_plot]]
    ck = list(Data_Out["set_points"]["segments"])[num_plot]

    x = Data_Out["set_points"]["segments"][ck]["Hs"]
    y = Data_Out["set_points"]["segments"][ck]["Tp"]

    # plotlims and scaling
    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]

    axis.set_xlim(x_lims[0], x_lims[1])
    axis.set_ylim(y_lims[0], y_lims[1])

    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + r"\JBO_logo.png") as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0.01, 0.8, 0.25, 0.15], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)

    # plotlims and scaling
    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]

    x_lim_erg = axis.set_xlim(x_lims)
    y_lim_erg = axis.set_ylim(y_lims)

    range_x = x_lim_erg[1] - x_lim_erg[0]
    range_y = y_lim_erg[1] - y_lim_erg[0]

    # skatterplot
    axis.scatter(
        x, y, c=cd[Input["config_scatter"]], cmap='cool', s=0.5, zorder=1)

    axis.hlines(1 / Input["f_0"],
                xmin=x_lim_erg[0], xmax=x_lim_erg[1], color=[0, 1, 0])
    axis.text(range_x * 0.7, 1 / Input["f_0"] + 0.015 * range_y,
              f'$f_0 = {Input["f_0"]}$ Hz', fontsize=6)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cmappable = ScalarMappable(
        Normalize(0, DEL_max), cmap='cool')
    cbar = plt.colorbar(mappable=cmappable, ax=axis)

    for tick_labels in cbar.ax.yaxis.get_ticklabels():
        tick_labels.set_fontsize(7)

    if Input["name_colorbar"] is not None:
        name_cbar = Input["name_colorbar"]
    else:
        name_cbar = (f"Bending DEL [Nm] | m = {Input['SN_slope']} "
                     f"| N_ref = {Input['N_ref']:.2e} "
                     f"| lifetime = {Input['design_life']}y")

    cbar.set_label(name_cbar)

    # Plot title
    if Input["name_angle"] is not None:
        name_angle = Input["name_angle"]
    else:
        name_angle = ck

    name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + \
                  (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                   f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                   f"| sample rate: {INPUT_SELECT['dt_sample']}")

    axis.set_title(name_angle)

    axis.set_xlabel(name_x_axis)
    axis.set_ylabel(name_y_axis)

    fig.tight_layout()

    FIG.append(fig)

    return FIG


def plot_RosePlot(Data_Out, Input):
    r_segments = Data_Out[list(Data_Out.keys())[
        0]]['grid']["r_uper_edge"]

    if INPUT_SELECT["include_360"]:
        seg = segments[1:]

    else:
        seg = segments
    n_r_seg = len(r_segments)
    width_seg = (seg[0][1] - seg[0][0]) * 2 * np.pi / 360

    theta = np.mean(seg, axis=1) * 2 * np.pi / 360

    fig, ax_polar = plt.subplots(
        subplot_kw={'projection': 'polar'}, figsize=size_1mal1)

    i = 0

    cmap = matplotlib.pyplot.get_cmap('cool', n_r_seg)

    for _ in r_segments:

        r = [np.sum(x["grid"]["prob"][0:n_r_seg - i])
             for x in Data_Out.values()]

        if INPUT_SELECT["include_360"]:
            r.pop(0)

        ax_polar.bar(theta, r, width=width_seg,
                     color=cmap(n_r_seg - i), alpha=1)
        i = i + 1

    if Input["label_by_section"]:
        ax_polar.set_xticks(theta)
        x_ticklabels = [str(round(label * 360 / (2 * np.pi), 2)
                            ) + ' °' for label in theta]
        ax_polar.set_xticklabels(x_ticklabels)

    ax_polar.tick_params(pad=5)
    ax_polar.tick_params(axis='x')

    ax_polar.set_yticks(ax_polar.get_yticks())

    y_ticklables = ax_polar.get_yticklabels()
    y_ticklables = [label.get_text() + ' %' for label in y_ticklables]
    ax_polar.set_yticklabels(y_ticklables)

    labels = list(r_segments)
    labels.insert(0, 0)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ax_polar.grid(Input["grid"], zorder=-1)

    cmappable = ScalarMappable(Normalize(0, 1), cmap=cmap)
    cbar = fig.colorbar(cmappable,
                        ax=ax_polar, pad=0.1
                        )

    cbar.set_ticks(ticks=np.linspace(0, 1, n_r_seg + 1), labels=labels)

    for tick_label in cbar.ax.yaxis.get_ticklabels():
        tick_label.set_fontsize(6)

    if Input["name_colorbar"] is None:
        cbar.set_label(Input["col_name_r"])
    else:
        cbar.set_label(Input["name_colorbar"])

    ax_polar.set_rlabel_position(Input["rot_r_label"])
    # Set 0 degrees to point upward (North)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)  # Rotate clockwise

    if Input["title"] is None:
        fig.suptitle(
            f"Roseplot '{Input['col_name_r']}' over '{Input['col_name_angle']}'" +
            "\n" + f"samples: {len(DATA):.1e}"
                   f" | {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                   f"| sample rate: {INPUT_SELECT['dt_sample']}",

            fontsize=7, y=0.93)
    else:
        fig.suptitle(Input["title"], fontsize=7, y=0.93)

    r_max = ax_polar.get_rmax()
    ax_polar.set_rmax(np.ceil(r_max))
    fig.tight_layout()

    return [fig]


def plot_tiled_break_steep(Data_Out, Input, LIMS):
    N_exp = len(Data_Out)

    i = 0

    c = {}
    c_max = 0
    c_min = 0
    for ck, cd in Data_Out.items():

        x = DATA_SEC[ck][COLNAMES["Hs"]]
        y = DATA_SEC[ck][COLNAMES["Tp"]]
        break_bool = cd["bool_break"]

        temp = np.array(gl.c_scatterplot(x, y))

        c_min_try = min(temp)

        if c_min_try < c_min:
            c_min = c_min_try

        c_max_try = max(temp)

        if c_max_try > c_max:
            c_max = c_max_try

        temp[np.array(break_bool) == False] = -1
        c[ck] = temp

    cmap = matplotlib.colormaps['cool']
    cmap.set_under('grey')

    FIG = []
    pages = int(np.ceil(N_exp / 6))

    # Name
    if Input["name_x_axis"] is not None:
        name_x_axis = Input["name_x_axis"]
    else:
        name_x_axis = COLNAMES["Hs"]

    if Input["name_y_axis"] is not None:
        name_y_axis = Input["name_y_axis"]
    else:
        name_y_axis = COLNAMES["Tp"]
    # plotlims and scaling
    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]
    for page, _ in enumerate(range(pages)):

        fig, ax = plt.subplots(3, 2, figsize=size_3mal2)

        ax_flat = ax.flatten()

        for i_page, _ in enumerate(range(6)):

            axis = ax_flat[i_page]

            # if section exists
            if i < N_exp:

                ck = list(Data_Out)[i]

                x = DATA_SEC[ck][COLNAMES["Hs"]]
                y = DATA_SEC[ck][COLNAMES["Tp"]]

                # JBO-Logo
                with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
                    image_bgr = plt.imread(file, format='png')

                axin = axis.inset_axes([0.05, 0.7, 0.5, 0.3], zorder=-1)
                axin.imshow(image_bgr, zorder=-1)
                axin.axis('off')

                # grid
                axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)

                # axis.set_aspect(aspect=range_x/range_y, adjustable=None)

                # Create a scatter plot colored by density
                cmap = matplotlib.colormaps['cool']
                cmap.set_under('grey')

                if max(c[ck]) == -1:
                    axis.scatter(x, y, color='grey', s=0.3,
                                 zorder=1, alpha=0.5)
                else:
                    axis.scatter(x, y, cmap=cmap,
                                 c=c[ck], s=0.3, zorder=1, alpha=0.5, vmin=0)

                # Plot title
                if Input["name_angle"] is not None:
                    name_angle = Input["name_angle"]
                else:
                    name_angle = ck

                # name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + \
                #              (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                #               f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                #               f"| sample rate: {INPUT_SELECT['dt_sample']}")

                # zwischenlösung für report, dannach löschen!
                name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + \
                              f"samples: {len(x):.4e} ({round(len(x) / len(DATA) * 100, 1)}%) "

                axis.set_title(name_angle)

                x_lim_erg = axis.set_xlim(x_lims)
                y_lim_erg = axis.set_ylim(y_lims)

                range_x = x_lim_erg[1] - x_lim_erg[0]
                range_y = y_lim_erg[1] - y_lim_erg[0]

                axis.set_aspect(aspect=range_x / range_y, adjustable=None)

                axis.set_xlabel(name_x_axis)
                axis.set_ylabel(name_y_axis)

                from matplotlib.lines import Line2D

                legend_elements = [Line2D([0], [0], marker='o', color=cmap(0),
                                          label=r'steep seas ($H_s/\lambda > ' +
                                                f'{Input["steep_crit"]}$)',
                                          markerfacecolor=cmap(0), markersize=5, linewidth=0)]

                axis.legend(handles=legend_elements, loc='lower right')

            else:

                x_lim_erg = axis.set_xlim(x_lims)
                y_lim_erg = axis.set_ylim(y_lims)

                range_x = x_lim_erg[1] - x_lim_erg[0]
                range_y = y_lim_erg[1] - y_lim_erg[0]

                axis.set_aspect(aspect=range_x / range_y, adjustable=None)

                axis.spines['top'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.spines['right'].set_visible(False)

                axis.tick_params(axis=u'both', which=u'both', length=0)
                axis.set_xticklabels({})
                axis.set_yticklabels({})
                axis.set_title(' \n ')
                axis.set_xlabel(' ')
                axis.set_ylabel(' ')
            i = i + 1

        fig.tight_layout()

        FIG.append(fig)

    return FIG


def plot_single_break_steep(Data_Out, Input, LIMS, num_plot):
    FIG = []
    fig, axis = plt.subplots(1, 1, figsize=size_1mal2)
    cd = Data_Out[list(Data_Out)[num_plot]]["bool_break"]
    ck = list(Data_Out)[num_plot]

    x = DATA_SEC[ck][COLNAMES["Hs"]]
    y = DATA_SEC[ck][COLNAMES["Tp"]]
    break_bool = cd

    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0.01, 0.8, 0.25, 0.15], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)

    c = np.array(gl.c_scatterplot(x, y))
    v_min = min(c)
    c[np.array(break_bool) == False] = -1

    # Create a scatter plot colored by density
    cmap = matplotlib.colormaps['cool']
    cmap.set_under('grey')

    # skatterplot
    axis.scatter(
        x, y, c=c, cmap=cmap, s=0.3, zorder=1, vmin=v_min)

    # Plot title
    if Input["name_angle"] is not None:
        name_angle = Input["name_angle"]
    else:
        name_angle = ck

    name_angle += f": {round(segments[num_plot][0], 2)}° to {round(segments[num_plot][1], 2)}°" + '\n' + \
                  (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                   f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                   f"| sample rate: {INPUT_SELECT['dt_sample']}")

    axis.set_title(name_angle)

    # Name
    if Input["name_x_axis"] is not None:
        name_x_axis = Input["name_x_axis"]
    else:
        name_x_axis = x.name

    if Input["name_y_axis"] is not None:
        name_y_axis = Input["name_y_axis"]
    else:
        name_y_axis = y.name

    plt.xlabel(name_x_axis)
    plt.ylabel(name_y_axis)
    fig.tight_layout()

    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]
    axis.set_ylim(y_lims)
    axis.set_xlim(x_lims)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color=cmap(0), label=r'steep seas ($H_s/\lambda >' + f'{Input["steep_crit"]}$)',
               markerfacecolor=cmap(0), markersize=5, linewidth=0)]
    axis.legend(handles=legend_elements, loc='lower right')

    FIG.append(fig)

    return FIG


def plot_tiled_DEL(Data_Out, Input, LIMS):
    N_exp = len(DATA_SEC)

    FIG = []
    i = 0

    # Name
    if Input["name_x_axis"] is not None:
        name_x_axis = Input["name_x_axis"]
    else:
        name_x_axis = "Wind speed [m/s]"

    if Input["name_y_axis"] is not None:
        name_y_axis = Input["name_y_axis"]
    else:
        name_y_axis = (f"Bending DEL [Nm] | m = {Input['SN_slope']} "
                       f"| N_ref = {Input['N_ref']:.2e} | lifetime = {Input['design_life']}y")

    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]
    pages = int(np.ceil(N_exp / 6))

    cmap_lines = LinearSegmentedColormap.from_list(
        "custom_colormap", ['#ff0000', '#00ff00'])
    keys = Data_Out["Compare"]["added"].columns
    range_colors = np.linspace(0, 1, len(keys))

    for page, _ in enumerate(range(pages)):

        fig, ax = plt.subplots(3, 2, figsize=size_3mal2)

        ax_flat = ax.flatten()

        for i_page, _ in enumerate(range(6)):

            axis = ax_flat[i_page]
            axis_2 = axis.twinx()
            # if section exists
            if i < N_exp:

                ck = list(Data_Out["Compare"]["vm-vise"].keys())[i]

                # read Data
                Force_added_table = {}
                Force_added_points = {}
                Force_added_compare = {}
                Force_vm_table = {}
                Force_vm_points = {}
                Force_vm_compare = {}

                x_edges = Data_Out["Vm_edges"]
                x = (x_edges[1:] + x_edges[:-1]) / 2

                fontsize = 6

                n = 0
                n_text = 0
                point_num = len(Data_Out["set_points"]["segments"][ck])

                barcolor = [0.85, 0.85, 0.85]
                spinecolor = [0.5, 0.5, 0.5]

                axis.set_ylabel('number of datapoints', color=spinecolor)
                axis.tick_params(axis='y')
                axis.spines['left'].set_color(spinecolor)
                axis.yaxis.label.set_color(spinecolor)
                axis.tick_params(axis='y', colors=spinecolor)
                axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                # axis.yaxis.tick_right()
                axis.bar(x, Data_Out["set_table"]["count_table"][ck], width=1,
                         bottom=None, align='center', color=barcolor, zorder=-1, alpha=0.5)

                axis_2.spines['left'].set_visible(False)

                for key in keys:
                    Force_added_table[key] = Data_Out["set_table"]["DEL"]["added"].loc[ck, key]
                    Force_added_points[key] = Data_Out["set_points"]["DEL"]["added"].loc[ck, key]
                    Force_added_compare[key] = Data_Out["Compare"]["added"].loc[ck, key]

                    Force_vm_table[key] = Data_Out["set_table"]["DEL"]["vm-vise"][ck][key]
                    Force_vm_points[key] = Data_Out["set_points"]["DEL"]["vm-vise"][ck][key]
                    Force_vm_compare[key] = Data_Out["Compare"]["vm-vise"][ck][key]

                    axis_2.plot(x, Force_vm_table[key], label=key + ' condensed', color=cmap_lines(
                        range_colors[n]), linestyle='--')
                    axis_2.plot(x, Force_vm_points[key],
                                label=key + ' Hindcast', color=cmap_lines(
                            range_colors[n]), linestyle='-')

                    axis_2.text(
                        0.6, 0.95 - n_text * 0.05, f"{key}", fontweight='bold', transform=axis.transAxes,
                        fontsize=fontsize)
                    axis_2.text(0.6, 0.95 - (n_text + 1) * 0.05,
                                f"DEL condensed: {Force_added_table[key]:.3e} Nm", transform=axis.transAxes,
                                fontsize=fontsize)
                    axis_2.text(0.6, 0.95 - (n_text + 2) * 0.05,
                                f"DEL Hindcast: {Force_added_points[key]:.3e} Nm", transform=axis.transAxes,
                                fontsize=fontsize)
                    axis_2.text(0.6, 0.95 - (n_text + 3) * 0.05,
                                f"condensed/Hindcast: {round(Force_added_compare[key], 1)} %", transform=axis.transAxes,
                                fontsize=fontsize)
                    n_text = n_text + 4
                    n = n + 1

                axis.set_xlabel(name_x_axis)
                axis_2.set_ylabel(name_y_axis)

                # legend

                axis_2.legend(fontsize=6, loc="lower right", framealpha=0.5)

                axis_2.set_xlim(x_lims[0], x_lims[1])
                axis_2.set_ylim(y_lims[0], y_lims[1])

                # JBO-Logo
                with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
                    image_bgr = plt.imread(file, format='png')

                axin = axis.inset_axes([0.05, 0.7, 0.5, 0.3], zorder=-2)
                axin.imshow(image_bgr, zorder=-2)
                axin.axis('off')

                # grid
                axis_2.grid(visible=True, color=[
                    0.7, 0.7, 0.7], zorder=4, axis='both')
                axis.grid(visible=True, color=[
                    0.7, 0.7, 0.7], zorder=4, axis='x')

                # Plot title
                if Input["name_angle"] is not None:
                    name_angle = Input["name_angle"]
                else:
                    name_angle = ck

                # name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + \
                #              (f"samples: {point_num:.1e} ({round(point_num / len(DATA) * 100, 1)}%) "
                #               f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                #               f"| sample rate: {INPUT_SELECT['dt_sample']}")

                # zwischenlösung für report, dannach löschen!
                name_angle += f": {round(segments[i][0], 2)}° to {round(segments[i][1], 2)}°" + '\n' + \
                              f"samples: {point_num:.4e} ({round(point_num / len(DATA) * 100, 1)}%) "

                axis.set_title(name_angle)
            else:

                x_lim_erg = axis_2.set_xlim(x_lims)
                y_lim_erg = axis_2.set_ylim(y_lims)

                range_x = x_lim_erg[1] - x_lim_erg[0]
                range_y = y_lim_erg[1] - y_lim_erg[0]

                axis_2.set_aspect(aspect=range_x / range_y, adjustable=None)

                x_lim_erg = axis.get_xlim()
                y_lim_erg = axis.get_ylim()

                range_x = x_lim_erg[1] - x_lim_erg[0]
                range_y = y_lim_erg[1] - y_lim_erg[0]

                axis.set_aspect(aspect=range_x / range_y, adjustable=None)

                axis_2.spines['top'].set_visible(False)
                axis_2.spines['bottom'].set_visible(False)
                axis_2.spines['left'].set_visible(False)
                axis_2.spines['right'].set_visible(False)

                axis_2_x_label = ['      ' for _ in range(
                    len(axis_2.get_xticks()))]
                axis_2_y_label = ['      ' for _ in range(
                    len(axis_2.get_yticks()))]
                axis_x_label = ['      ' for _ in range(
                    len(axis.get_xticks()))]
                axis_y_label = ['      ' for _ in range(
                    len(axis.get_yticks()))]

                axis.set_xticks(axis.get_xticks())
                axis.set_yticks(axis.get_yticks())

                axis_2.set_xticks(axis_2.get_xticks())
                axis_2.set_yticks(axis_2.get_yticks())

                axis_2.tick_params(axis=u'both', which=u'both', length=0)
                axis_2.set_xticklabels(axis_2_x_label)
                axis_2.set_yticklabels(axis_2_y_label)
                axis_2.set_title(' \n ')
                axis_2.set_ylabel(' ')
                axis_2.set_xlabel(' ')

                axis.spines['top'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.spines['right'].set_visible(False)

                axis.tick_params(axis=u'both', which=u'both', length=0)
                axis.set_xticklabels(axis_x_label)
                axis.set_yticklabels(axis_y_label)
                axis.set_title(' \n ')
                axis.set_xlabel('  ')
                axis.set_ylabel('  ')

            i = i + 1

        fig.tight_layout()

        FIG.append(fig)

    return FIG


def plot_single_DEL(Data_Out, Input, LIMS, num_plot):
    FIG = []

    fig, axis = plt.subplots(1, 1, figsize=size_1mal2)

    # Name
    if Input["name_x_axis"] is not None:
        name_x_axis = Input["name_x_axis"]
    else:
        name_x_axis = "Wind speed [m/s]"

    if Input["name_y_axis"] is not None:
        name_y_axis = Input["name_y_axis"]
    else:
        name_y_axis = (f"Bending DEL [Nm] | m = {Input['SN_slope']} "
                       f"| N_ref = {Input['N_ref']:.2e} "
                       f"| lifetime = {Input['design_life']}y")

    x_lims = LIMS["x_lims"]
    y_lims = LIMS["y_lims"]

    cmap_lines = LinearSegmentedColormap.from_list(
        "custom_colormap", ['#ff0000', '#00ff00'])
    keys = Data_Out["Compare"]["added"].columns
    range_colors = np.linspace(0, 1, len(keys))

    axis_2 = axis.twinx()

    ck = list(Data_Out["Compare"]["vm-vise"].keys())[num_plot]

    # read Data
    Force_added_table = {}
    Force_added_points = {}
    Force_added_compare = {}
    Force_vm_table = {}
    Force_vm_points = {}
    Force_vm_compare = {}

    x_edges = Data_Out["Vm_edges"]
    x = (x_edges[1:] + x_edges[:-1]) / 2

    fontsize = 6

    n = 0
    n_text = 0
    point_num = len(Data_Out["set_points"]["segments"][ck])

    barcolor = [0.85, 0.85, 0.85]
    spinecolor = [0.5, 0.5, 0.5]

    axis.set_ylabel('number of datapoints', color=spinecolor)
    axis.tick_params(axis='y')
    axis.spines['left'].set_color(spinecolor)
    axis.yaxis.label.set_color(spinecolor)
    axis.tick_params(axis='y', colors=spinecolor)
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # axis.yaxis.tick_right()
    axis.bar(x, Data_Out["set_table"]["count_table"][ck], width=1,
             bottom=None, align='center', color=barcolor, zorder=-1, alpha=0.5)

    axis_2.spines['left'].set_visible(False)

    for key in keys:
        Force_added_table[key] = Data_Out["set_table"]["DEL"]["added"].loc[ck, key]
        Force_added_points[key] = Data_Out["set_points"]["DEL"]["added"].loc[ck, key]
        Force_added_compare[key] = Data_Out["Compare"]["added"].loc[ck, key]

        Force_vm_table[key] = Data_Out["set_table"]["DEL"]["vm-vise"][ck][key]
        Force_vm_points[key] = Data_Out["set_points"]["DEL"]["vm-vise"][ck][key]
        Force_vm_compare[key] = Data_Out["Compare"]["vm-vise"][ck][key]

        axis_2.plot(x, Force_vm_table[key], label=key + ' condensed', color=cmap_lines(
            range_colors[n]), linestyle='--')
        axis_2.plot(x, Force_vm_points[key],
                    label=key + ' Hindcast', color=cmap_lines(
                range_colors[n]), linestyle='-')

        axis_2.text(0.75, 0.95 - n_text * 0.05,
                    f"{key}", fontweight='bold', transform=axis.transAxes, fontsize=fontsize)
        axis_2.text(0.75, 0.95 - (n_text + 1) * 0.05,
                    f"DEL condensed: {Force_added_table[key]:.3e} Nm", transform=axis.transAxes, fontsize=fontsize)
        axis_2.text(0.75, 0.95 - (n_text + 2) * 0.05,
                    f"DEL Hindcast: {Force_added_points[key]:.3e} Nm", transform=axis.transAxes, fontsize=fontsize)
        axis_2.text(0.75, 0.95 - (n_text + 3) * 0.05,
                    f"condensed/Hindcast: {round(Force_added_compare[key], 1)} %", transform=axis.transAxes,
                    fontsize=fontsize)
        n_text = n_text + 4
        n = n + 1

    axis.set_xlabel(name_x_axis)
    axis_2.set_ylabel(name_y_axis)

    # legend

    axis_2.legend(fontsize=6, loc="lower right", framealpha=0.5)

    axis_2.set_xlim(x_lims[0], x_lims[1])
    axis_2.set_ylim(y_lims[0], y_lims[1])

    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0.01, 0.8, 0.25, 0.15], zorder=-2)
    axin.imshow(image_bgr, zorder=-2)
    axin.axis('off')

    # grid
    axis_2.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4, axis='both')
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4, axis='x')

    # Plot title
    if Input["name_angle"] is not None:
        name_angle = Input["name_angle"]
    else:
        name_angle = ck

    name_angle += f": {round(segments[num_plot][0], 2)}° to {round(segments[num_plot][1], 2)}°" + '\n' + \
                  (f"samples: {point_num:.1e} ({round(point_num / len(DATA) * 100, 1)}%) "
                   f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                   f"| sample rate: {INPUT_SELECT['dt_sample']}")

    axis.set_title(name_angle)

    FIG.append(fig)

    return FIG


def plot_tables(Data_Out, Input):
    VMHS_Data = Data_Out["VMHS"]["table_content"].round(2)
    i = 0
    for col in VMHS_Data.columns[1:]:

        VMHS_Data[col] = VMHS_Data[col].map('{:.2f}'.format)

        if INPUT_SELECT["include_360"] and i == 0:
            col_new = 'omni'
        else:
            col_new = str(round(np.mean(segments[i]), 2)) + '°'

        VMHS_Data.rename({col: col_new}, axis=1, inplace=True)
        i = i + 1

    VMTP_Data = Data_Out["VMTP"]["table_content"].round(2)
    i = 0
    for col in VMTP_Data.columns[1:]:

        VMTP_Data[col] = VMTP_Data[col].map('{:.2f}'.format)

        if INPUT_SELECT["include_360"] and i == 0:
            col_new = 'omni'
        else:
            col_new = str(round(np.mean(segments[i]), 2)) + '°'

        VMTP_Data.rename({col: col_new}, axis=1, inplace=True)
        i = i + 1

    MASK_VMHS = np.array(Data_Out["VMHS"]["isData"])
    MASK_VMTP = np.array(Data_Out["VMTP"]["isData"])

    N_rows = np.size(VMHS_Data.values, 0)

    N_cols = np.size(VMHS_Data.values, 1)

    seg_width = round(segments[-1][1] - segments[-1][0], 2)

    # VMHS
    # Create a figure and axis
    # A4 size in inches (landscape)
    fig_VMHS, ax = plt.subplots(figsize=size_3mal2)

    ax.axis('off')

    ax.axis('tight')

    colums = list(VMHS_Data.columns)

    if Input["Tables"]["name_v_hub"] is None:
        colums[0] = "v_hub [m/s]"
    else:
        colums[0] = Input["Tables"]["name_v_hub"]

    if Input["SELECT"]["include_360"]:
        colums[1] = 'omni'

    table = ax.table(cellText=VMHS_Data.values,
                     bbox=[0, 0, 1, 1]
                     )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=0)

    # Change font color based on boolean mask
    for i in range(N_rows):
        for j in range(1, N_cols):
            if MASK_VMHS[i, j] == 0:
                table[(i, j)].set_text_props(color=[0.5, 0.5, 0.5])
            if np.isnan(float(VMHS_Data.iloc[i, j])):
                table[(i, j)].get_text().set_text('')

    # Alternate background color for every second row
    for i, cell in enumerate(table.get_celld().values()):
        if i % (2 * N_cols) >= N_cols:  # The row index within each pair of rows
            cell.set_facecolor('#f0f0f0')  # Grey background color

    # Center the text vertically and horizontally in each cell
    for key, cell in table.get_celld().items():
        cell.set_text_props(verticalalignment='center',
                            horizontalalignment='center')

    for j, _ in enumerate(list(VMHS_Data.columns)):
        cell = table[(0, j)]
        cell.get_text().set_text(colums[j])
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor(Colors["JBO_green"])  # Blue background color

    y_range = ax.get_ylim()[1]
    ax.text(0, 1.05 * y_range,
            f"H_s [m] for each {seg_width}° wind sector [° from north] ", fontsize=10, horizontalalignment='center',
            fontweight='bold',
            verticalalignment='center')

    fig_VMHS.tight_layout()

    #   save_figs_as_pdf([fig], Input["Tables"]["name_plot"][0])

    # Create a figure and axis
    # A4 size in inches (landscape)
    fig_VMTP, ax = plt.subplots(figsize=size_3mal2)

    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=VMTP_Data.values,
                     bbox=[0, 0, 1, 1]
                     )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=0)

    # Change font color based on boolean mask
    for i in range(N_rows):
        for j in range(1, N_cols):
            if MASK_VMTP[i, j] == 0:
                table[(i, j)].set_text_props(color=[0.5, 0.5, 0.5])
            if np.isnan(float(VMTP_Data.iloc[i, j])):
                table[(i, j)].get_text().set_text('')

    # Alternate background color for every second row
    for i, cell in enumerate(table.get_celld().values()):
        if i % (2 * N_cols) >= N_cols:  # The row index within each pair of rows
            cell.set_facecolor('#f0f0f0')  # Grey background color

    # Center the text vertically and horizontally in each cell
    for key, cell in table.get_celld().items():
        cell.set_text_props(verticalalignment='center',
                            horizontalalignment='center')

    for j, _ in enumerate(list(VMHS_Data.columns)):
        cell = table[(0, j)]
        cell.get_text().set_text(colums[j])
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor(Colors["JBO_green"])  # Blue background color

    y_range = ax.get_ylim()[1]
    ax.text(0, 1.05 * y_range,
            f"T_p [1/s] for each {seg_width}° wind sector [° from north] ", fontsize=10, horizontalalignment='center',
            fontweight='bold',
            verticalalignment='center')
    fig_VMTP.tight_layout()

    return [fig_VMHS, fig_VMTP]


def plot_histo(x, Axlabels, Lims, Input):
    FIG = []
    fig, axis = plt.subplots(1, 1, figsize=size_1mal2_slim)

    # plotlims and scaling
    x_lims = Lims["x_lims"]
    y_lims = Lims["y_lims"]

    x_lim_erg = axis.set_xlim(x_lims)

    range_x = x_lim_erg[1] - x_lim_erg[0]

    # get significant Digits
    x_str = [str(value) for value in x.values]
    lenths = [len(dig.split('.')[1]) for dig in x_str]

    sig_dig = max(lenths)

    if sig_dig > 2: sig_dig = 2

    bins = int(range_x * 10 ** (sig_dig) - 1)

    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0, 0.5, 0.4, 0.40], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)

    # Plot title
    if Input["title_single"] is not None:
        name_angle = Input["title_single"]
    else:
        name_angle = "Histogram " + x.name

    # name_angle += '\n' + (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
    #                       f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
    #                       f"| sample rate: {INPUT_SELECT['dt_sample']} | binsize: {round(range_x / bins, 2)}")

    # plot data
    axis.hist(list(x), bins=bins, range=x_lim_erg, stacked=True, color='black')
    axis.set_ylim(y_lims)

    axis.set_title(name_angle)

    axis.set_xlabel(Axlabels["x"])
    axis.set_ylabel(Axlabels["y"])
    fig.tight_layout()
    FIG.append(fig)

    return FIG


def plot_timeseries(x, Axlabels, Lims, Input):
    FIG = []
    axis = x.plot(color='black', figsize=size_1mal2_slim, linewidth=0.3)
    fig = axis.get_figure()

    # plotlims and scaling
    x_lims = Lims["x_lims"]
    y_lims = Lims["y_lims"]

    x_lim_erg = axis.set_xlim(x_lims)

    range_x = x_lim_erg[1] - x_lim_erg[0]

    # plot data
    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0, 0.5, 0.4, 0.40], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)
    # Plot title
    if Input["title_single"] is not None:
        name_angle = Input["title_single"]
    else:
        name_angle = "Timeseries " + x.name

    name_angle += ('\n' + (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                           f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                           f"| sample rate: {INPUT_SELECT['dt_sample']}") + "\n" +
                   f"mean: {round(np.mean(x), 2)} | standard deviation: {round(np.std(x), 2)}"
                   f" | max value: {round(max(x), 2)} | min value: {round(min(x), 2)}")

    axis.set_ylim(y_lims)

    axis.set_title(name_angle)

    plt.xlabel(Axlabels["x"])
    plt.ylabel(Axlabels["y"])
    fig.tight_layout()

    FIG.append(fig)

    return FIG


def plot_CompAngle_points(Data_Out, Axlabels, Lims, Input):
    x = Data_Out["points"][COLNAMES['angle']]
    y = Data_Out["points"]["Diff_angles"]
    Hs_null = Data_Out["points"]["bool_Hs_null"]
    FIG = []
    fig, axis = plt.subplots(1, 1, figsize=size_1mal2_slim)

    # plot data
    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0, 0.5, 0.4, 0.40], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)
    # Plot title
    if Input["title"] is not None:
        name_angle = Input["title"]
    else:
        name_angle = "Deviation of " + y.name + " from " + x.name

    name_angle += '\n' + (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                          f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                          f"| sample rate: {INPUT_SELECT['dt_sample']}")

    axis.set_title(name_angle)

    axis.set_xlabel(Axlabels["x"])
    axis.set_ylabel(Axlabels["y"])

    c = gl.c_scatterplot(x[~Hs_null], y[~Hs_null])

    axis.scatter(x.values[~Hs_null], y.values[~Hs_null], cmap='cool', c=c, s=3, alpha=0.8)
    if any(Hs_null):
        axis.scatter(x.values[Hs_null], y.values[Hs_null], color='grey', s=3, alpha=0.8,
                     label=f"Significant Wave Heigt = 0 ({len(Hs_null[Hs_null == True])} points, {len(Hs_null[Hs_null == True]) / len(Hs_null[Hs_null == False]) * 100} %)")

    axis.plot(np.array(Data_Out["Angle_mean"]["x"]), Data_Out["Angle_mean"]["mean"], color="black",
              label=f"rolling absolute mean deviation (global: {round(np.mean(Data_Out['Angle_mean']['mean']), 2)} deg)")
    axis.autoscale(enable=True, axis='x', tight=True)

    if Lims["x_lims"][0] is not None: axis.set_xlim(left=Lims["x_lims"][0])
    if Lims["x_lims"][1] is not None: axis.set_xlim(right=Lims["x_lims"][1])
    if Lims["y_lims"][1] is not None: axis.set_ylim(top=Lims["y_lims"][0])
    if Lims["y_lims"][0] is not None: axis.set_ylim(bottom=Lims["y_lims"][1])

    axis.grid(True)
    axis.legend()

    fig.tight_layout()
    FIG.append(fig)

    return FIG


def plot_Extreme_timeseries(x, Data_Out, Axlabels, Lims, Input):
    FIG = []
    axis = x.plot(color="black", figsize=size_1mal2_slim, linewidth=0.3)
    fig = axis.get_figure()

    # plotlims and scaling
    x_lims = Lims["x_lims"]
    y_lims = Lims["y_lims"]

    # plot data
    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0, 0.5, 0.4, 0.40], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)
    # Plot title
    if Input["title_timeseries"] is not None:
        name_angle = Input["title_timeseries"]
    else:
        name_angle = f"Timeseries of {x.name} with {Input['freq_samp']} extreme value(s) per year"
    name_angle += '\n' + (f"samples: {len(x):.1e} ({round(len(x) / len(DATA) * 100, 1)}%) "
                          f"| {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()} "
                          f"| sample rate: {INPUT_SELECT['dt_sample']}")

    axis.set_title(name_angle)

    if Lims["x_lims"] is None:
        axis.set_xlim([x.index[0], x.index[-1]])
    else:
        axis.set_xlim(Lims["x_lims"])
    if Lims["y_lims"] is None:
        axis.set_ylim([x.index[0], x.index[-1]])
    else:
        axis.set_ylim(Lims["y_lims"])

    axis.set_xlabel(Axlabels["x"])
    axis.set_ylabel(Axlabels["y"])

    for x_vline in Data_Out["years"]["newyear"]:
        line_newyear = axis.axvline(x_vline, alpha=0.5, color="grey", linewidth=0.8)

    for x_vline in Data_Out["years"]["start_window"]:
        line_start_window = axis.axvline(x_vline, alpha=0.5, color="blue", linewidth=0.8)

    axis_x_max = Data_Out["points"]["x_max"].plot(style=".", color="red")
    line_x_max, _ = axis_x_max.get_legend_handles_labels()
    axis.set_ylim(bottom=y_lims[0], top=y_lims[1])
    axis.set_xlim(left=x_lims[0], right=x_lims[1])

    axis.grid(True)
    axis.legend([line_x_max[1], line_newyear, line_start_window],
                [f'maximal values ({len(Data_Out["points"]["x_sortet"])})', 'year seperation',
                 f'intervall seperation ({Input["time_window_offeset"]} relative offset)'], loc='upper right')

    fig.tight_layout()
    FIG.append(fig)

    return FIG


def plot_Extreme_qq(Data_Out, Axlabels, Lims, Input):
    x_theorie = Data_Out["points"]["x_theorie"]
    x_sorted = Data_Out["points"]["x_sortet"]
    Hs_lim_up = Data_Out["points"]["perc_up"]
    Hs_lim_low = Data_Out["points"]["perc_down"]

    FIG = []
    fig, axis = plt.subplots(1, 1, figsize=size_1mal1)
    # plot data
    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0.1, 0.4, 0.8, 0.8], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4)
    # Plot title
    if Input["title_qq"] is not None:
        name_angle = Input["title_qq"]
    else:
        name_angle = f"comparison of real and theroretical values (gumbel) of" + "\n" + f"{Input['col_name_values']}"

    name_angle += "\n" + f"samples: {len(DATA): .1e} | {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()}"

    axis.set_title(name_angle)

    axis.set_xlabel(Axlabels["x"])
    axis.set_ylabel(Axlabels["y"])

    axis.scatter(x_theorie, x_sorted, color="red", s=10, label=f"maximal Values ({len(x_theorie)})")
    axis.plot(x_theorie, Hs_lim_low, color="grey", linestyle=":",
              label=f'{Input["perc_down"]}th and {Input["perc_up"]}th percentile ')
    axis.plot(x_theorie, Hs_lim_up, color="grey", linestyle=":")

    y_lims = axis.get_ylim()
    x_lims = axis.get_xlim()
    lim_max = max(y_lims[1], x_lims[1])
    lim_min = min(y_lims[0], x_lims[0])
    axis.set_ylim((lim_min, lim_max))
    axis.set_xlim((lim_min, lim_max))
    axis.plot([0, 100], [0, 100], color="grey")

    if Lims["x_lims"][0] is not None: axis.set_xlim(left=Lims["x_lims"][0])
    if Lims["x_lims"][1] is not None: axis.set_xlim(right=Lims["x_lims"][1])
    if Lims["y_lims"][1] is not None: axis.set_ylim(top=Lims["y_lims"][0])
    if Lims["y_lims"][0] is not None: axis.set_ylim(bottom=Lims["y_lims"][1])

    axis.grid(True)

    axis.legend()

    fig.tight_layout()
    FIG.append(fig)

    return FIG


def plot_Extreme_T_Return(Data_Out, Axlabels, Lims, Input):
    perc_middle = Data_Out["T_R_extra"]["perc_middle"]
    perc_up = Data_Out["T_R_extra"]["perc_up"]
    perc_down = Data_Out["T_R_extra"]["perc_down"]
    T_R_real = Data_Out["points"]["T_R_real"]
    T_R_gumbel = Data_Out["T_R_extra"]["T_r"]
    x_sortet = Data_Out["points"]["x_sortet"]

    FIG = []
    fig, axis = plt.subplots(1, 1, figsize=size_1mal2)

    # plot data
    # JBO-Logo
    with matplotlib.cbook.get_sample_data(path + '\\JBO_logo.png') as file:
        image_bgr = plt.imread(file, format='png')

    axin = axis.inset_axes([0.1, 0.4, 0.4, 0.4], zorder=-1)
    axin.imshow(image_bgr, zorder=-1)
    axin.axis('off')

    # grid
    axis.grid(visible=True, color=[0.7, 0.7, 0.7], zorder=4, which='both')
    # Plot title
    if Input["title_qq"] is not None:
        name_angle = Input["title_qq"]

    else:
        name_angle = f"returnperiods of extreme values of < {Input['col_name_values']} > and extrapolation (gumbel)"

    name_angle += "\n" + f"samples: {len(DATA): .1e} | {DATA.index[0].round('1d').date()} to {DATA.index[-1].round('1d').date()}"
    axis.set_title(name_angle)

    axis.set_xlabel(Axlabels["x"])
    axis.set_ylabel(Axlabels["y"])

    axis.scatter(T_R_real, x_sortet, color="red", s=10, label="real values")
    axis.scatter(Input['T_Return_single'], Data_Out['T_return_single']["middle"], color="blue", s=15,
                 label="expected magnitudes")

    if Input["std_dev"]:
        label_up = "theorie + std_dev"
        label_down = "theorie - std_dev"
    else:
        label_up = f"{Input['perc_up']}th percentile"
        label_down = f"{Input['perc_down']}th percentile"

    if Input["mode_Intervall"] == 1:
        axis.plot(T_R_gumbel[0:-1], perc_middle[0:-1], color="black", label="theory (gumbel)")
        axis.plot(T_R_gumbel[-2:], perc_middle[-2:], color="grey")
        axis.plot(T_R_gumbel[0:-1], perc_up[0:-1], color="black", linestyle="--", label=label_up)
        axis.plot(T_R_gumbel[-2:], perc_up[-2:], color="grey", linestyle="--")
        axis.plot(T_R_gumbel[0:-1], perc_down[0:-1], color="black", linestyle=":",
                  label=label_down)
        axis.plot(T_R_gumbel[-2:], perc_down[-2:], color="grey", linestyle=":")

    if Input["mode_Intervall"] == 2:
        axis.plot(T_R_gumbel, perc_middle, color="black", label="theory (gumbel)")
        axis.plot(T_R_gumbel, perc_up, color="black", linestyle="--",
                  label=label_up)
        axis.plot(T_R_gumbel, perc_down, color="black", linestyle=":",
                  label=label_down)

    axis.set_xscale('log')

    if Lims["x_lims"][0] is not None: axis.set_xlim(left=Lims["x_lims"][0])
    if Lims["x_lims"][1] is not None: axis.set_xlim(right=Lims["x_lims"][1])
    if Lims["y_lims"][1] is not None: axis.set_ylim(top=Lims["y_lims"][0])
    if Lims["y_lims"][0] is not None: axis.set_ylim(bottom=Lims["y_lims"][1])
    n_text = 0
    for year, up, middle, low in zip(Input['T_Return_single'], Data_Out['T_return_single']['up'],
                                     Data_Out['T_return_single']['middle'], Data_Out['T_return_single']['low']):
        axis.text(0.7, 0.3 - n_text * 0.05,
                  f" {year} years:", transform=axis.transAxes, fontsize=6)
        axis.text(0.8, 0.3 - n_text * 0.05, f"{round(low, 1)}", transform=axis.transAxes, fontsize=6)
        axis.text(0.85, 0.3 - n_text * 0.05, f"{round(middle, 1)}", transform=axis.transAxes, fontsize=6)
        axis.text(0.9, 0.3 - n_text * 0.05, f"{round(up, 1)}", transform=axis.transAxes, fontsize=6)
        n_text = n_text + 1

    axis.legend()

    fig.tight_layout()
    FIG.append(fig)

    return FIG
