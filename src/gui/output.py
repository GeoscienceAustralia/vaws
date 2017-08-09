'''
    output.py - output module, postprocess and plot display engine
'''
import matplotlib as mpl
from matplotlib import cm, colors
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

import numpy as np

from vaws import house
from vaws.zone import Zone

class PlotFlyoverCallback(object):
    def __init__(self, axes, data, statusbar, num_cols, num_rows):
        self.axes = axes
        self.data = data
        self.statusbar = statusbar
        self.num_rows = num_rows
        self.num_cols = num_cols

    def __call__(self, event):
        if event.inaxes != self.axes:
            return
        col = int(event.xdata + 0.5)
        if col > self.num_cols - 1:
            return
        row = int(event.ydata + 0.5)
        if row > self.num_rows - 1:
            return
        locStr = 'Average damage @ %.3f m/sec for Zone %s' % (self.data[row, col], zone.getZoneLocFromGrid(col, row))
        self.statusbar.showMessage(locStr)


class PlotClickCallback(object):
    def __init__(self, figure, axes, plotKey, owner, aHouse, walldir=None):
        self.figure = figure
        self.axes = axes
        self.plotKey = plotKey
        self.house = aHouse
        self.owner = owner
        self.walldir = walldir
        self.selected, = axes.plot([0], [0], 'ws', ms=12, alpha=1.0, visible=False, markeredgecolor='black',
                                   markeredgewidth=2)

    def select(self, col, row):
        self.selected.set_visible(True)
        self.selected.set_data(col, row)
        self.figure.canvas.draw()

    def __call__(self, event):
        if event.inaxes != self.axes:
            return

        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)

        mplDict['south'].selected.set_visible(False)
        mplDict['north'].selected.set_visible(False)
        mplDict['east'].selected.set_visible(False)
        mplDict['west'].selected.set_visible(False)

        self.select(col, row)

        zoneLoc = zone.getZoneLocFromGrid(col, row)

        if self.walldir is not None:
            if self.walldir == 'south':
                zoneLoc = 'WS%s' % (zoneLoc)
            elif self.walldir == 'north':
                zoneLoc = 'WN%s' % (zoneLoc)
            elif self.walldir == 'east':
                zoneLoc = 'WE%d%d' % (col + 2, row + 1)
            elif self.walldir == 'west':
                zoneLoc = 'WW%d%d' % (col + 2, row + 1)

        self.owner.onSelectZone(zoneLoc)

        if zoneLoc in house.zoneByLocationMap:
            z = house.zoneByLocationMap[zoneLoc]
            self.owner.onZoneSelected(z, self.plotKey)

            ctg = house.ctgMap[self.plotKey]
            for c in z.located_conns:
                if c.ctype.group.id == ctg.id:
                    self.owner.onSelectConnection(c.connection_name)
                    break


def plot_wall_damage_show(fig,
                          wall_south_V,
                          wall_north_V,
                          wall_west_V,
                          wall_east_V,
                          num_major_cols, num_major_rows,
                          num_minor_cols, num_minor_rows,
                          v_min, v_max):
    # add the legend colorbar axes to base of plot
    fig.clf()
    fig.canvas.draw()
    left = 0.1
    bottom = left
    width = (1.0 - left * 2.0)
    height = 0.05
    axLegend = fig.add_axes([left, bottom, width, height])
    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    cb1 = mpl.colorbar.ColorbarBase(axLegend, cmap=cmap, norm=norm, orientation='horizontal')
    cb1.set_label('Wind Speed')

    # setup south ticks
    xticks = range(0, num_major_cols)
    xticklabels = []
    for tick in xticks:
        base = 'WS'
        letter = chr(ord('A') + tick)
        xticklabels.append(base + letter)
    yticks = range(0, num_major_rows)
    yticklabels = range(1, num_major_rows + 1)

    # add the south wall
    bottom = 0.2
    height = 0.1
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(wall_south_V, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    mplDict['south'] = PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house'], 'south')
    fig.canvas.mpl_connect('motion_notify_event',
                           PlotFlyoverCallback(axPlot, wall_south_V, mplDict['statusbar'], num_major_cols,
                                               num_major_rows))
    fig.canvas.mpl_connect('button_press_event', mplDict['south'])

    # setup north ticks
    xticklabels = []
    for tick in xticks:
        base = 'WN'
        letter = chr(ord('A') + tick)
        xticklabels.append(base + letter)

    # add the north wall
    bottom = 0.8
    height = 0.1
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(wall_north_V, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    mplDict['north'] = PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house'], 'north')
    fig.canvas.mpl_connect('motion_notify_event',
                           PlotFlyoverCallback(axPlot, wall_north_V, mplDict['statusbar'], num_major_cols,
                                               num_major_rows))
    fig.canvas.mpl_connect('button_press_event', mplDict['north'])

    # setup west ticks
    xticks = range(0, num_minor_cols)
    xticklabels = []
    for tick in xticks:
        xticklabels.append('WW%d' % (tick + 2))
    yticks = range(0, num_minor_rows)
    yticklabels = range(1, num_minor_rows + 1)

    # add the west wall
    bottom = 0.4
    height = 0.1
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(wall_west_V, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    mplDict['west'] = PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house'], 'west')
    fig.canvas.mpl_connect('motion_notify_event',
                           PlotFlyoverCallback(axPlot, wall_west_V, mplDict['statusbar'], num_minor_cols,
                                               num_minor_rows))
    fig.canvas.mpl_connect('button_press_event', mplDict['west'])

    # setup east ticks
    xticklabels = []
    for tick in xticks:
        xticklabels.append('WE%d' % (tick + 2))

    # add the east wall
    bottom = 0.6
    height = 0.1
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(wall_east_V, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    mplDict['east'] = PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house'], 'east')
    fig.canvas.mpl_connect('motion_notify_event',
                           PlotFlyoverCallback(axPlot, wall_east_V, mplDict['statusbar'], num_minor_cols,
                                               num_minor_rows))
    fig.canvas.mpl_connect('button_press_event', mplDict['east'])

    fig.canvas.draw()


def plot_damage_show(fig, grouped, values_grid, xlim_max, ylim_max,
                     v_min, v_max, v_step, file_name=None):

    fig.axes.figure.clf()
    fig.canvas.draw()

    # add the legend colorbar axes
    left = 0.1
    bottom = 0.1
    width = (1.0 - left * 2.0)
    height = 0.05
    axLegend = fig.figure.add_axes([left, bottom, width, height])

    cmap = cm.jet_r
    bounds = np.linspace(v_min, v_max, v_step)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_under('gray')
    cb1 = mpl.colorbar.ColorbarBase(axLegend.axes,
                                    cmap=cmap,
                                    norm=norm,
                                    spacing='proportional',
                                    ticks=bounds,
                                    format='%.1f',
                                    orientation='horizontal')
    cb1.set_label('Wind speed (m/s)', size=10)
    cb1.ax.tick_params(labelsize=8)

    # add the heatmap
    left = 0.1
    bottom = 0.2
    width = (1.0 - left * 2.0)
    height = 0.75
    axPlot = fig.figure.add_axes([left, bottom, width, height])

    for irow, row in grouped.iterrows():
        axPlot.annotate(irow, row['centroid'], color='w', weight='bold',
                        fontsize=8, ha='center', va='center')

    p = PatchCollection(grouped['coords'].tolist(), cmap=cmap, norm=norm)
    p.set_array(values_grid)
    axPlot.add_collection(p)

    axPlot.set_xlim([0, xlim_max])
    axPlot.set_ylim([0, ylim_max])
    axPlot.set_xbound(lower=0.0, upper=xlim_max)
    axPlot.set_ybound(lower=0.0, upper=ylim_max)
    axPlot.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    axPlot.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    axPlot.tick_params(axis=u'both', which=u'both', length=0)

    group_key = grouped['group_name'].unique()[0]
    axPlot.set_title('Heatmap of damage capacity for {}'.format(group_key))
    axPlot.format_coord = format_coord

    # fig.canvas.mpl_connect('motion_notify_event',
    #                        PlotFlyoverCallback(axPlot, v_damaged_at, mplDict['statusbar'], numCols, numRows))
    # fig.canvas.mpl_connect('button_press_event',
    #                        PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house']))
    fig.canvas.draw()

    if file_name:
        fig.savefig('{}.png'.format(file_name), dpi=150)
        plt.close(fig)


def plot_wind_event_damage(mp_widget, v, di):
    mp_widget.axes.scatter(v, di, s=8, marker='+', label='_nolegend_')


def plot_wind_event_mean(mp_widget, v, di):
    mp_widget.axes.scatter(v, di, s=20, c='r', marker='o', label="Means")


def plot_fitted_curve(mp_widget, v, di, label="Fitted Curve", alpha=1.0, col='b'):
    mp_widget.axes.plot(v, di, label=label, alpha=alpha, c=col, linestyle='-')


def plot_model_curve(mp_widget, v, di, label="Model Curve"):
    mp_widget.axes.plot(v, di, label=label)


def plot_wind_event_show(mp_widget, num_iters, Vmin, Vmax):
    mp_widget.axes.set_title('Vulnerability Curve (n = %d)' % (num_iters))
    mp_widget.axes.set_xlabel('Impact Wind speed (m/s)')
    mp_widget.axes.set_ylabel('Damage Index')
    mp_widget.axes.set_xlim((Vmin, Vmax))
    mp_widget.axes.set_ylim((0.0, 1.1))
    mp_widget.axes.figure.canvas.draw()


def plot_fragility_curve(v, fl, label, alpha=1.0, col='b'):
    mplDict['fragility'].axes.plot(v, fl, label=label, c=col, alpha=alpha)


def plot_fragility_show(mp_widget, num_iters, Vmin, Vmax):
    mp_widget.axes.set_title('Fragility Curve (n = %d)' % num_iters)
    mp_widget.axes.set_xlabel('Impact Wind speed (m/s)')
    mp_widget.axes.set_ylabel('Probability of Damage State')
    mp_widget.axes.set_xlim((Vmin, Vmax))
    mp_widget.axes.set_ylim((0.0, 1.0))
    mp_widget.axes.legend(loc=2,
                          fancybox=True,
                          shadow=True,
                          fontsize='small')

    mp_widget.axes.figure.canvas.draw()


def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)
    return Zone.get_zone_location_from_grid((col, row,))
