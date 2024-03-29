'''
    output.py - output module, postprocess and plot display engine
'''
from matplotlib import cm, colors, ticker, colorbar
from matplotlib.collections import PatchCollection

import logging
from numpy import ceil, linspace

'''
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
'''


def plot_damage_show(fig, grouped, values_grid, xlim_max, ylim_max,
                     v_min, v_max, v_step, house_number=0):

    logger = logging.getLogger(__name__)

    fig.axes.figure.clf()
    fig.canvas.draw()

    # add the legend colorbar axes
    left = 0.1
    bottom = 0.1
    width = (1.0 - left * 2.0)
    height = 0.05
    axLegend = fig.figure.add_axes([left, bottom, width, height])

    cmap = cm.jet_r
    bounds = linspace(v_min, v_max, v_step)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_under('gray')
    cb1 = colorbar.ColorbarBase(axLegend.axes,
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

    try:
        p = PatchCollection(grouped['coords'].tolist(), cmap=cmap, norm=norm)
    except AttributeError:
        logger.warning('Heatmap can not be drawn due to missing coordinates')
    else:
        p.set_array(values_grid)
        p.set_edgecolor('black')
        axPlot.add_collection(p)

        for irow, row in grouped.iterrows():
            axPlot.annotate(irow, row['centroid'], color='w', weight='bold',
                            fontsize=8, ha='center', va='center')

    axPlot.set_xlim([0, xlim_max])
    axPlot.set_ylim([0, ylim_max])
    axPlot.set_xbound(lower=0.0, upper=xlim_max)
    axPlot.set_ybound(lower=0.0, upper=ylim_max)
    axPlot.xaxis.set_major_formatter(ticker.NullFormatter())
    axPlot.yaxis.set_major_formatter(ticker.NullFormatter())
    axPlot.tick_params(axis=u'both', which=u'both', length=0)
    # axPlot.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
    # axPlot.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())

    group_key = grouped['group_name'].unique()[0]
    if house_number == 0:
        axPlot.set_title(f'Heatmap of failure wind speed for {group_key}')
    else:
        axPlot.set_title(f'Heatmap of failure wind speed for {group_key} of model {house_number}')
    # axPlot.format_coord = format_coord

    fig.canvas.draw()

    return axPlot


def plot_load_show(fig, grouped, values_grid, xlim_max, ylim_max, v_min, v_max, v_step=10):

    logger = logging.getLogger(__name__)

    fig.axes.figure.clf()
    fig.canvas.draw()

    # add the legend colorbar axes
    left = 0.1
    bottom = 0.1
    width = (1.0 - left * 2.0)
    height = 0.05
    axLegend = fig.figure.add_axes([left, bottom, width, height])

    cmap = cm.jet_r
    bounds = linspace(v_min, v_max, v_step)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_under('gray')
    cb1 = colorbar.ColorbarBase(axLegend.axes,
                                    cmap=cmap,
                                    norm=norm,
                                    spacing='proportional',
                                    ticks=bounds,
                                    format='%.2f',
                                    orientation='horizontal')
    cb1.set_label('Load (kN)', size=10)
    cb1.ax.tick_params(labelsize=8)

    # add the heatmap
    left = 0.1
    bottom = 0.2
    width = (1.0 - left * 2.0)
    height = 0.75
    axPlot = fig.figure.add_axes([left, bottom, width, height])

    try:
        p = PatchCollection(grouped['coords'].tolist(), cmap=cmap, norm=norm)
    except AttributeError:
        logger.warning('Plot can not be drawn due to missing coordinates')
    else:
        p.set_array(values_grid)
        p.set_edgecolor('black')
        axPlot.add_collection(p)

        for irow, row in grouped.iterrows():
            axPlot.annotate(irow, row['centroid'], color='w', weight='bold',
                            fontsize=8, ha='center', va='center')

        axPlot.set_xlim([0, xlim_max])
        axPlot.set_ylim([0, ylim_max])
        axPlot.set_xbound(lower=0.0, upper=xlim_max)
        axPlot.set_ybound(lower=0.0, upper=ylim_max)
        axPlot.xaxis.set_major_formatter(ticker.NullFormatter())
        axPlot.yaxis.set_major_formatter(ticker.NullFormatter())
        axPlot.tick_params(axis=u'both', which=u'both', length=0)
        # axPlot.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        # axPlot.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())

    # group_key = grouped['group_name'].unique()[0]
    # if house_number == 0:
    #     axPlot.set_title('Heatmap of failure wind speed for {}'.format(group_key))
    # else:
    #     axPlot.set_title('Heatmap of failure wind speed for {} of model {} '.format(group_key,
    #                                                                                 house_number))
    # # axPlot.format_coord = format_coord

    fig.canvas.draw()


def plot_pressure_show(fig, groups, values, xlim_max, ylim_max, v_min, v_max, v_step=10):

    logger = logging.getLogger(__name__)

    fig.axes.figure.clf()
    fig.canvas.draw()

    # add the legend colorbar axes
    left = 0.1
    bottom = 0.1
    width = (1.0 - left * 2.0)
    height = 0.05
    axLegend = fig.figure.add_axes([left, bottom, width, height])

    cmap = cm.jet_r
    bounds = linspace(v_min, v_max, v_step)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_under('gray')
    cb1 = colorbar.ColorbarBase(axLegend.axes,
                                    cmap=cmap,
                                    norm=norm,
                                    spacing='proportional',
                                    ticks=bounds,
                                    format='%.2f',
                                    orientation='horizontal')
    cb1.set_label('Value', size=10)
    cb1.ax.tick_params(labelsize=8)

    # add the heatmap
    left = 0.1
    bottom = 0.2
    width = (1.0 - left * 2.0)
    height = 0.75
    axPlot = fig.figure.add_axes([left, bottom, width, height])

    try:
        _coord = [item[1] for item in groups]
        p = PatchCollection(_coord, cmap=cmap, norm=norm)
    except AttributeError:
        logger.warning('Plot can not be drawn due to missing coordinates')
    else:
        p.set_array(values)
        p.set_edgecolor('black')
        axPlot.add_collection(p)

        for item in groups:
            axPlot.annotate(item[0], item[2], color='w', weight='bold',
                            fontsize=8, ha='center', va='center')

        axPlot.set_xlim([0, xlim_max])
        axPlot.set_ylim([0, ylim_max])
        axPlot.set_xbound(lower=0.0, upper=xlim_max)
        axPlot.set_ybound(lower=0.0, upper=ylim_max)
        axPlot.xaxis.set_major_formatter(ticker.NullFormatter())
        axPlot.yaxis.set_major_formatter(ticker.NullFormatter())
        axPlot.tick_params(axis=u'both', which=u'both', length=0)
        # axPlot.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        # axPlot.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())

    fig.canvas.draw()


def plot_wind_event_damage(ax, v, di):
    ax.scatter(v, di, s=8, c='k', marker='+', label='_nolegend_', alpha=0.8)


def plot_wind_event_mean(ax, v, di):
    ax.scatter(v, di, s=20, c='r', marker='o', label="Mean")


def plot_fitted_curve(ax, v, di, label="Fitted Curve", alpha=1.0, col='b'):
    ax.plot(v, di, label=label, alpha=alpha, c=col, linestyle='-')


def plot_model_curve(ax, v, di, label="Model Curve"):
    ax.plot(v, di, label=label)


def plot_wind_event_show(ax, num_iters, Vmin, Vmax):
    ax.set_title('Vulnerability Curve (n = %d)' % (num_iters))
    ax.set_xlabel('Impact Wind speed (m/s)')
    ax.set_ylabel('Damage Index')
    ax.set_xlim((Vmin, Vmax))
    ax.set_ylim((0.0, 1.1))
    # mp_widget.figure.canvas.draw()


def plot_fragility_show(ax, num_iters, Vmin, Vmax):

    ax.set_title(f'Fragility Curve (n = {num_iters})')
    ax.set_xlabel('Impact Wind speed (m/s)')
    ax.set_ylabel('Probability of Damage State')
    ax.set_xlim((Vmin, Vmax))
    ax.set_ylim((0.0, 1.0))
    # ax.legend(loc=2, fancybox=True, shadow=True, fontsize='small')



# def format_coord(x, y):
#     col = int(x + 0.5)
#     row = int(y + 0.5)
#     return Zone.get_zone_location_from_grid((col, row,))


def plot_influence(fig, cfg, conn_name, file_name=None):

    logger = logging.getLogger(__name__)

    fig.axes.figure.clf()

    try:
        infl_dic = cfg.influences[conn_name]
    except KeyError:
        if conn_name in cfg.connections:
            logger.warning(f'influence is not defined for {conn_name}')
        else:
            logger.info(f'skipped: conn {conn_name} is not defined')
    else:
        _list_groups = [cfg.connections.loc[conn_name].group_name]
        for key in infl_dic.keys():
            try:
                _group_name = cfg.connections.loc[key].group_name
            except KeyError:
                if (key in cfg.zones) and ('zone' not in _list_groups):
                    _list_groups.append('zone')
            else:
                if _group_name not in _list_groups:
                    _list_groups.append(_group_name)

        no_rows = ceil(len(_list_groups) / 2.0)

        dic_ax = {}
        for i, name in enumerate(_list_groups, 1):
            dic_ax[name] = fig.figure.add_subplot(no_rows, 2, i)

        for group_name in _list_groups:

            if group_name == 'zone':
                for key, value in cfg.zones.items():
                    try:
                        p = PatchCollection([value['coords']], facecolors='none')
                    except AttributeError:
                        pass
                    else:
                        dic_ax['zone'].add_collection(p)
            else:
                grouped = cfg.connections.loc[
                    cfg.connections['group_name'] == group_name]
                p = PatchCollection(grouped['coords'].tolist(), facecolors='none')
                dic_ax[group_name].add_collection(p)

        dic_ax = draw_influence(cfg, infl_dic, dic_ax, conn_name)

        for item in _list_groups:
            set_axis_etc(dic_ax[item],
                         title=item,
                         xlim_max=cfg.house['length'],
                         ylim_max=cfg.house['width'])

    finally:
        fig.canvas.draw()

    if file_name:
        fig.savefig(f'{file_name}.png', dpi=150)


def set_axis_etc(ax, title, xlim_max, ylim_max):

    ax.set_title(title)
    ax.set_xlim([0, xlim_max])
    ax.set_ylim([0, ylim_max])
    ax.set_xbound(lower=0.0, upper=xlim_max)
    ax.set_ybound(lower=0.0, upper=ylim_max)

    # Hide major tick labels
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.tick_params(axis=u'both', which=u'both', length=0)


def plot_influence_patch(fig, cfg, failed_conn_name, conn_name, file_name=None):

    logger = logging.getLogger(__name__)

    fig.axes.figure.clf()

    try:
        infl_dic = cfg.influence_patches[failed_conn_name][conn_name]
    except KeyError:
        logger.warning(f'influence patch is not defined for {failed_conn_name}:{conn_name}')
    else:

        _list_groups = [cfg.connections.loc[failed_conn_name].group_name]
        _name = cfg.connections.loc[conn_name].group_name
        if _name not in _list_groups:
            _list_groups.append(_name)

        for key in infl_dic.keys():
            try:
                _group_name = cfg.connections.loc[key].group_name
            except KeyError:
                if (key in cfg.zones) and ('zone' not in _list_groups):
                    _list_groups.append('zone')
            else:
                if _group_name not in _list_groups:
                    _list_groups.append(_group_name)

        no_rows = ceil(len(_list_groups) / 2.0)

        dic_ax = {}
        for i, name in enumerate(_list_groups, 1):
            dic_ax[name] = fig.figure.add_subplot(no_rows, 2, i)

        for group_name in _list_groups:

            if group_name == 'zone':
                for key, value in cfg.zones.items():
                    try:
                        p = PatchCollection([value['coords']], facecolors='none')
                    except AttributeError:
                        pass
                    else:
                        dic_ax['zone'].add_collection(p)
            else:
                grouped = cfg.connections.loc[
                    cfg.connections['group_name'] == group_name]
                p = PatchCollection(grouped['coords'].tolist(), facecolors='none')
                dic_ax[group_name].add_collection(p)

        # failed connection: grey
        target = cfg.connections.loc[failed_conn_name]
        p = PatchCollection([target['coords']], facecolors='grey')
        dic_ax[target.group_name].add_collection(p)
        dic_ax[target.group_name].annotate(failed_conn_name, target['centroid'],
                                           color='k', weight='bold',
                                           fontsize='small', ha='center', va='center')

        draw_influence(cfg, infl_dic, dic_ax, conn_name)

        for item in _list_groups:
            set_axis_etc(dic_ax[item],
                         title=item,
                         xlim_max=cfg.house['length'],
                         ylim_max=cfg.house['width'])

    finally:
        fig.canvas.draw()

    if file_name:
        fig.savefig(f'{file_name}.png', dpi=150)


def draw_influence(cfg, infl_dic, dic_ax, conn_name):

    # target: blue
    target = cfg.connections.loc[conn_name]
    p = PatchCollection([target['coords']], facecolors='skyblue')
    dic_ax[target.group_name].add_collection(p)
    dic_ax[target.group_name].annotate(conn_name, target['centroid'],
                                       color='k', weight='bold',
                                       fontsize='small', ha='center',
                                       va='center')
    # influences: red
    for key, value in infl_dic.items():

        face_color, font_weight, font_size = 'orange', 'bold', 'x-small'
        _str = f'{key}:{value}'

        if key in cfg.zones:
            item = cfg.zones[key]
            ax_key = 'zone'
        else:
            item = cfg.connections.loc[key]
            ax_key = item['group_name']

        try:
            p = PatchCollection([item['coords']], facecolors=face_color)
        except AttributeError:
            pass
        else:
            try:
                dic_ax[ax_key].annotate(_str, item['centroid'], color='k',
                                    weight=font_weight,
                                    fontsize=font_size, ha='center', va='center')
            except KeyError:
                pass
            else:
                dic_ax[ax_key].add_collection(p)

    return dic_ax
