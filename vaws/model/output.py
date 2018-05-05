"""
    output.py - output module, postprocess and plot display engine
"""
from numpy import linspace, ceil
import matplotlib.pyplot as plt
from matplotlib import cm, colorbar, colors, ticker
from matplotlib.collections import PatchCollection
import matplotlib as mpl

import logging


def plot_heatmap(grouped, values, vmin, vmax, vstep, xlim_max, ylim_max,
                 file_name=None):
    """

    Args:
        grouped: pd.DataFrame (x_coord, y_coord, width, height)
        values: np.array(N,1)
        vmin: min. of scale in color bar
        vmax: max. of scale in color bar
        vstep: no. of scales in color bar
        file_name: file to save

    Returns:

    """

    group_key = grouped['group_name'].unique()[0]

    # xlim_max = 6
    # ylim_max = 8

    # xlim_max = (grouped['x_coord'] + 0.5 * grouped['width']).max()
    # ylim_max = (grouped['y_coord'] + 0.5 * grouped['height']).max()

    fig = plt.figure()

    left = 0.1
    bottom = 0.2
    width = 1.0 - left * 2.0
    height = 0.75
    ax1 = fig.add_axes([left, bottom, width, height])

    #
    # xticks = list(set(xticks))
    # yticks = list(set(yticks))
    # xticks.sort()
    # yticks.sort()

    left = 0.1
    bottom = 0.1
    width = 1.0 - left * 2.0
    height = 0.05
    ax2 = fig.add_axes([left, bottom, width, height])

    cmap = cm.jet_r
    # cmap = cm.YlOrRd_r
    bounds = linspace(vmin, vmax, vstep)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_under('gray')

    cb = colorbar.ColorbarBase(ax2,
                               cmap=cmap,
                               norm=norm,
                               spacing='proportional',
                               ticks=bounds,
                               boundaries=bounds,
                               format='%.1f',
                               orientation='horizontal')

    cb.set_label('Wind speed (m/s)', size=10)
    cb.ax.tick_params(labelsize=8)

    try:
        p = PatchCollection(grouped['coords'].tolist(), cmap=cmap, norm=norm)
    except AttributeError:
        logging.warning('coords are not provided in the connections.csv')

    else:
        p.set_array(values)
        ax1.add_collection(p)

        for irow, row in grouped.iterrows():

            # patches = [row['coords']]
            # p = PatchCollection(patches, label=row['zone_loc'])
            #p.set_array(values)
            # ax1.add_collection(p)
            ax1.annotate(irow, row['centroid'], color='w', weight='bold',
                         fontsize=8, ha='center', va='center')

        ax1.set_title('Heatmap of failure wind speed for {}'.format(group_key))

        ax1.set_xlim([0, xlim_max])
        ax1.set_ylim([0, ylim_max])
        ax1.set_xbound(lower=0.0, upper=xlim_max)
        ax1.set_ybound(lower=0.0, upper=ylim_max)

        # Hide major tick labels
        ax1.xaxis.set_major_formatter(ticker.NullFormatter())
        ax1.yaxis.set_major_formatter(ticker.NullFormatter())
        ax1.tick_params(axis=u'both', which=u'both', length=0)

        # Customize minor tick labels
        # ax1.xaxis.set_minor_locator(ticker.FixedLocator(xticks))
        # _list = [num2str(i) for i in range(1, len(xticks) + 1)]
        # ax1.xaxis.set_minor_formatter(ticker.FixedFormatter(_list))
        #
        # ax1.yaxis.set_minor_locator(ticker.FixedLocator(yticks))
        # _list = [i for i in range(1, len(yticks) + 1)]
        # ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(_list))

    if file_name:
        fig.savefig('{}.png'.format(file_name), dpi=150)
    plt.close(fig)


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
        _str = '{}:{}'.format(key, value)

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

            dic_ax[ax_key].add_collection(p)


def set_axis_etc(ax, xlim_max, ylim_max):

    ax.set_xlim([0, xlim_max])
    ax.set_ylim([0, ylim_max])
    ax.set_xbound(lower=0.0, upper=xlim_max)
    ax.set_ybound(lower=0.0, upper=ylim_max)

    # Hide major tick labels
    ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.tick_params(axis=u'both', which=u'both', length=0)


def plot_influence(cfg, conn_name, file_name=None):

    fig = plt.figure()

    #fig.axes.figure.clf()

    try:
        infl_dic = cfg.influences[conn_name]
    except KeyError:
        logging.warning('influence is not defined for {}'.format(conn_name))
    else:
        _list_groups = [cfg.connections.loc[conn_name].group_name]
        for key in infl_dic.keys():
            if key in cfg.zones:
                _list_groups.append('zone')
            elif cfg.connections.loc[key].group_name not in _list_groups:
                _list_groups.append(cfg.connections.loc[key].group_name)

        no_rows = ceil(len(_list_groups) / 2.0)

        dic_ax = {}
        for i, name in enumerate(_list_groups, 1):
            dic_ax[name] = fig.add_subplot(no_rows, 2, i)

        xlim_max, ylim_max = cfg.house['length'], cfg.house['width']

        # zone
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

        draw_influence(cfg, infl_dic, dic_ax, conn_name)

    finally:

        for item in _list_groups:
            dic_ax[item].set_title('{}'.format(item))
            set_axis_etc(dic_ax[item], xlim_max, ylim_max)

        fig.canvas.draw()

    if file_name:
        fig.savefig('{}.png'.format(file_name), dpi=150)
    fig.close()

'''
def plot_curve(x_value, y_value, **kwargs):
    plt.plot(x_value, y_value, **kwargs)


def plot_show(axis_range, *args, **kwargs):
    """

    Args:
        axis_range: [xmin, xmax, ymin, ymax] 
        *args: no_models
        **kwargs: title, xlabel, ylabel, filename, loc

    Returns:

    """
    plt.axis(axis_range)
    if args:
        no_sims = args[0]
        plt.title('{} (n = {:d})'.format(kwargs['title'], no_sims))
    else:
        plt.title(kwargs['title'])

    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.legend(loc=kwargs.get('loc', 2))
    if 'filename' in kwargs:
        plt.savefig(kwargs['filename'])
    plt.close('all')
'''