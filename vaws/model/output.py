"""
    output.py - output module, postprocess and plot display engine
"""
from numpy import linspace
import matplotlib.pyplot as plt
from matplotlib import cm, colorbar, colors, ticker
from matplotlib.collections import PatchCollection


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

    p = PatchCollection(grouped['coords'].tolist(), cmap=cmap, norm=norm)
    p.set_array(values)
    ax1.add_collection(p)

    for irow, row in grouped.iterrows():

        # patches = [row['coords']]
        # p = PatchCollection(patches, label=row['zone_loc'])
        #p.set_array(values)
        # ax1.add_collection(p)
        ax1.annotate(irow, row['centroid'], color='w', weight='bold',
                     fontsize=8, ha='center', va='center')

    ax1.set_title('Heatmap of damage capacity for {}'.format(group_key))

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
