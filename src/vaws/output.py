"""
    output.py - output module, postprocess and plot display engine
"""
from matplotlib import cm, colorbar, colors, ticker
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import matplotlib.pyplot as plt
import numpy as np

import version
from zone import num2str


def plot_heatmap(grouped, values, vmin, vmax, vstep, file_name):
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

    xlim_max = (grouped['x_coord'] + 0.5 * grouped['width']).max()
    ylim_max = (grouped['y_coord'] + 0.5 * grouped['height']).max()

    patches, xticks, yticks = [], [], []

    # assumed rectangle but should be polygon later
    for _, row in grouped.iterrows():
        rect = Rectangle((row['x_coord'] - row['width'] / 2.0,
                          row['y_coord'] - row['height'] / 2.0),  # (x, y)
                         row['width'],  # width
                         row['height'],  # height
                         )
        xticks.append(row['x_coord'])
        yticks.append(row['y_coord'])
        patches.append(rect)

    xticks = list(set(xticks))
    yticks = list(set(yticks))
    xticks.sort()
    yticks.sort()

    fig = plt.figure()

    left = 0.1
    bottom = 0.2
    width = 1.0 - left * 2.0
    height = 0.75
    ax1 = fig.add_axes([left, bottom, width, height])

    left = 0.1
    bottom = 0.1
    width = 1.0 - left * 2.0
    height = 0.05
    ax2 = fig.add_axes([left, bottom, width, height])

    cmap = cm.jet_r
    bounds = np.linspace(vmin, vmax, vstep)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_under('gray')

    cb = colorbar.ColorbarBase(ax2,
                               cmap=cmap,
                               norm=norm,
                               spacing='proportional',
                               ticks=bounds,
                               boundaries=bounds,
                               format='%.1f',
                               orientation='horizontal',
                               )
    cb.set_label('Wind speed (m/s)', size=10)
    cb.ax.tick_params(labelsize=8)

    p = PatchCollection(patches, cmap=cmap, norm=norm)
    p.set_array(values)
    ax1.add_collection(p)
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
    ax1.xaxis.set_minor_locator(ticker.FixedLocator(xticks))
    _list = [num2str(i) for i in range(1, len(xticks) + 1)]
    ax1.xaxis.set_minor_formatter(ticker.FixedFormatter(_list))

    ax1.yaxis.set_minor_locator(ticker.FixedLocator(yticks))
    _list = [i for i in range(1, len(yticks) + 1)]
    ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(_list))

    fig.savefig('{}.png'.format(file_name), dpi=150)
    plt.close(fig)


def plot_damage_show(plotKey, v_damaged_at, numCols, numRows, v_min,
                     v_max, file_name):
    """

    Args:
        plotKey:
        id_sim:
        v_damaged_at:
        numCols:
        numRows:
        v_min:
        v_max:
        file_name:

    Returns:

    """
    xticks = range(numCols)
    xticklabels = [chr(ord('A')+tick) for tick in xticks]
    yticks = range(numRows)
    yticklabels = range(1, numRows+1)

    # add the legend colorbar axes
    fig = plt.figure()
    fig.clf()
    fig.canvas.draw()
    left = 0.1
    bottom = left
    width = (1.0 - left*2.0)
    height = 0.05
    axLegend = fig.add_axes([left, bottom, width, height])
    cmap = cm.jet_r
    # norm = colors.Normalize(vmin=v_min, vmax=v_max)
    bounds = np.linspace(v_min, v_max, 21)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cb1 = colorbar.ColorbarBase(axLegend, cmap=cmap, norm=norm,
                                orientation='horizontal')
    cb1.set_label('Wind Speed')
    cb1.ax.tick_params(labelsize=1)

    # add the heatmap
    left = 0.1
    bottom = 0.2
    width = (1.0 - left*2.0)
    height = 0.75
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(v_damaged_at, cmap=cm.jet_r, interpolation='nearest',
                  origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    axPlot.set_title('Wind Speed Damaged At Heatmap For {}'.format(plotKey))
    # not required for static map
    # axPlot.format_coord = format_coord
    fig.savefig('{}.png'.format(file_name))
    plt.close(fig)


def plot_fitted_curve(v, di, label="Fitted Curve", alpha=1.0, col='b'):
    plt.plot(v, di, label=label, alpha=alpha, c=col)
    

def plot_model_curve(v, di, label="Model Curve"):
    plt.plot(v, di, label=label)
    

def plot_fragility_curve(v, fl, label, alpha=1.0, col='b'):
    plt.plot(v, fl, label=label, c=col, alpha=alpha)


def plot_fragility_show(num_iters, Vmin, Vmax, output_folder):
    plt.axis([Vmin, Vmax, 0, 1.0])
    plt.title('Fragility Curve (n = %d)' % num_iters)
    plt.xlabel('Impact Wind speed (m/s)')
    plt.ylabel('Probability of Damage State')
    plt.legend(loc=2)
    if output_folder:
        plt.savefig(output_folder + "/windfrag_curve.png")
    else:
        plt.show()

    plt.close('all')


def plot_boundary_profile(z, m, p):
    label = 'Profile %s' % (p)
    plt.plot(z, m, label=label)
    

def plot_boundary_profile_show(tcat):
    plt.axis([0, 1.4, 0, 35])
    plt.title('Gust envelope profiles (normalized) TC %s' % (tcat))
    plt.legend(loc=2)
    plt.xlabel('Mz,cat 2/M10, cat %s' % (tcat))
    plt.ylabel('Height(m)')
    plt.show()
    

def plot_damage_costs(v, dmg, index, col):
    shapes = ['s', 'o', 'd']
    plt.scatter(v, dmg, marker=shapes[index], c=col)
    plt.hist(v)
    

def plot_damage_costs_show(Vmin, Vmax):
    plt.title('Repair Cost Calculations - %s' % ("everything"))
    plt.axis([Vmin, Vmax, 0, 300000])
    plt.xlabel('Impact Wind speed (m/s)')
    plt.ylabel('Damage Cost($)')
    plt.show()
    

def plot_wind_qz_show():
    plt.title('QZ Scatter')
    plt.xlabel('Impact Wind speed (m/s)')
    plt.ylabel('QZ')
    plt.show()
    

def plot_wind_event_show(num_iters, Vmin, Vmax, output_folder):
    plt.axis([Vmin, Vmax, 0, 1.2])
    plt.title('Vulnerability Curve (n = %d): %s' % (num_iters, version.VERSION_DESC))
    plt.xlabel('Impact Wind speed (m/s)')
    plt.ylabel('Damage Index')
    plt.legend(loc=2)
    if output_folder:
        plt.savefig(output_folder + "/windvuln_curve.png")
        plt.close('all')
    else:
        plt.show()
            

def plot_show(show_legend=False):
    if show_legend:
        plt.legend(loc=2)
        plt.show()
    
