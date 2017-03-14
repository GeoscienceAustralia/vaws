"""
    output.py - output module, postprocess and plot display engine
"""
from matplotlib import cm, colorbar, colors
import matplotlib.pyplot as plt
import version
import numpy as np


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


def plot_pdf(y):
    plt.hist(y, bins=50)
    

def plot_wind_event_damage(v, di):
    plt.scatter(v, di, s=8, marker='+', label='_nolegend_')


def plot_wind_event_mean(v, di):
    plt.scatter(v, di, s=20, c='r', marker='o', label="Means")
    

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
    
"""
def testme():
    print "output.py says hi"
    n = 100
    scatter([10]*n, normal(0.2, 0.02, n))
    scatter([15]*n, normal(0.25, 0.02, n))
    scatter([20]*n, normal(0.35, 0.02, n))
    scatter([25]*n, normal(0.40, 0.03, n))
    scatter([30]*n, normal(0.60, 0.04, n))
    show()

if __name__ == '__main__': testme()
"""
