'''
    output.py - output module, postprocess and plot display engine
'''
from numpy.random import normal
from matplotlib.pyplot import *
import version

# ------------------------------------------------------------
def plot_damage_show(plotKey, v_damaged_at, numCols, numRows, v_min, v_max):
    pass
    
# ------------------------------------------------------------
def plot_pdf(y):
    hist(y, bins=50)
    
# ------------------------------------------------------------
def plot_wind_event_damage(v, di):
    scatter(v, di, s=8, marker='+', label='_nolegend_')

# ------------------------------------------------------------
def plot_wind_event_mean(v, di):
    scatter(v, di, s=20, c='r', marker='o', label="Means")
    
# ------------------------------------------------------------
def plot_fitted_curve(v, di, label="Fitted Curve", alpha=1.0, col='b'):
    plot(v, di, label=label, alpha=alpha, c=col)
    
# ------------------------------------------------------------
def plot_model_curve(v, di, label="Model Curve"):
    plot(v, di, label=label)
    
# ------------------------------------------------------------
def plot_fragility_curve(v, fl, label, alpha=1.0, col='b'):
    plot(v, fl, label=label, c=col, alpha=alpha)

# ------------------------------------------------------------
def plot_fragility_show(num_iters, Vmin, Vmax, output_folder):
    axis([Vmin, Vmax, 0, 1.0])
    title('Fragility Curve (n = %d)' % num_iters)
    xlabel('Impact Wind speed (m/s)')
    ylabel('Probability of Damage State')
    legend(loc=2)
    if output_folder:
        savefig(output_folder + "/windfrag_curve.png")
    else:
        show()
    close('all')

# ------------------------------------------------------------
def plot_boundary_profile(z, m, p):
    label = 'Profile %s' % (p)
    plot(z, m, label=label)
    
# ------------------------------------------------------------
def plot_boundary_profile_show(tcat):
    axis([0, 1.4, 0, 35])
    title('Gust envelope profiles (normalized) TC %s' % (tcat))
    legend(loc=2)
    xlabel('Mz,cat 2/M10, cat %s' % (tcat))
    ylabel('Height(m)')
    show()
    
# ------------------------------------------------------------
def plot_damage_costs(v, dmg, index, col):
    shapes = ['s', 'o', 'd']
    scatter(v, dmg, marker=shapes[index], c=col)
    #hist(v)
    
# ------------------------------------------------------------
def plot_damage_costs_show(Vmin, Vmax):
    title('Repair Cost Calculations - %s' % ("everything"))
    #axis([Vmin, Vmax, 0, 300000])
    xlabel('Impact Wind speed (m/s)')
    ylabel('Damage Cost($)')
    show()
    
# ------------------------------------------------------------
def plot_wind_qz_show():
    title('QZ Scatter')
    xlabel('Impact Wind speed (m/s)')
    ylabel('QZ')
    show()
    
# ------------------------------------------------------------
def plot_wind_event_show(num_iters, Vmin, Vmax, output_folder):
    axis([Vmin, Vmax, 0, 1.2])
    title('Vulnerability Curve (n = %d): %s' % (num_iters, version.VERSION_DESC))
    xlabel('Impact Wind speed (m/s)')
    ylabel('Damage Index')
    legend(loc=2)
    if output_folder:
        savefig(output_folder + "/windvuln_curve.png")
        close('all')
    else:
        show()
            
# ------------------------------------------------------------
def plot_show(show_legend=False):
    if show_legend:
        legend(loc=2)
    show()
    
# ------------------------------------------------------------
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
    