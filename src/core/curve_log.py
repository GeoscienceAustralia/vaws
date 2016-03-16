# wrapper around non linear regression solver
#    tweaked for martins fragility task
#
import numpy as np
import math
import scipy.special
from optparse import OptionParser
from scipy.optimize.minpack import leastsq
from matplotlib.pyplot import *

MEDIAN = 0
BETA = 1

# --------------------------------------------------------------
def model_func(coeff_arr, x_arr):
    median = coeff_arr[MEDIAN]
    beta = coeff_arr[BETA]
    sqrt2 = math.sqrt(2.0)
    return 0.5 + 0.5 * scipy.special.erf((np.log(x_arr) - np.log(median)) / (beta * sqrt2))   

# --------------------------------------------------------------
def objective_func(coeff_arr, obs_arr, x_arr, verbose):
    diff = obs_arr - model_func(coeff_arr, x_arr)
    return diff

# --------------------------------------------------------------
def plot_func(coeff_arr, x_arr):
    plot(x_arr, model_func(coeff_arr, x_arr))

# -----------------------------------------------------------------
def plot_func_on(widget, coeff_arr, x_arr, col):    
    widget.axes.plot(x_arr, model_func(coeff_arr, x_arr), c=col, linewidth=3)
    widget.axes.figure.canvas.draw()
    widget.axes.set_xlim((0.0, 2.0))
    widget.axes.set_ylim((0.0, 1.0))
    
# --------------------------------------------------------------
def generate_observations(coeff_arr, x_arr, max_perc_err=0):
    yarr = model_func(coeff_arr, x_arr)
    if max_perc_err > 0:
        error_perc_arr = (np.random.randn(len(x_arr))/100.0) * np.random.random_integers(0, max_perc_err, 1)
        yarr += 0.2 * error_perc_arr
    return yarr

# --------------------------------------------------------------
def generate_bad_observations_zero(x_arr, max_perc_err):
    return x_arr * 0.0

# --------------------------------------------------------------
def generate_bad_observations_ramp(x_arr, max_perc_err):
    obs_arr = np.ones(len(x_arr))
    for i in range(0, 10):
        obs_arr[i] = 0
    return obs_arr

# --------------------------------------------------------------
def generate_guess(x_arr, obs_arr):
    return [30.0, 0.5]

# --------------------------------------------------------------  
def solve_for_observations(obs_arr, x_arr, coeff_guess, verbose):
    coeff_final = None
    sum_squared = 0
    if verbose:
        print '\n\nsolve::coeff_guess: ', coeff_guess
    try:
        coeff_final, cov_x, infodict, msg, ier = leastsq(objective_func, 
                                                         coeff_guess, 
                                                         args=(obs_arr, x_arr, verbose), 
                                                         full_output=True)
        if verbose:
            print "number of iterations: ", infodict['nfev']
            print "msg: ", msg
            print "ier: ", ier
            
        if ier in range(1, 5):
            if verbose:
                for i in range(len(coeff_final)):
                    print "A[%d]  =%8.3f +- %.4f (one standard deviation)" % (i, coeff_final[i], np.sqrt(cov_x[i,i]))  
            fitted_vals = model_func(coeff_final, x_arr)
            ss = (fitted_vals - obs_arr)**2
            sum_squared = ss.sum()  
    
    except Exception, e:
        print 'exception: coeff_final(%s) sum_squared(%d) ', (e, coeff_final, sum_squared)
            
    return coeff_final, sum_squared

# --------------------------------------------------------------
def fit_curve(x_arr, obs_arr, verbose=False):
    guess_arr = generate_guess(x_arr, obs_arr)
    coeff_arr, ss =  solve_for_observations(obs_arr, x_arr, guess_arr, verbose)
    if verbose:
        print 'Curve Solution: MEDIAN: %f, BETA: %f\n\n' % (coeff_arr[MEDIAN], coeff_arr[BETA])
    return coeff_arr, ss
    
# --------------------------------------------------------------
if __name__ == '__main__':
    USAGE = '%prog'
    parser = OptionParser(usage=USAGE, version="0.1")
    parser.add_option("-m", "--median", dest="median", action="store", type="float", default=0.3)
    parser.add_option("-b", "--beta", dest="beta", action="store", type="float", default=0.2)
    parser.add_option("-p", dest="plot", action="store_true", help="Plot", default=False)
    parser.add_option("-z", dest="zerogen", action="store_true", default=False)
    parser.add_option("-r", dest="rampgen", action="store_true", default=False)
    parser.add_option("-f", "--fake_error", dest="fake_error", action="store", help="Fake observations", type="float", default=0.0)
    parser.add_option("--xrange", dest="xrange", action="store", help="X range", type="float", nargs=2)
    parser.add_option("--xsteps", dest="xsteps", action="store", help="X steps", type="int", default=100)
    parser.add_option("--verbose", dest="verbose", action="store_true", help="Print more information", default=False)
    (options, args) = parser.parse_args()
    
    if options.xrange is None:
        options.xrange = [0.0, 2.0]        
    x_arr = np.linspace(options.xrange[0], options.xrange[1], options.xsteps)
    
    coeff_arr = [options.median, options.beta]
    
    obs_arr = None
    if options.zerogen:
        obs_arr = generate_bad_observations_zero(x_arr, options.fake_error)
    elif options.rampgen:
        obs_arr = generate_bad_observations_ramp(x_arr, options.fake_error)
    else:
        obs_arr = generate_observations(coeff_arr, x_arr, options.fake_error)

    coeff_solution = fit_curve(x_arr, obs_arr, options.verbose)
    
    if options.plot:
        hold(True)
        if not options.zerogen and not options.rampgen:
            plot_func(coeff_arr, x_arr)
        scatter(x_arr, obs_arr)
        plot_func(coeff_solution, x_arr)
        show()

