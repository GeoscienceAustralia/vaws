'''
    Curve module - fun with curves
'''
import numpy
from scipy.optimize.minpack import leastsq
 
# --------------------------------------------------------------
def single_exponential_given_V(A, x_arr):
    if x_arr >= 0:
        value = numpy.power(x_arr/numpy.exp(A[0]), 1/A[1])
    else:
        value = 0.0
    return 1 - numpy.exp(-value)

# --------------------------------------------------------------
def objective(A, x_arr, obs_arr):
    y = single_exponential_given_V(A, x_arr)
    diff = obs_arr - y
    return diff

# --------------------------------------------------------------
def generate_observations(coeff_arr, x_arr, max_perc_err=0):
    yarr = single_exponential_given_V(coeff_arr, x_arr)
    if max_perc_err > 0:
        error_perc_arr = (numpy.random.randn(len(x_arr))/100.0) * numpy.random.random_integers(0, max_perc_err, 1)
        yarr += 0.2 * error_perc_arr
    return yarr

# --------------------------------------------------------------
def calc_alpha_beta(ws1, di1, ws2, di2):
    a = numpy.log(ws1/ws2)
    if di1 == 1.0: di1 = 0.99   # for the call to log
    if di2 == 1.0: di2 = 0.99   # for the call to log
    b = -numpy.log(1 - di1)
    c = -numpy.log(1 - di2)
    d = numpy.log(b/c)
    alpha = (a/d) 
    beta = numpy.log( ws1 / numpy.power(b, alpha) ) 
    return alpha, beta
    
# --------------------------------------------------------------
def generate_guess(wind_speeds, damage_indexes):
    i = numpy.searchsorted(damage_indexes, 0.01, side='right')
    if i == damage_indexes.size:
        ws0 = wind_speeds[0]
        di0 = damage_indexes[0]
    else:
        ws0 = wind_speeds[i]
        di0 = damage_indexes[i]
        
    i = numpy.searchsorted(damage_indexes, 0.2, side='right')
    if i == damage_indexes.size:
        ws1 = wind_speeds[0]
        di1 = damage_indexes[0]
    else:
        ws1 = wind_speeds[i]
        di1 = damage_indexes[i]
        
    i = numpy.searchsorted(damage_indexes, 0.9, side='right')
    if i == damage_indexes.size:
        ws2 = wind_speeds[damage_indexes.size-3]
        di2 = damage_indexes[damage_indexes.size-3]        
    else:
        ws2 = wind_speeds[i]
        di2 = damage_indexes[i]
    
    a1, b1 = calc_alpha_beta(ws2, di2, ws1, di1)
    a2, b2 = calc_alpha_beta(ws2, di2, ws0, di0)
    return (a1+a2)/2, (b1+b2)/2

# --------------------------------------------------------------  
def fit_curve(x_arr, obs_arr, verbose=False):
    guess_alpha, guess_beta = generate_guess(x_arr, obs_arr)
    guess_x0 = [guess_beta, guess_alpha]            
    try:
        A_final, cov_x, infodict, msg, ier = leastsq(objective, 
                                                     guess_x0, 
                                                     args=(x_arr, obs_arr), 
                                                     full_output=True, 
                                                     warning=True)
    except Exception, e:
        return guess_x0, 0
    if ier not in range(1, 5):
        return guess_x0, 0
    
    fitted_vals = single_exponential_given_V(A_final, x_arr)
    ss = (fitted_vals - obs_arr)**2  
    return A_final, ss.sum()
    
