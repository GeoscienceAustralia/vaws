import sys
import numpy
from scipy.optimize.minpack import leastsq
import matplotlib.pyplot as matplot
import pylab

# various functions
def single_exponential(A, t):
    return A[0] + A[1] * numpy.exp(-t/A[2])

def objective(A, t, y0, func):
    return y0 - func(A, t)

n = 50
t = numpy.linspace(0.1, 8.0, n)
x0 = [1.0, 134.0, 1.6]
y = single_exponential(x0, t)

# read data
datafile = "decay.txt"
data = pylab.mlab.load(datafile)
t_exp = data[:,0]
y_exp = data[:,1]

# define cost function - adapt to your usage
#
# single exponential
function = single_exponential
x0 = [0., y_exp[0], 1e-1]
param = (t_exp, y_exp, function)

# perform least squares fit
A_final, cov_x, infodict, msg, ier = leastsq(objective, x0, args=param, full_output=True, warning=True)
if ier != 1:
    print "No fit!"
    print msg
    sys.exit(0)
y_final = function(A_final, t_exp)
chi2 = sum((y_exp-y_final)**2 / y_final)

# print resulting parameters and their std. deviations
print "Optimized parameters:"
resultfile = file(datafile + ".result", "w")
for i in range(len(A_final)):
    print>>resultfile, "# A[%d]  =%8.3f +- %.4f" % (i, A_final[i], numpy.sqrt(cov_x[i,i]))
    print "A[%d]  =%8.3f +- %.4f" % (i, A_final[i], numpy.sqrt(cov_x[i,i]))
print>>resultfile, "# chi^2 =%8.3f" % (chi2,)
print "chi^2 =", chi2

# plot data (must be last)
matplot.scatter(t_exp, y_exp)
matplot.plot(t_exp, y_final)
#matplot.plot(t_exp, y_exp-y_final)
matplot.show()
