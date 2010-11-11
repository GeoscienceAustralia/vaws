'''
    regress_linear - example of using SciPy stats linear regression technique
'''
from scipy import *
from pylab import *
from scipy import stats

n = 50
t = linspace(-5, 5, n)

a = 0.8; b = -4
x = polyval([a,b],t)
xn = x + randn(n)

(a_s, b_s, r_val, p_val, stderr) = stats.linregress(t, xn)
print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f, std error= %.3f' % (
        a,b,a_s,b_s,stderr))
print "R-squared", r_val**2
print "p-value", p_val
