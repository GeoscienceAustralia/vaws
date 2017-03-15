'''
    regress_poly - example of using SciPy polynomal regression technique
'''
from scipy import *
from pylab import *

n = 50
t = linspace(-5, 5, n)
a = -0.5; b = 0; c = 0

x = polyval([a,b,c],t)
xn = x + randn(n)

(ar,br,cr) = polyfit(t, xn, 2)
xr = polyval([ar,br,cr], t)
err = sqrt(sum((xr-xn)**2)/n)

print('Linear regression using polyfit')
print('parameters: a=%.2f b=%.2f c=%.2f \nregression: a=%.2f b=%.2f c=%.2f, ms error= %.3f' % (
        a,b,c,ar,br,cr,err))
plot(t, x, 'g')
plot(t, xn, 'k.')
plot(t, xr, 'r')
show()
