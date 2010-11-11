import math

def lognormal_mean(m, stddev): 
    return math.log(m) - (0.5 * math.log(1.0 + (stddev*stddev)/(m*m)))

def lognormal_stddev(m, stddev):
    return math.sqrt(math.log((stddev*stddev)/(m*m) + 1))

def lognormal_underlying_mean(m, stddev):
    if m == 0 or stddev == 0:
        return 0
    return math.exp(m + 0.5*stddev*stddev)

def lognormal_underlying_stddev(m, stddev):
    if m == 0 or stddev == 0:
        return 0
    return math.sqrt((math.exp(stddev*stddev)-1.0) * math.exp(2.0*m + stddev*stddev))
    