
import numpy

# ------------------------------------------------------------
def readArrayFromCSV(filename, dtype_string, separator=',', skiprows=1, numharvest=-1):
    """ Read a file with an arbitrary number of columns.
        The type of data in each column is arbitrary
        It will be cast to the given dtype at runtime
    """
    dtype = numpy.dtype(dtype_string)
    cast = numpy.cast
    data = [[] for dummy in xrange(len(dtype))]
    skip = skiprows
    harvested = 0

    for line in open(filename, 'r'):
        skip = skip - 1
        if skip >= 0: continue
        if line[0] == '#': continue
        if line[0] == ' ': continue        
        fields = line.strip().split(separator)
        for i, number in enumerate(fields): 
            data[i].append(number)
        harvested += 1    
        if harvested == numharvest:
            break
            
    for i in xrange(len(dtype)): 
        data[i] = cast[dtype[i]](data[i])
            
    return numpy.rec.array(data, dtype=dtype)

