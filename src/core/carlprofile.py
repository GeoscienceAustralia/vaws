'''
Adapted from article: http://stackoverflow.com/questions/1171166/how-can-i-profile-a-sqlalchemy-powered-application
'''

import cProfile as profiler
import gc, pstats, time

def profile(fn):
    def wrapper(*args, **kw):
        elapsed, stat_loader, result = _profile("foo.txt", fn, *args, **kw)
        stats = stat_loader()
        stats.sort_stats('cumulative')
        stats.print_stats(30)
        stats.sort_stats('time')
        stats.print_stats(30)
        # stats.print_callers()
        return result
    return wrapper

def _profile(filename, fn, *args, **kw):
    load_stats = lambda: pstats.Stats(filename)
    gc.collect()
    began = time.time()
    profiler.runctx('result = fn(*args, **kw)', globals(), locals(), filename=filename)
    ended = time.time()
    return ended - began, load_stats, locals()['result']

@profile
def testme():
    import scipy.stats
    for i in xrange(100000):
        num_items = scipy.stats.poisson.rvs(50)

if __name__ == '__main__':
    testme()