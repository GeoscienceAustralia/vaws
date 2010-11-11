'''
    Generate Debris Sources module
        - given radius (max debris distance), angle (of debris cone), building spacing and method (grid or staggered)
        - output array of (x,y) sources (ie houses)
'''
import math
import numpy

# --------------------------------------------------------------
def genGrid(radius, angle, spacing, restrict_yord=False):
    srcs = []   
    xord = spacing
    yord = 0
    yordlim = radius/6.0
    while xord <= radius:
        srcs.append((xord,yord))
        yordmax = xord * math.tan(math.radians(angle)/2.0)
        if restrict_yord:
            yordmax = yordlim if yordlim <= yordmax else yordmax
        while yord < yordmax:
            yord = yord + spacing
            if yord <= yordmax:
                srcs.append((xord, yord))
                srcs.append((xord, -yord))
        yord = 0
        xord += spacing
    return srcs

# --------------------------------------------------------------
def genStaggered(radius, angle, spacing, restrict_yord=False):
    srcs = []   
    xord = spacing
    yord = 0
    yordlim = radius/6.0
    while xord <= radius:
        yordmax = xord * math.tan(math.radians(angle)/2.0)
        if restrict_yord:
            yordmax = yordlim if yordlim <= yordmax else yordmax
        if int(xord/spacing) % 2 == 1:
            srcs.append((xord,yord))
            yord = yord + spacing
            while yord <= yordmax:
                srcs.append((xord, yord))
                srcs.append((xord, -yord))
                yord += spacing
        else:
            yord = yord + spacing/2
            while yord <= yordmax:
                srcs.append((xord, yord))
                srcs.append((xord, -yord))
                yord += spacing
        yord = 0
        xord += spacing
    return srcs
    
# -------------------------------------------------------------- unit tests
if __name__ == '__main__':    
    import unittest
    
    class SourceGeneratorTestCase(unittest.TestCase):
        def test_grid(self):
            self.assertEquals(len(genGrid(100.0, 45.0, 20.0)), 13)
            
        def test_staggered(self):
            self.assertEquals(len(genStaggered(100.0, 45.0, 20.0)), 15)
            
        def test_plot(self):
            from matplotlib.pyplot import show, scatter
            sources = genStaggered(100.0, 45.0, 20.0, True)
            a = numpy.array(sources)
            scatter(a[:,0], a[:,1], s=50)
            show()
            
    suite = unittest.TestLoader().loadTestsFromTestCase(SourceGeneratorTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
    



