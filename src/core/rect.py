'''
    2D Rect Library - for management of building footprints.
        - rotate
        - point within bounds
        - points contain plotting state (pyplot)
        - rects can render as patch with associated pointsets (useful for rendering debris maps)
'''
import math
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
    
## -----------------------------------------------------------------------------
class Point:
    def __init__(self, x=0.0, y=0.0, col='b', shape='s', size=20, alpha=1.0):
        self.x = x
        self.y = y
        self.set_plot(col, shape, size, alpha)
        
    def set_plot(self, col, shape, size, alpha):
        self.plot_color = col
        self.plot_shape = shape
        self.plot_size = size
        self.plot_alpha = alpha

    def __str__( self ):
        return "Point(%.3f, %.3f)" % (self.x, self.y)
    
    def __add__(self, p):
        return Point(self.x+p.x, self.y+p.y)

    def __sub__(self, p):
        return Point(self.x-p.x, self.y-p.y)

    def __mul__(self, scalar):
        return Point(self.x*scalar, self.y*scalar)

    def __div__(self, scalar):
        return Point(self.x/scalar, self.y/scalar)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def distance_to(self, p):
        return (self - p).length()

    def as_tuple(self):
        return (self.x, self.y)

    def clone(self):
        return Point(self.x, self.y)

    def integerize(self):
        self.x = int(self.x)
        self.y = int(self.y)

    def floatize(self):
        self.x = float(self.x)
        self.y = float(self.y)

    def move_to(self, x, y):
        self.x = x
        self.y = y

    def slide(self, p):
        self.x = self.x + p.x
        self.y = self.y + p.y

    def slide_xy(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy

    def rotate(self, rad):
        """Rotate counter-clockwise by rad radians.
        """
        s, c = [f(rad) for f in (math.sin, math.cos)]
        x, y = (c*self.x - s*self.y, s*self.x + c*self.y)
        return Point(x,y)

    def rotate_about(self, p, theta):
        """Rotate counter-clockwise around a point, by theta degrees.
        """
        result = self.clone()
        result.slide(-p.x, -p.y)
        result.rotate(theta)
        result.slide(p.x, p.y)
        return result

## --------------------------------------------------------------------------
class Rect:
    def __init__(self, length, width, rotation=0, extension=0):
        self.rotation = rotation
        self.extension = extension
        self.length = length
        self.width = width
        self.generate_points()
        if self.rotation != 0:
            self.rotate(self.rotation)
            if self.rotation > 0:
                self.points[0].slide_xy(-self.extension, 0)
            else:
                self.points[3].slide_xy(-self.extension, 0)

    def set_points(self, points):
        self.points = points
        
    def generate_points(self):
        self.points = []
        l = self.length/2.0
        w = self.width/2.0        
        if self.rotation == 0:    
            self.points.append(Point(-l-self.extension, w))
            self.points.append(Point(l, w))
            self.points.append(Point(l, -w))
            self.points.append(Point(-l-self.extension, -w))
        else:
            self.points.append(Point(-l, w))
            self.points.append(Point(l, w))
            self.points.append(Point(l, -w))
            self.points.append(Point(-l, -w))
        self.origin_points = self.points[:]
       
    def contains(self, pt):
        if self.rotation != 0:
            pt = pt.rotate(math.radians(-self.rotation))
        x,y = pt.as_tuple()
        return self.origin_points[0].x <= x <= self.origin_points[1].x and self.origin_points[2].y <= y <= self.origin_points[0].y

    def top_left(self):
        return self.points[0]

    def bottom_right(self):
        return self.points[2]

    def rotate(self, deg):
        points = []
        for pt in self.points:
            points.append(pt.rotate(math.radians(deg)))
        self.set_points(points)
        return self
        
    def get_vertices(self):
        vertices = [(self.top_left().x, self.top_left().y), 
                    (self.points[1].x, self.points[1].y), 
                    (self.bottom_right().x, self.bottom_right().y), 
                    (self.points[3].x, self.points[3].y), 
                    (0,0)]
        return np.array(vertices, float)
        
    def get_path(self):
        codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
        return Path(self.get_vertices(), codes)

    def render(self, title, points=[], srcs=[]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_patch(PathPatch(self.get_path(), edgecolor='blue', alpha=0.3))
        pt1 = Point()
        ax.scatter(pt1.x, pt1.y, marker='+') 
        for src in srcs:
            ax.scatter(src.x, src.y, c=src.plot_color, marker=src.plot_shape, s=src.plot_size, alpha=src.plot_alpha)
        for pt in points:
            ax.scatter(pt.x, pt.y, c=pt.plot_color, marker=pt.plot_shape, s=pt.plot_size, alpha=pt.plot_alpha)
        ax.set_title(title)
        fig.canvas.draw()
        ax.axes.set_xlim((-100, 100))
        ax.axes.set_ylim((-100, 100))
        plt.show()
        
    def __str__( self ):
        return "<Rect (%s,%s,%s,%s)>" % (self.points[0], self.points[1], self.points[2], self.points[3]) 
        
# -------------------------------------------------------------- unit tests
if __name__ == '__main__':
    import unittest
    
    class MyTestCase(unittest.TestCase):
        def test_basic(self):
            rot = -45.0
            ext = 20.0
            r = Rect(8.0, 23.0, rot, ext)
            points = []
            points.append(Point(-6.45, -1.56))
            points.append(Point(6.45, -1.56))
            points.append(Point(10.88, 4.514))
            points.append(Point(7.773, 12.66))
            r.render('Sample Debris Field', points, [Point(5,5,'k','s',500,0.4)])
            
        def test_contains(self):
            rect = Rect(30, 8)
            self.assertTrue(rect.contains(Point()))
            self.assertFalse(rect.contains(Point(-100, -1.56)))
            self.assertFalse(rect.contains(Point(10.88, 4.514)))
            self.assertFalse(rect.contains(Point(7.773, 12.66)))
            rect = Rect(30, 8, 45)
            self.assertTrue(rect.contains(Point(-6.45, -1.56)))
            self.assertFalse(rect.contains(Point(6.45, -1.56)))
            self.assertFalse(rect.contains(Point(10.88, 4.514)))
            self.assertTrue(rect.contains(Point(7.773, 12.66)))
            
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
            
            
            
            
            
            
            
            
            
            
            
            