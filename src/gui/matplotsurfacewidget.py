'''
    Matplotlib 3D Surface widget adapted from provided matplot 2d widget example.
'''
__version__ = "1.0.0"

from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import QSize
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.size'] = 9

class MatplotSurfaceWidget(Canvas):
    def __init__(self, parent=None, title='', xlabel='', ylabel='', zlabel='',
                 xlim=None, ylim=None, zlim=None, xscale='linear', yscale='linear',
                 width=4, height=3, dpi=100, hold=False):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        
        Canvas.__init__(self, self.figure)
        self.setParent(parent)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
        
        self.axes = Axes3D(self.figure)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_zlabel(zlabel)
        if xscale is not None:
            self.axes.set_xscale(xscale)
        if yscale is not None:
            self.axes.set_yscale(yscale)
        if xlim is not None:
            self.axes.set_xlim3d(*xlim)
        if ylim is not None:
            self.axes.set_ylim3d(*ylim)
        if zlim is not None:
            self.axes.set_zlim3d(*zlim)
        self.axes.hold(hold)
        

    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(10, 10)
    

#    def save_figure( self ):
#        filetypes = self.canvas.get_supported_filetypes_grouped()
#        sorted_filetypes = filetypes.items()
#        sorted_filetypes.sort()
#        default_filetype = self.canvas.get_default_filetype()
#
#        start = "image." + default_filetype
#        filters = []
#        selectedFilter = None
#        for name, exts in sorted_filetypes:
#            exts_list = " ".join(['*.%s' % ext for ext in exts])
#            filter = '%s (%s)' % (name, exts_list)
#            if default_filetype in exts:
#                selectedFilter = filter
#            filters.append(filter)
#        filters = ';;'.join(filters)
#
#        fname = QtGui.QFileDialog.getSaveFileName(
#            self, "Choose a filename to save to", start, filters, selectedFilter)
#        if fname:
#            try:
#                self.canvas.print_figure( unicode(fname) )
#            except Exception, e:
#                QtGui.QMessageBox.critical(
#                    self, "Error saving file", str(e),
#                    QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)
                
#===============================================================================
#   Example
#===============================================================================
def on_press(event):
    print 'you pressed', event.button, event.xdata, event.ydata
    
if __name__ == '__main__':
    import sys
    import random
    import numpy as np
    from matplotlib import cm
    from PyQt4.QtGui import QMainWindow, QApplication
    
    class ApplicationWindow(QMainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.mplwidget = MatplotSurfaceWidget(self, title='Example',
                                              xlabel='A-Z',
                                              ylabel='1-N',
                                              zlabel='Breakage V',
                                              hold=True)
            self.mplwidget.setFocus()
            self.setCentralWidget(self.mplwidget)
            self.plot(self.mplwidget.axes)
            
            cid = self.mplwidget.mpl_connect('button_press_event', on_press)
            
        def plot(self, axes):
            X = np.arange(-5, 5, 0.25)
            Y = np.arange(-5, 5, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X**2 + Y**2)
            Z = np.sin(R)
            axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.cool_r)
        
    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())
