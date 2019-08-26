# Imports
from PyQt5.QtWidgets import QWidget, QSizePolicy, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')

# Matplotlib canvas class to create figure
class MatplotlibCanvas(Canvas):
    def __init__(self, parent=None, title=None, xlabel=None, ylabel=None,
                 xlim=None, ylim=None, hold=False, xscale='linear', yscale='linear',
                 width=4, height=3):
        self.figure = Figure(figsize=(width, height))
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if xscale is not None:
            self.axes.set_xscale(xscale)
        if yscale is not None:
            self.axes.set_yscale(yscale)
        if xlim is not None:
            self.axes.set_xlim(*xlim)
        if ylim is not None:
            self.axes.set_ylim(*ylim)

        if not hold:
            self.figure.clear()

        super().__init__(self.figure)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

# Matplotlib widget
class MatplotlibWidget(QWidget):
    def __init__(self, parent=None, title=None, xlabel=None, ylabel=None,
                 xlim=None, ylim=None, hold=False, xscale='linear', yscale='linear',
                 width=4, height=3):
        super().__init__(parent)   # Inherit from QWidget
        self.canvas = MatplotlibCanvas(
                parent=self,
                title=title, xlabel=xlabel, ylabel=ylabel, hold=hold,
                xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale,
                width=width, height=height) # Create canvas object
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QVBoxLayout(self)         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)

        self.axes = self.canvas.axes
        self.figure = self.canvas.figure

if __name__ == '__main__':
    #   Example
    import sys
    from PyQt5.QtWidgets import QMainWindow, QApplication
    from numpy import linspace

    class ApplicationWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.mplwidget = MatplotlibWidget(self, title='Example',
                                              xlabel='Linear scale',
                                              ylabel='Log scale',
                                              hold=True, yscale='log')
            self.mplwidget.setFocus()
            self.setCentralWidget(self.mplwidget)
            self.plot(self.mplwidget.axes)

        def plot(self, axes):
            x = linspace(-10, 10)
            axes.plot(x, x**2)
            axes.plot(x, x**3)

    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())
