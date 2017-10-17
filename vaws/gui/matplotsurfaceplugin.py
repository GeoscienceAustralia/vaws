# Adapted from 2d example
from PyQt4.QtGui import QIcon
from PyQt4.QtDesigner import QPyDesignerCustomWidgetPlugin

import os
from matplotlib import rcParams
from matplotsurfacewidget import MatplotSurfaceWidget

rcParams['font.size'] = 9

class MatplotSurfacePlugin(QPyDesignerCustomWidgetPlugin):
    def __init__(self, parent=None):
        QPyDesignerCustomWidgetPlugin.__init__(self)
        self._initialized = False
    def initialize(self, formEditor):
        if self._initialized:
            return
        self._initialized = True
    def isInitialized(self):
        return self._initialized
    def createWidget(self, parent):
        return MatplotSurfaceWidget(parent)
    def name(self):
        return "MatplotSurfaceWidget"
    def group(self):
        return "Python(x,y)"
    def icon(self):
        image = os.path.join(rcParams['datapath'], 'images', 'matplotlib.png')
        return QIcon(image)
    def toolTip(self):
        return ""
    def whatsThis(self):
        return ""
    def isContainer(self):
        return False
    def domXml(self):
        return '<widget class="MatplotSurfaceWidget" name="mplsurface">\n' \
               '</widget>\n'
    def includeFile(self):
        return "matplotsurfacewidget"


if __name__ == '__main__':
    import sys
    from PyQt4.QtGui import QApplication
    app = QApplication(sys.argv)
    widget = MatplotSurfacePlugin()
    widget.show()
    sys.exit(app.exec_())
