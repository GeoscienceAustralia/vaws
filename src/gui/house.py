from house_ui import Ui_Dialog
import mixins 
from vaws import scenario

from PyQt4.QtCore import *
from PyQt4.QtGui import *


class HouseViewer(QDialog, Ui_Dialog, mixins.PersistSizePosMixin):
    def __init__(self, house, parent=None):
        super(HouseViewer, self).__init__(parent)
        mixins.PersistSizePosMixin.__init__(self, "HouseViewer")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle('House Viewer: ' + house.house_name)
        self.ui.buttonBox.addButton(QDialogButtonBox.Ok)        
        self.ui.replacementCost.setText('$ %.2f' % house.replace_cost)
        self.ui.height.setText('%.3f' % house.height)
        self.ui.cpeV.setText('%.3f' % house.cpe_V)
        self.ui.cpeK.setText('%.3f' % house.cpe_k)
        self.ui.cpeStructV.setText('%.3f' % house.cpe_struct_V)
        self.ui.width.setText('%.3f' % house.width)
        self.ui.length.setText('%.3f' % house.length)

        # fill in walls table
        self.ui.tree.clear()
        parentFromWalls = {}
        for w in house.walls:
            # add wall if it's not already there
            ancestor = parentFromWalls.get(w)
            if ancestor is None:
                ancestor = QTreeWidgetItem(self.ui.tree, [scenario.Scenario.dirs[w.direction-1], 'wall', '%.3f'%w.area])
                parentFromWalls[w] = ancestor
            # add children of wall        
            for cov in w.coverages:
                item = QTreeWidgetItem(ancestor, ['', cov.description, '%.3f'%cov.area, cov.type.name])
        self.ui.tree.resizeColumnToContents(0)
        self.ui.tree.resizeColumnToContents(1)  
        
    def accept(self):
        QDialog.accept(self)
        
    def reject(self):
        QDialog.reject(self)
        
        
if __name__ == '__main__':
    import sys
    from vaws import database
    database.configure()
    s = scenario.Scenario(20, 40.0, 120.0, 60.0, '2')
    s.setHouseName('Group 4 House')
    app = QApplication(sys.argv)
    myapp = HouseViewer(s.house)
    myapp.show()
    app.exec_()
