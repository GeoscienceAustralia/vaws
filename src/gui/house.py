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
        self.ui.replacementCost.setText('$ %.2f' % house.df_house['replace_cost'][0])
        self.ui.height.setText('%.3f' % house.df_house['height'][0])
        self.ui.cpeV.setText('%.3f' % house.df_house['cpe_cov'][0])
        self.ui.cpeK.setText('%.3f' % house.df_house['cpe_k'][0])
        self.ui.cpeStructV.setText('%.3f' % house.df_house['cpe_str_cov'][0])
        self.ui.width.setText('%.3f' % house.df_house['width'][0])
        self.ui.length.setText('%.3f' % house.df_house['length'][0])

        # fill in walls table
        self.ui.tree.clear()
        parentFromWalls = {}
        # TODO when DEBRIS completed
        # for w in house.walls:
        #     # add wall if it's not already there
        #     ancestor = parentFromWalls.get(w)
        #     if ancestor is None:
        #         ancestor = QTreeWidgetItem(self.ui.tree, [scenario.Scenario.dirs[w.direction-1], 'wall', '%.3f'%w.area])
        #         parentFromWalls[w] = ancestor
        #     # add children of wall
        #     for cov in w.coverages:
        #         item = QTreeWidgetItem(ancestor, ['', cov.description, '%.3f'%cov.area, cov.type.name])
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
    my_app = HouseViewer(s.house)
    my_app.show()
    app.exec_()
