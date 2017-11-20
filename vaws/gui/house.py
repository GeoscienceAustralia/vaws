from house_ui import Ui_Dialog
import mixins 

from PyQt4.QtCore import *
from PyQt4.QtGui import *


class HouseViewer(QDialog, Ui_Dialog, mixins.PersistSizePosMixin):
    def __init__(self, cfg, parent=None):
        super(HouseViewer, self).__init__(parent)
        mixins.PersistSizePosMixin.__init__(self, "HouseViewer")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle('House Viewer: {}'.format(cfg.house_name))
        self.ui.buttonBox.addButton(QDialogButtonBox.Ok)        
        self.ui.replacementCost.setText('$ {:.3f}'.format(cfg.house['replace_cost']))
        self.ui.height.setText('{:.3f}'.format(cfg.house['height']))
        self.ui.cpeV.setText('{:.3f}'.format(cfg.house['cpe_cov']))
        self.ui.cpeK.setText('{:.3f}'.format(cfg.house['cpe_k']))
        self.ui.cpeStructV.setText('{:.3f}'.format(cfg.house['cpe_str_cov']))
        self.ui.cpeStructK.setText('{:.3f}'.format(cfg.house['cpe_str_k']))
        self.ui.width.setText('{:.3f}'.format(cfg.house['width']))
        self.ui.length.setText('{:.3f}'.format(cfg.house['length']))

        # fill in walls table
        self.ui.tree.clear()
        parentFromWalls = {}
        for w, grouped in cfg.coverages.groupby('wall_name'):
            _area = grouped['area'].sum()
            ancestor = QTreeWidgetItem(
                self.ui.tree, ['{:d}'.format(w), '', '{:.3f}'.format(_area)])
            parentFromWalls[w] = ancestor
            # add children of wall
            for _, _ps in grouped.iterrows():
                _ = QTreeWidgetItem(ancestor, ['', _ps['description'],
                                               '{:.3f}'.format(_ps['area']),
                                               _ps['coverage_type']])
        self.ui.tree.resizeColumnToContents(0)
        self.ui.tree.resizeColumnToContents(1)

    def accept(self):
        QDialog.accept(self)
        
    def reject(self):
        QDialog.reject(self)
