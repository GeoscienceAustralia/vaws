
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QTreeWidgetItem, QDialogButtonBox

from vaws.gui.house_ui import Ui_Dialog
from vaws.gui.mixins import PersistSizePosMixin


class HouseViewer(QDialog, Ui_Dialog, PersistSizePosMixin):
    def __init__(self, cfg, parent=None):
        super(HouseViewer, self).__init__(parent)
        PersistSizePosMixin.__init__(self, "HouseViewer")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle('House Viewer: {}'.format(cfg.model_name))
        self.ui.buttonBox.addButton(QDialogButtonBox.Ok)        
        self.ui.replacementCost.setText('$ {:.3f}'.format(cfg.house['replace_cost']))
        self.ui.height.setText('{:.3f}'.format(cfg.house['height']))
        self.ui.cpeV.setText('{:.3f}'.format(cfg.house['cpe_cv']))
        self.ui.cpeK.setText('{:.3f}'.format(cfg.house['cpe_k']))
        self.ui.cpeStructV.setText('{:.3f}'.format(cfg.house['cpe_str_cv']))
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
