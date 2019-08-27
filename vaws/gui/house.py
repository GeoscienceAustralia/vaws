
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
        self.setWindowTitle(f'House Viewer: {cfg.model_name}')
        self.ui.buttonBox.addButton(QDialogButtonBox.Ok)
        self.ui.replacementCost.setText(f"$ {cfg.house['replace_cost']:.3f}")
        self.ui.height.setText(f"{cfg.house['height']:.3f}")
        self.ui.cpeV.setText(f"{cfg.house['cpe_cv']:.3f}")
        self.ui.cpeK.setText(f"{cfg.house['cpe_k']:.3f}")
        self.ui.cpeStructV.setText(f"{cfg.house['cpe_str_cv']:.3f}")
        self.ui.cpeStructK.setText(f"{cfg.house['cpe_str_k']:.3f}")
        self.ui.width.setText(f"{cfg.house['width']:.3f}")
        self.ui.length.setText(f"{cfg.house['length']:.3f}")

        # fill in walls table
        self.ui.tree.clear()
        parentFromWalls = {}
        for w, grouped in cfg.coverages.groupby('wall_name'):
            _area = grouped['area'].sum()
            ancestor = QTreeWidgetItem(
                self.ui.tree, [f'{w:d}', '', f'{_area:.3f}'])
            parentFromWalls[w] = ancestor
            # add children of wall
            for _, _ps in grouped.iterrows():
                _ = QTreeWidgetItem(ancestor, ['', _ps['description'],
                                               f"{_ps['area']:.3f}",
                                               _ps['coverage_type'],
                                               _ps['repair_type']],
                                               )
        self.ui.tree.resizeColumnToContents(0)
        self.ui.tree.resizeColumnToContents(1)

    def accept(self):
        QDialog.accept(self)

    def reject(self):
        QDialog.reject(self)
