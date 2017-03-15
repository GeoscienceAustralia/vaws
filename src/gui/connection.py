from connection_ui import Ui_Dialog
import mixins 
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from vaws import damage, house, scenario

class ConnectionViewer(QDialog, Ui_Dialog, mixins.PersistSizePosMixin):
    def __init__(self, model_conn, model_patches, house_results, parent=None):
        super(ConnectionViewer, self).__init__(parent)
        mixins.PersistSizePosMixin.__init__(self, "ConnectionViewer")

        self.model_conn = model_conn
        self.model_patches = model_patches
        self.house_results = house_results
        
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.name.setText(model_conn.connection_name)
        self.ui.zone_location.setText(model_conn.location_zone.zone_name)
        self.ui.edge.setText('True' if model_conn.edge else 'False')
        self.ui.connection_type.setText(model_conn.ctype.connection_type)
        self.ui.connection_group.setText(model_conn.ctype.group.group_name)
        self.ui.buttonBox.addButton(QDialogButtonBox.Ok)
        
        # update influences table
        dict = damage.inflZonesByConn[model_conn]
        mixins.setupTable(self.ui.influences, dict)
        for irow, z in enumerate(sorted(dict)):
            self.ui.influences.setItem(irow, 0, QTableWidgetItem('%s' % z.zone_name))
            self.ui.influences.setItem(irow, 1, QTableWidgetItem('%.3f' % dict[z]))
        mixins.finiTable(self.ui.influences)
        
        # update patches table - flattened view of: (*PatchConn --> *Influence zone, Influence Patch)
        mixins.setupTable(self.ui.patches, model_patches)
        for irow, patch in enumerate(model_patches):
            conn = house.connByIDMap[patch[0]]
            zone = house.zoneByIDMap[patch[1]]
            p = patch[2]
            self.ui.patches.setItem(irow, 0, QTableWidgetItem(conn.connection_name))
            self.ui.patches.setItem(irow, 1, QTableWidgetItem(zone.zone_name))
            self.ui.patches.setItem(irow, 2, QTableWidgetItem('%.3f' % p))
        mixins.finiTable(self.ui.patches)
        
        # update house_results display
        for (house_num, hr) in enumerate(house_results):
            for cr in hr[4]:
                if cr[7] == model_conn.id:
                    damage_report = cr[5]
                    load = damage_report.get('load', 0)
                    parent = QTreeWidgetItem(self.ui.connections, ['H%d (%s/%.3f)'%(house_num+1, scenario.Scenario.dirs[hr[2]], hr[3]),
                                                           '%.3f'%cr[2] if cr[2] != 0 else '', 
                                                           '%.3f'%cr[3], 
                                                           '%.3f'%cr[4], 
                                                           '%.3f'%load if load != 0 else ''])
                    for infl_dict in damage_report.get('infls', {}):
                        QTreeWidgetItem(parent, ['%.3f(%s) = %.3f(i) * %.3f(area) * %.3f(pz)' % (
                                            infl_dict['load'], 
                                            infl_dict['name'], 
                                            infl_dict['infl'], 
                                            infl_dict['area'], 
                                            infl_dict['pz'])])                        
                    break

        self.ui.connections.resizeColumnToContents(1)
        self.ui.connections.resizeColumnToContents(2)
        self.ui.connections.resizeColumnToContents(3)
        self.ui.connections.resizeColumnToContents(4)
        header_view = self.ui.connections.header()
        header_view.resizeSection(0, 350)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setSizeGripEnabled(True)
        self.setWindowTitle('Connection Viewer (%s)' % (model_conn.connection_name))        

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
    my_app = ConnectionViewer(s.house.connections[400], database.db.qryConnectionPatchesFromDamagedConn(400), [])
    my_app.show()
    app.exec_()
