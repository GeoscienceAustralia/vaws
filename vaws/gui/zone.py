from zone_ui import Ui_Dialog
import mixins 
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from model import scenario, house, zone


class ZoneBrowser(QDialog, Ui_Dialog, mixins.PersistSizePosMixin):
    def __init__(self, zone_name, plot_key, model_house, house_results, parent=None):
        super(ZoneBrowser, self).__init__(parent)
        mixins.PersistSizePosMixin.__init__(self, "ZoneBrowser")

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setSizeGripEnabled(True)
        self.setWindowTitle('Zone Editor (%s)' % (zone_name))
        self.model_house = model_house
        self.plot_key = plot_key
        self.house_results = house_results
    
        self.ui.buttonBox.addButton(QDialogButtonBox.Ok)
        self.ui.buttonBox.addButton(QDialogButtonBox.Cancel)
        
        self.connect(self.ui.group_filter, SIGNAL("currentIndexChanged(QString)"), self.updateFilter)
        self.connect(self.ui.pdf_plot_button, SIGNAL("clicked(bool)"), self.plot_pdf)
    
        self.ui.zone_name.setText(zone_name)
        self.ui.area.setValidator(QDoubleValidator(0, 100.0, 3, self.ui.area))
        self.ui.cpi_alpha.setValidator(QDoubleValidator(-100.0, 100.0, 3, self.ui.cpi_alpha))
        
        self.ui.house_cpe_k.setText('%.3f' % self.model_house.cpe_k)
        self.ui.house_cpe_v.setText('%.3f' % self.model_house.cpe_V)
        self.ui.house_cpe_struct_v.setText('%.3f' % self.model_house.cpe_struct_V)
    
        self.updateZoneInfo(zone_name)
        
    def plot_pdf(self, checked):
        col = self.ui.table.currentColumn()
        row = self.ui.table.currentRow()
        if col == 0:
            return
        mean = float(self.ui.table.item(row, col).text())
        self.plot_cdf(self.ui.pdf_plot, mean, True if col == 2 else False)
        
    def plot_cdf(self, plot_widget, cpe_mean, use_struct):
        A = zone.calc_A(self.model_house.cpe_k)
        B = zone.calc_B(self.model_house.cpe_k)
        x = []
        n = 10000
        for i in xrange(n):
            rv = zone.sample_gev(cpe_mean, 
                                 A, B, 
                                 self.model_house.cpe_V if use_struct else self.model_house.cpe_struct_V, 
                                 self.model_house.cpe_k)
            x.append(rv)
        plot_widget.axes.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
        plot_widget.axes.set_title('Sample Hist (mean=%.3f, n=%d)' % (cpe_mean, 10000))
        plot_widget.axes.figure.canvas.draw()
        
    def updateZoneInfo(self, zone_name):
        self.model_zone = house.zoneByLocationMap[unicode(zone_name)]
        self.ui.area.setText('%.3f' % self.model_zone.zone_area)
        self.ui.cpi_alpha.setText('%.3f' % self.model_zone.cpi_alpha)
        self.updateConnectionsList(self.ui.group_filter.currentText())
        
        # update table of coefficients
        dirs = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']
        self.ui.table.setUpdatesEnabled(False)
        self.ui.table.blockSignals(True)
        self.ui.table.setRowCount(len(dirs))
        for irow, wind_dir in enumerate(dirs):
            self.ui.table.setItem(irow, 0, QTableWidgetItem('%s' % wind_dir))
            float_val = getattr(self.model_zone, 'coeff_%s' % wind_dir)
            self.ui.table.setItem(irow, 1, QTableWidgetItem( '%.3f' % float_val ))
            float_val = getattr(self.model_zone, 'struct_coeff_%s' % wind_dir)
            self.ui.table.setItem(irow, 2, QTableWidgetItem( '%.3f' % float_val ))
            float_val = getattr(self.model_zone, 'eaves_coeff_%s' % wind_dir)
            self.ui.table.setItem(irow, 3, QTableWidgetItem( '%.3f' % float_val ))
        self.ui.table.resizeColumnsToContents()
        self.ui.table.setUpdatesEnabled(True)
        self.ui.table.blockSignals(False)
        self.ui.table.selectColumn(1)
        
    def updateFilter(self, filter):
        expanded_items = []
        for tli in range(self.ui.connections.topLevelItemCount()):
            item = self.ui.connections.topLevelItem(tli)
            if item.isExpanded():
                expanded_items.append(tli)
                
        str_filter = unicode(filter)
        self.updateConnectionsList(str_filter)
        
        for tli in expanded_items:
            item = self.ui.connections.topLevelItem(tli)
            item.setExpanded(True)
        
    def updateConnectionsList(self, filter):
        filter = unicode(filter)
        mixins.setupTable(self.ui.zones, self.house_results)
        self.ui.connections.clear()
    
        irow = 0
        for (house_num, hr) in enumerate(self.house_results):
            # zone
            zone_results_dict = hr[0]        
            zr = zone_results_dict[self.model_zone.zone_name]
            self.ui.zones.setItem(irow, 0, QTableWidgetItem('H%d (%s/%.3f)' % (house_num+1, scenario.Scenario.dirs[hr[2]], hr[3])))
            self.ui.zones.setItem(irow, 1, QTableWidgetItem("%0.3f" % zr[1]))
            self.ui.zones.setItem(irow, 2, QTableWidgetItem("%0.3f" % zr[2]))
            self.ui.zones.setItem(irow, 3, QTableWidgetItem("%0.3f" % zr[3]))
            irow += 1
            
            # connections
            parent = QTreeWidgetItem(self.ui.connections, ['H%d (%s/%.3f)'%(house_num+1, scenario.Scenario.dirs[hr[2]], hr[3])])
            for cr in hr[4]:
                damage_report = cr[5]
                if cr[1] != self.model_zone.zone_name:
                    continue
                if filter != "All" and filter.lower() != cr[6]:
                    continue
                
                load = damage_report.get('load', 0)
                conn_parent = QTreeWidgetItem(parent, ['%s'%cr[0], 
                                                       '%.3f'%cr[2] if cr[2] != 0 else '', 
                                                       '%.3f'%cr[3], 
                                                       '%.3f'%cr[4], 
                                                       '%.3f'%load if load != 0 else ''])
                for infl_dict in damage_report.get('infls', {}):
                    QTreeWidgetItem(conn_parent, ['%.3f(%s) = %.3f(i) * %.3f(area) * %.3f(pz)' % (infl_dict['load'], 
                                        infl_dict['name'], infl_dict['infl'], infl_dict['area'], infl_dict['pz'])])
                    
        mixins.finiTable(self.ui.zones)
        self.ui.connections.resizeColumnToContents(1)
        self.ui.connections.resizeColumnToContents(2)
        self.ui.connections.resizeColumnToContents(3)
        self.ui.connections.resizeColumnToContents(4)
        header_view = self.ui.connections.header()
        header_view.resizeSection(0, 350)
        
    def accept(self):
        self.model_zone.zone_area = float(unicode(self.ui.area.text()))
        self.model_zone.cpi_alpha = float(unicode(self.ui.cpi_alpha.text()))
       
        for i, wind_dir in enumerate(['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']):
            coeff_name = 'coeff_%s' % wind_dir
            float_val = float(self.ui.table.item(i, 1).text())
            setattr(self.model_zone, coeff_name, float_val)
            
            coeff_name = 'struct_coeff_%s' % wind_dir
            float_val = float(self.ui.table.item(i, 2).text())
            setattr(self.model_zone, coeff_name, float_val)
            
            coeff_name = 'eaves_coeff_%s' % wind_dir
            float_val = float(self.ui.table.item(i, 3).text())
            setattr(self.model_zone, coeff_name, float_val)
        
        QDialog.accept(self)
        
    def reject(self):
        QDialog.reject(self)
        
        
if __name__ == '__main__':
    import sys
    from model import database
    database.configure()
    s = scenario.Scenario(20, 40.0, 120.0, 60.0, '2')
    s.setHouseName('Group 4 House')
    app = QApplication(sys.argv)
    my_app = ZoneBrowser(s.house.zones[0].zone_name, 'sheeting', s.house, [])
    my_app.show()
    app.exec_()
