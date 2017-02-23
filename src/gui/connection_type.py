from connection_type_ui import Ui_Dialog
import mixins 
from vaws import stats
from PyQt4.QtCore import *
from PyQt4.QtGui import *


class ConnectionTypeEditor(QDialog, Ui_Dialog, mixins.PersistSizePosMixin):
    def __init__(self, conntype, house, parent=None):
        super(ConnectionTypeEditor, self).__init__(parent)
        mixins.PersistSizePosMixin.__init__(self, "ConnectionTypeEditor")
        
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.conntype = conntype
        self.house = house
        self.ui.buttonBox.addButton(QDialogButtonBox.Ok)
        self.ui.buttonBox.addButton(QDialogButtonBox.Cancel)
        self.ui.connectionType.setText(conntype.connection_type)
        
        self.ui.costingArea.setText('%.3f' % conntype.costing_area)
        self.ui.costingArea.setValidator(QDoubleValidator(-100.0, 100.0, 3, self.ui.costingArea))
        
        self.ui.strengthMean.setText('%.3f' % stats.lognormal_underlying_mean(conntype.strength_mean, conntype.strength_std_dev))
        self.ui.strengthMean.setValidator(QDoubleValidator(-100.0, 100.0, 3, self.ui.strengthMean))
        
        self.ui.strengthSigma.setText('%.3f' % stats.lognormal_underlying_stddev(conntype.strength_mean, conntype.strength_std_dev))
        self.ui.strengthSigma.setValidator(QDoubleValidator(-100.0, 100.0, 3, self.ui.strengthSigma))
        
        self.ui.deadloadMean.setText('%.3f' % stats.lognormal_underlying_mean(conntype.deadload_mean, conntype.deadload_std_dev))
        self.ui.deadloadMean.setValidator(QDoubleValidator(-100.0, 100.0, 3, self.ui.costingArea))
        
        self.ui.deadloadSigma.setText('%.3f' % stats.lognormal_underlying_stddev(conntype.deadload_mean, conntype.deadload_std_dev))
        self.ui.deadloadSigma.setValidator(QDoubleValidator(-100.0, 100.0, 3, self.ui.costingArea))
        
        self.ui.group.setText(conntype.group.group_name)
        
        self.connect(self.ui.plot_strength_button, SIGNAL("clicked(bool)"), self.plot_strength)
        self.connect(self.ui.plot_deadload_button, SIGNAL("clicked(bool)"), self.plot_deadload)
        
        # load up connections grid
        mixins.setupTable(self.ui.connections)
        irow = 0
        for c in house.connections:
            if c.ctype.id == conntype.id:
                self.ui.connections.insertRow(irow)
                self.ui.connections.setItem(irow, 0, QTableWidgetItem(c.ctype.connection_type))
                self.ui.connections.setItem(irow, 1, QTableWidgetItem(c.location_zone.zone_name))
                self.ui.connections.setItem(irow, 2, QTableWidgetItem('False' if c.edge == 0 else 'True'))
                self.ui.connections.setItem(irow, 3, QTableWidgetItem("%0.3f" % c.result_failure_v))
                self.ui.connections.setItem(irow, 4, QTableWidgetItem("%d" % c.result_failure_v_i))        
                irow += 1
        mixins.finiTable(self.ui.connections)
        
    def plot_strength(self):
        mean = float(unicode(self.ui.strengthMean.text()))
        stddev = float(unicode(self.ui.strengthSigma.text()))
        self.plot_cdf(self.ui.plot_strength, mean, stddev)
    
    def plot_deadload(self):
        mean = float(unicode(self.ui.deadloadMean.text()))
        stddev = float(unicode(self.ui.deadloadSigma.text()))
        self.plot_cdf(self.ui.plot_deadload, mean, stddev)
    
    def plot_cdf(self, plot_widget, mean, stddev):
        import numpy.random
        m = stats.lognormal_mean(mean, stddev)
        s = stats.lognormal_stddev(mean, stddev)
        x = []
        n = 10000
        for i in xrange(n):
            rv = numpy.random.lognormal(m, s)
            x.append(rv)
        plot_widget.axes.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
        plot_widget.axes.set_title('Sample Hist (mean: %.3f, stddev: %.3f, n=%d' % (mean, stddev, n))
        plot_widget.axes.figure.canvas.draw()    
    
    def accept(self):
        self.conntype.costing_area = float(unicode(self.ui.costingArea.text()))
        
        self.conntype.set_strength_params(float(unicode(self.ui.strengthMean.text())),
                                          float(unicode(self.ui.strengthSigma.text())))
        
        self.conntype.set_deadload_params(float(unicode(self.ui.deadloadMean.text())),
                                          float(unicode(self.ui.deadloadSigma.text())))
        
        QDialog.accept(self)
        
    def reject(self):
        QDialog.reject(self)
        
        
if __name__ == '__main__':
    import sys
    from vaws import scenario, database
    database.configure()
    s = scenario.Scenario(20, 40.0, 120.0, 60.0, '2')
    s.setHouseName('Group 4 House')
    app = QApplication(sys.argv)
    myapp = ConnectionTypeEditor(s.house.conn_types[1], s.house)
    myapp.show()
    app.exec_()
