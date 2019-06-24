
from PyQt5.QtCore import QVariant, QSettings
from PyQt5.QtWidgets import QTableWidget


class PersistSizePosMixin(object):
    def __init__(self, name):
        self.name = name
        
    def initSizePosFromSettings(self):
        settings = QSettings()
        key = self.name + "/Geometry"
        if settings.contains(key):
            self.restoreGeometry(settings.value(key))
            
        key = self.name + "/State"
        if settings.contains(key):
            val = settings.value(key)
            self.restoreState(val)
      
    def storeSizePosToSettings(self):
        settings = QSettings()  
        settings.setValue(self.name + "/Geometry", QVariant(self.saveGeometry()))
        settings.setValue(self.name + "/State", QVariant(self.saveState()))


def setupTable(t, l=None):
    t.setUpdatesEnabled(False)
    t.blockSignals(True)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    if l is not None:
        t.setRowCount(len(l))
    t.setSelectionBehavior(QTableWidget.SelectRows)


def finiTable(t):
    t.resizeColumnsToContents()
    t.setUpdatesEnabled(True)
    t.blockSignals(False)