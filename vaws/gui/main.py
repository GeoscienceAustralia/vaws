#!/usr/bin/env python
# adjust python path so we may import things from peer packages
import sys
import shutil
import time
import os.path
import logging
import warnings

import matplotlib.pyplot as plt
import h5py
import numpy as np
from collections import OrderedDict
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
from PyQt4.QtCore import SIGNAL, QTimer, Qt, QSettings, QVariant, QString, QFile
from PyQt4.QtGui import QProgressBar, QLabel, QMainWindow, QApplication, QTableWidget, QPixmap,\
                        QTableWidgetItem, QDialog, QCheckBox, QFileDialog, QIntValidator,\
                        QDoubleValidator, QMessageBox, QTreeWidgetItem, QInputDialog, QSplashScreen

from vaws.model.constants import WIND_DIR, DEBRIS_TYPES_KEYS, VUL_DIC, BLDG_SPACING
from vaws.model.house import House
from vaws.model.debris import generate_debris_items
from vaws.gui.house import HouseViewer
from vaws.model.curve import vulnerability_lognorm, vulnerability_weibull, \
    vulnerability_weibull_pdf
from vaws.model.stats import compute_arithmetic_mean_stddev, sample_lognorm_given_mean_stddev

from vaws.gui.main_ui import Ui_main
from vaws.model.main import process_commandline, set_logger, \
    simulate_wind_damage_to_houses
from vaws.model.config import Config, INPUT_DIR, OUTPUT_DIR
from vaws.model.version import VERSION_DESC
from vaws.model.damage_costing import compute_water_ingress_given_damage

from vaws.gui.output import plot_wind_event_damage, plot_wind_event_mean, \
                        plot_wind_event_show, plot_fitted_curve, \
                        plot_fragility_show, plot_damage_show, plot_influence, \
                        plot_influence_patch, plot_load_show, plot_pressure_show

from mixins import PersistSizePosMixin, setupTable, finiTable

my_app = None

SOURCE_DIR = os.path.dirname(__file__)
VAWS_DIR = os.sep.join(SOURCE_DIR.split(os.sep)[:-2])
SCENARIOS_DIR = os.sep.join(SOURCE_DIR.split(os.sep)[:-1])
SCENARIOS_DIR = os.path.join(SCENARIOS_DIR, 'scenarios')
CONFIG_TEMPL = "Scenarios (*.cfg)"
DEFAULT_SCENARIO = os.path.join(SCENARIOS_DIR, 'default', 'default.cfg')

warnings.filterwarnings("ignore")


def progress_callback(percent_done):
    my_app.statusProgressBar.setValue(percent_done)
    QApplication.processEvents()
    if my_app.stopTriggered:
        my_app.stopTriggered = False
        return False
    else:
        return True


class MyForm(QMainWindow, Ui_main, PersistSizePosMixin):
    def __init__(self, parent=None, init_scenario=None):
        super(MyForm, self).__init__(parent)
        PersistSizePosMixin.__init__(self, "MainWindow")

        self.logger = logging.getLogger(__name__)

        self.ui = Ui_main()
        self.ui.setupUi(self)

        windowTitle = VERSION_DESC
        self.setWindowTitle(unicode(windowTitle))

        self.cfg = init_scenario
        self.results_dict = None

        #self.ui.numHouses.setValidator(QIntValidator(1, 10000, self.ui.numHouses))
        #self.ui.numHouses.setText('{:d}'.format(self.cfg.no_models))
        #self.ui.houseName.setText(self.cfg.house['name'])

        # debris
        #self.ui.debrisRegion.setText(self.cfg.region_name)
        #self.ui.flighttimeMean.setValidator(
        #    QDoubleValidator(0.0, 100.0, 3, self.ui.flighttimeMean))
        #self.ui.flighttimeStddev.setValidator(QDoubleValidator(
        #    0.0, 100.0, 3, self.ui.flighttimeStddev))

        #self.ui.heatmap_house.setRange(0, self.cfg.no_models)
        #self.ui.heatmap_house.setValue(0)
        #self.ui.heatmap_houseLabel.setText('{:d}'.format(0))

        self.statusProgressBar = QProgressBar()
        self.statusProgressBar.setMinimum(0)
        self.statusProgressBar.setMaximum(100)
        self.statusBar().addPermanentWidget(self.statusProgressBar)
        self.statusProgressBar.hide()        
        self.statusBarScenarioLabel = QLabel()
        self.statusBarScenarioLabel.setText('Scenario: None')
        self.statusBar().addPermanentWidget(self.statusBarScenarioLabel)
        
        self.dirty_scenario = False         # means scenario has changed
        # self.dirty_conntypes = False        # means connection_types have been modified
        self.has_run = False
        self.initSizePosFromSettings()

        # top panel
        self.connect(self.ui.actionOpen_Scenario, SIGNAL("triggered()"),
                     self.open_scenario)
        self.connect(self.ui.actionRun, SIGNAL("triggered()"),
                     self.runScenario)
        self.connect(self.ui.actionStop, SIGNAL("triggered()"),
                     self.stopScenario)
        self.connect(self.ui.actionSave_Scenario, SIGNAL("triggered()"),
                     self.save_scenario)
        self.connect(self.ui.actionSave_Scenario_As, SIGNAL("triggered()"),
                     self.save_as_scenario)
        self.connect(self.ui.actionHouse_Info, SIGNAL("triggered()"),
                     self.showHouseInfoDlg)
        # TODO: actionNew missing

        # test panel
        self.connect(self.ui.testDebrisButton, SIGNAL("clicked()"),
                     self.testDebrisSettings)
        self.connect(self.ui.testConstructionButton, SIGNAL("clicked()"),
                     self.testConstructionLevels)
        self.connect(self.ui.testWaterIngressButton, SIGNAL("clicked()"),
                     self.testWaterIngress)

        # Scenario panel
        self.connect(self.ui.windMin, SIGNAL("valueChanged(int)"),
                     lambda x: self.onSliderChanged(self.ui.windMinLabel, x))
        self.connect(self.ui.windMax, SIGNAL("valueChanged(int)"),
                     lambda x: self.onSliderChanged(self.ui.windMaxLabel, x))

        # debris panel
        self.connect(self.ui.debrisRadius, SIGNAL("valueChanged(int)"),
                     lambda x: self.onSliderChanged(self.ui.debrisRadiusLabel, x))
        self.connect(self.ui.debrisAngle, SIGNAL("valueChanged(int)"),
                     lambda x: self.onSliderChanged(self.ui.debrisAngleLabel, x))
        self.connect(self.ui.sourceItems, SIGNAL("valueChanged(int)"),
                     lambda x: self.onSliderChanged(self.ui.sourceItemsLabel, x))

        # options
        self.connect(self.ui.redV, SIGNAL("valueChanged(int)"),
                     lambda x: self.onSliderChanged(self.ui.redVLabel, x))
        self.connect(self.ui.blueV, SIGNAL("valueChanged(int)"),
                     lambda x: self.onSliderChanged(self.ui.blueVLabel, x))
        self.connect(self.ui.vStep, SIGNAL("valueChanged(int)"),
                     lambda x: self.onSliderChanged(self.ui.vStepLabel, x))
        self.connect(self.ui.applyDisplayChangesButton, SIGNAL("clicked()"),
                     self.updateDisplaySettings)

        self.statusBar().showMessage('Loading')
        self.stopTriggered = False
        # self.selected_zone = None
        # self.selected_conn = None
        # self.selected_plotKey = None

        # self.ui.sourceItems.setValue(-1)
        # scenario
        self.init_terrain_category()
        self.ui.windDirection.clear()
        self.ui.windDirection.addItems(WIND_DIR)

        # debris
        self.init_debris_region()
        self.ui.buildingSpacing.clear()
        self.ui.buildingSpacing.addItems([str(x) for x in BLDG_SPACING])

        # RHS window
        self.init_pressure()

        self.init_influence_and_patch()

        self.init_heatmap_group()

        self.init_load_plot()

        # self.connect(self.ui.slider_influence, SIGNAL("valueChanged(int)"),
        #              lambda x: self.onSliderChanged(self.ui.slider_influenceLabel, x))

        # self.connect(self.ui.slider_patch, SIGNAL("valueChanged(int)"),
        #              lambda x: self.onSliderChanged(self.ui.slider_patchLabel, x))

        # self.connect(self.ui.cpi_house, SIGNAL("valueChanged(int)"),
        #              lambda x: self.onSliderChanged(self.ui.cpi_houseLabel, x))
        self.ui.spinBox_cpi.valueChanged.connect(self.cpi_plot_change)

        self.update_ui_from_config()
        # QTimer.singleShot(0, self.set_scenario)

        self.statusBar().showMessage('Ready')

    def init_terrain_category(self):

        self.disconnect(self.ui.terrainCategory,
                        SIGNAL("currentIndexChanged(QString)"),
                        self.updateTerrainCategoryTable)
        self.ui.terrainCategory.clear()
        self.ui.terrainCategory.addItems(os.listdir(self.cfg.path_wind_profiles))
        self.connect(self.ui.terrainCategory,
                     SIGNAL("currentIndexChanged(QString)"),
                     self.updateTerrainCategoryTable)

    def init_debris_region(self):

        self.disconnect(self.ui.debrisRegion,
                        SIGNAL("currentIndexChanged(QString)"),
                        self.updateDebrisRegionsTable)
        self.ui.debrisRegion.clear()
        self.ui.debrisRegion.addItems(self.cfg.debris_regions.keys())
        self.connect(self.ui.debrisRegion,
                     SIGNAL("currentIndexChanged(QString)"),
                     self.updateDebrisRegionsTable)

    def init_heatmap_group(self):

        self.disconnect(self.ui.comboBox_heatmap,
                        SIGNAL("currentIndexChanged(QString)"),
                        self.heatmap_house_change)
        self.ui.comboBox_heatmap.clear()
        self.ui.comboBox_heatmap.addItems(self.cfg.list_groups)
        self.connect(self.ui.comboBox_heatmap,
                     SIGNAL("currentIndexChanged(QString)"),
                     self.heatmap_house_change)

        self.ui.spinBox_heatmap.valueChanged.connect(self.heatmap_house_change)

    def init_pressure(self):

        pressure_keys = ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'cpi_alpha', 'edge']
        self.ui.mplpressure.figure.clear()
        self.ui.mplpressure.axes.cla()
        self.ui.mplpressure.axes.figure.canvas.draw()

        # init combobox
        self.disconnect(self.ui.comboBox_pressure,
                        SIGNAL("currentIndexChanged(QString)"),
                        self.updatePressurePlot)
        self.ui.comboBox_pressure.clear()
        self.ui.comboBox_pressure.addItems(pressure_keys)
        self.connect(self.ui.comboBox_pressure,
                     SIGNAL("currentIndexChanged(QString)"),
                     self.updatePressurePlot)

        if str(self.ui.comboBox_pressure.currentText()) == pressure_keys[0]:
            self.updatePressurePlot()
        else:
            self.ui.comboBox_pressure.setValue(pressure_keys[0])

        # init combobox2
        self.disconnect(self.ui.comboBox2_pressure,
                        SIGNAL("currentIndexChanged(QString)"),
                        self.updatePressurePlot)
        self.ui.comboBox2_pressure.clear()
        self.ui.comboBox2_pressure.addItems(WIND_DIR[:-1])
        self.connect(self.ui.comboBox2_pressure,
                     SIGNAL("currentIndexChanged(QString)"),
                     self.updatePressurePlot)

        if str(self.ui.comboBox2_pressure.currentText()) == WIND_DIR[0]:
            self.updatePressurePlot()
        else:
            self.ui.comboBox2_pressure.setValue(WIND_DIR[0])

    def init_influence_and_patch(self):
        # init_influence
        self.ui.mplinfluecnes.figure.clear()
        self.ui.mplinfluecnes.axes.cla()
        self.ui.mplinfluecnes.axes.figure.canvas.draw()

        self.ui.spinBox_influence.valueChanged.connect(self.updateInfluence)

        self.ui.spinBox_influence.setRange(self.cfg.list_connections[0],
                                           self.cfg.list_connections[-1])
        if self.ui.spinBox_influence.value() == self.cfg.list_connections[0]:
            self.updateInfluence()
        else:
            self.ui.spinBox_influence.setValue(self.cfg.list_connections[0])

        # init_patch
        self.ui.mplpatches.figure.clear()
        self.ui.mplpatches.axes.cla()
        self.ui.mplpatches.axes.figure.canvas.draw()

        _list = sorted(self.cfg.influence_patches.keys())

        if _list:
            self.ui.spinBox_patch.valueChanged.connect(self.updateComboBox_patch)

            self.ui.spinBox_patch.setRange(_list[0], _list[-1])
            if self.ui.spinBox_patch.value() == _list[0]:
                self.updateComboBox_patch()
            else:
                self.ui.spinBox_patch.setValue(_list[0])
        else:
            self.logger.info('patch is not defined')

        # self.disconnect(self.ui.slider_patch, SIGNAL("valueChanged(int)"),
        #                 self.updateComboBox)

    # def onZoneSelected(self, z, plotKey):
    #     self.selected_zone = z
    #     self.selected_plotKey = plotKey

    # def onSelectConnection(self, connection_name):
    #     for irow in range(len(self.cfg.house.connections)):
    #         if unicode(self.ui.connections.item(irow, 0).text()) == connection_name:
    #             self.ui.connections.setCurrentCell(irow, 0)
    #             break
        
    # def onSelectZone(self, zoneLoc):
    #     for irow in range(len(self.cfg.house.zones)):
    #         if unicode(self.ui.zones.item(irow, 0).text()) == zoneLoc:
    #             self.ui.zones.setCurrentCell(irow, 0)
    #             break

    def init_load_plot(self):

        self.disconnect(self.ui.comboBox_load,
                        SIGNAL("currentIndexChanged(QString)"),
                        self.load_connection_change)
        self.ui.comboBox_load.clear()
        self.ui.comboBox_load.addItems(self.cfg.list_groups)
        self.connect(self.ui.comboBox_load,
                     SIGNAL("currentIndexChanged(QString)"),
                     self.load_connection_change)

        self.ui.spinBox_load.valueChanged.connect(self.load_connection_change)

        self.ui.doubleSpinBox_load.valueChanged.connect(self.load_connection_change)

    def onSliderChanged(self, label, x):
        label.setText('{:d}'.format(x))

    def updateVulnCurve(self):

        _array = self.results_dict['house']['di']

        self.ui.mplvuln.axes.hold(True)

        plot_wind_event_show(self.ui.mplvuln, self.cfg.no_models,
                             self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])

        mean_cols = _array.T.mean(axis=0)
        plot_wind_event_mean(self.ui.mplvuln, self.cfg.wind_speeds, mean_cols)

        # plot the individuals
        for col in _array.T:
            plot_wind_event_damage(self.ui.mplvuln, self.cfg.wind_speeds, col)

        with h5py.File(self.cfg.file_results, "r") as f:
            try:
                param1 = f['vulnerability']['weibull']['param1'].value
                param2 = f['vulnerability']['weibull']['param2'].value
            except KeyError:
                pass
            else:
                try:
                    y = vulnerability_weibull(self.cfg.wind_speeds, param1, param2)
                except KeyError as msg:
                    self.logger.warning(msg)
                else:
                    plot_fitted_curve(self.ui.mplvuln,
                                      self.cfg.wind_speeds,
                                      y,
                                      col='r',
                                      label="Weibull")

                    self.ui.wb_coeff_1.setText('{:.3f}'.format(param1))
                    self.ui.wb_coeff_2.setText('{:.3f}'.format(param2))

            try:
                param1 = f['vulnerability']['lognorm']['param1'].value
                param2 = f['vulnerability']['lognorm']['param2'].value
            except KeyError:
                pass
            else:
                try:
                    y = vulnerability_lognorm(self.cfg.wind_speeds, param1, param2)
                except KeyError as msg:
                    self.logger.warning(msg)
                else:
                    plot_fitted_curve(self.ui.mplvuln,
                                      self.cfg.wind_speeds,
                                      y,
                                      col='b',
                                      label="Lognormal")

                    self.ui.ln_coeff_1.setText('{:.3f}'.format(param1))
                    self.ui.ln_coeff_2.setText('{:.3f}'.format(param2))

            self.ui.mplvuln.axes.legend(loc=2,
                                        fancybox=True,
                                        shadow=True,
                                        fontsize='small')

            self.ui.mplvuln.axes.figure.canvas.draw()

    def updateFragCurve(self):

        _array = self.results_dict['house']['di']
        plot_fragility_show(self.ui.mplfrag, self.cfg.no_models,
                            self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])

        self.ui.mplfrag.axes.hold(True)
        self.ui.mplfrag.axes.plot(self.cfg.wind_speeds, _array, 'k+', 0.3)

        with h5py.File(self.cfg.file_results, "r") as f:

            for ds, value in self.cfg.fragility.iterrows():
                try:
                    param1 = f['fragility'][ds]['param1'].value
                    param2 = f['fragility'][ds]['param2'].value
                except KeyError as msg:
                    self.logger.warning(msg)
                else:
                    y = vulnerability_lognorm(self.cfg.wind_speeds, param1, param2)

                    plot_fitted_curve(self.ui.mplfrag, self.cfg.wind_speeds, y,
                                      col=value['color'], label=ds)

        self.ui.mplfrag.axes.legend(loc=2,
                                    fancybox=True,
                                    shadow=True,
                                    fontsize='small')

        self.ui.mplfrag.axes.figure.canvas.draw()

    def updateDisplaySettings(self):
        if self.has_run:
            self.heatmap_house_change()

    def showHouseInfoDlg(self):
        dlg = HouseViewer(self.cfg)
        dlg.exec_()

    def updateDebrisRegionsTable(self):

        self.cfg.set_region_name(str(self.ui.debrisRegion.currentText()))

        # load up debris regions
        _debris_region = self.cfg.debris_regions[self.cfg.region_name]
        setupTable(self.ui.debrisRegions, _debris_region)

        no_values = len(self.cfg.debris_types[self.cfg.debris_types.keys()[0]])

        for i, key in enumerate(DEBRIS_TYPES_KEYS):

            _list = [(k, v) for k, v in _debris_region.items() if key in k]

            _list.sort()

            for j, (k, v) in enumerate(_list):

                self.ui.debrisRegions.setItem(no_values*i + j, 0,
                                              QTableWidgetItem(k))
                self.ui.debrisRegions.setItem(no_values*i + j, 1,
                                              QTableWidgetItem('{:.3f}'.format(v)))

        finiTable(self.ui.debrisRegions)

    def updateTerrainCategoryTable(self):

        self.cfg.file_wind_profiles = str(self.ui.terrainCategory.currentText())
        self.cfg.set_wind_profiles()

        self.ui.boundaryProfile.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ui.boundaryProfile.setRowCount(len(self.cfg.profile_heights))
        self.ui.boundaryProfile.setSelectionBehavior(QTableWidget.SelectRows)
        self.ui.boundaryProfile.clearContents()

        for irow, head_col in enumerate(self.cfg.profile_heights):
            self.ui.boundaryProfile.setItem(
                irow, 0, QTableWidgetItem('{:.3f}'.format(head_col)))

        for key, _list in self.cfg.wind_profiles.items():
            for irow, value in enumerate(_list):
                self.ui.boundaryProfile.setItem(
                    irow, key, QTableWidgetItem('{:.3f}'.format(value)))

        self.ui.boundaryProfile.resizeColumnsToContents()

    def updateConnectionTypeTable(self):
        # load up connection types grid
        setupTable(self.ui.connectionsTypes, self.cfg.types)
        for irow, (index, ctype) in enumerate(self.cfg.types.items()):
            self.ui.connectionsTypes.setItem(
                irow, 0, QTableWidgetItem(index))
            self.ui.connectionsTypes.setItem(
                irow, 1, QTableWidgetItem('{:.3f}'.format(ctype['strength_mean'])))
            self.ui.connectionsTypes.setItem(
                irow, 2, QTableWidgetItem('{:.3f}'.format(ctype['strength_std'])))
            self.ui.connectionsTypes.setItem(
                irow, 3, QTableWidgetItem('{:.3f}'.format(ctype['dead_load_mean'])))
            self.ui.connectionsTypes.setItem(
                irow, 4, QTableWidgetItem('{:.3f}'.format(ctype['dead_load_std'])))
            self.ui.connectionsTypes.setItem(
                irow, 5, QTableWidgetItem(ctype['group_name']))
            self.ui.connectionsTypes.setItem(
                irow, 6, QTableWidgetItem('{:.3f}'.format(ctype['costing_area'])))
        finiTable(self.ui.connectionsTypes)
        
    def updateZonesTable(self):
        setupTable(self.ui.zones, self.cfg.zones)

        for irow, (index, z) in enumerate(self.cfg.zones.items()):
            self.ui.zones.setItem(
                irow, 0, QTableWidgetItem(index))
            self.ui.zones.setItem(
                irow, 1, QTableWidgetItem('{:.3f}'.format(z['area'])))
            # self.ui.zones.setItem(
            #     irow, 2, QTableWidgetItem('{:.3f}'.format(z['cpi_alpha'])))
            # for _dir in range(8):
            #     self.ui.zones.setItem(
            #         irow, 3 + _dir, QTableWidgetItem('{:.3f}'.format(z['cpe_mean'][_dir])))
            # for _dir in range(8):
            #     self.ui.zones.setItem(
            #         irow, 11 + _dir, QTableWidgetItem('{:.3f}'.format(z['cpe_str_mean'][_dir])))
            # for _dir in range(8):
            #     self.ui.zones.setItem(
            #         irow, 19 + _dir, QTableWidgetItem('{:.3f}'.format(z['cpe_eave_mean'][_dir])))
        finiTable(self.ui.zones)
    
    def updateConnectionGroupTable(self):
        setupTable(self.ui.connGroups, self.cfg.groups)
        for irow, (index, ctg) in enumerate(self.cfg.groups.items()):
            self.ui.connGroups.setItem(irow, 0, QTableWidgetItem(index))
            self.ui.connGroups.setItem(irow, 1, QTableWidgetItem('{:d}'.format(ctg['dist_order'])))
            self.ui.connGroups.setItem(irow, 2, QTableWidgetItem(ctg['dist_dir']))
            self.ui.connGroups.setItem(irow, 3, QTableWidgetItem(index))

            #checked = self.cfg.flags.get('conn_type_group_{}'.format(index), False)
            #cellWidget = QCheckBox()
            #cellWidget.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            #self.ui.connGroups.setCellWidget(irow, 4, cellWidget)
        finiTable(self.ui.connGroups)

    def update_house_panel(self):

        self.updateConnectionTable()
        self.updateConnectionTypeTable()
        self.updateConnectionGroupTable()
        self.updateZonesTable()
        self.updateDamageTable()

    def updateDamageTable(self):
        # load up damage scenarios grid
        setupTable(self.ui.damageScenarios, self.cfg.costings)
        for irow, (name, _inst) in enumerate(self.cfg.costings.items()):
            self.ui.damageScenarios.setItem(irow, 0, QTableWidgetItem(name))
            self.ui.damageScenarios.setItem(irow, 1, QTableWidgetItem(
                '{:.3f}'.format(_inst.surface_area)))
            self.ui.damageScenarios.setItem(irow, 2, QTableWidgetItem(
                '{:.3f}'.format(_inst.envelope_repair_rate)))
            self.ui.damageScenarios.setItem(irow, 3, QTableWidgetItem(
                '{:.3f}'.format(_inst.internal_repair_rate)))
        finiTable(self.ui.damageScenarios)

    def updateConnectionTable(self):
        # load up connections grid
        setupTable(self.ui.connections, self.cfg.connections)
        for irow, (index, c) in enumerate(self.cfg.connections.iterrows()):
            self.ui.connections.setItem(irow, 0,
                                        QTableWidgetItem(c['group_name']))
            self.ui.connections.setItem(irow, 1,
                                        QTableWidgetItem(c['type_name']))
            self.ui.connections.setItem(irow, 2,
                                        QTableWidgetItem(c['zone_loc']))
            # edge = 'False'
            # if c['edge'] != 0:
            #     edge = 'True'
            # self.ui.connections.setItem(irow, 3, QTableWidgetItem(edge))
        finiTable(self.ui.connections)
        self.ui.connections.horizontalHeader().resizeSection(2, 70)
        self.ui.connections.horizontalHeader().resizeSection(1, 110)

    # def updateModelFromUI(self):
    #     if self.dirty_conntypes:
    #         irow = 0
    #         for ctype in self.cfg.house.conn_types:
    #             sm = float(unicode(self.ui.connectionsTypes.item(irow, 1).text()))
    #             ss = float(unicode(self.ui.connectionsTypes.item(irow, 2).text()))
    #             dm = float(unicode(self.ui.connectionsTypes.item(irow, 3).text()))
    #             ds = float(unicode(self.ui.connectionsTypes.item(irow, 4).text()))
    #             ctype.set_strength_params(sm, ss)
    #             ctype.set_deadload_params(dm, ds)
    #             irow += 1
    #         self.dirty_conntypes = False
    #
    #     self.cfg.updateModel()
    
    def stopScenario(self):
        self.stopTriggered = True
        
    def runScenario(self):
        self.statusBar().showMessage('Running Scenario')
        self.statusProgressBar.show()
        self.update_config_from_ui()
        self.cfg.process_config()

        self.ui.spinBox_heatmap.setRange(0, self.cfg.no_models)
        #self.ui.heatmap_houseLabel.setText('{:d}'.format(0))

        self.ui.spinBox_load.setRange(1, self.cfg.no_models)
        self.ui.doubleSpinBox_load.setRange(self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])

        self.ui.mplsheeting.axes.cla()
        self.ui.mplsheeting.axes.figure.canvas.draw()

        # self.ui.mplbatten.axes.cla()
        # self.ui.mplbatten.axes.figure.canvas.draw()
        #
        # self.ui.mplrafter.axes.cla()
        # self.ui.mplrafter.axes.figure.canvas.draw()

        self.ui.connection_type_plot.axes.cla()
        self.ui.connection_type_plot.axes.figure.canvas.draw()

        self.ui.breaches_plot.axes.cla()
        self.ui.breaches_plot.axes.figure.canvas.draw()

        self.ui.spinBox_cpi.setRange(0, self.cfg.no_models)
        # self.ui.cpi_house.setValue(0)
        # self.ui.cpi_houseLabel.setText('{:d}'.format(0))

        self.ui.cpi_plot.axes.cla()
        self.ui.cpi_plot.axes.figure.canvas.draw()

        self.ui.wateringress_plot.axes.cla()
        self.ui.wateringress_plot.axes.figure.canvas.draw()

        self.ui.mplvuln.axes.cla()
        self.ui.mplvuln.axes.figure.canvas.draw()

        self.ui.mplfrag.axes.cla()
        self.ui.mplfrag.axes.figure.canvas.draw()

        # run simulation with progress bar
        self.ui.actionStop.setEnabled(True)
        self.ui.actionRun.setEnabled(False)
        self.ui.actionOpen_Scenario.setEnabled(False)
        self.ui.actionSave_Scenario.setEnabled(False)
        self.ui.actionSave_Scenario_As.setEnabled(False)

        # attempt to run the simulator, being careful with exceptions...
        try:
            run_time, self.results_dict = simulate_wind_damage_to_houses(self.cfg,
                                                              call_back=progress_callback)

            if run_time is not None:
                self.statusBar().showMessage(
                    unicode('Simulation complete in {:0.3f}'.format(run_time)))

                # self.results_dict = bucket
                self.updateVulnCurve()
                self.updateFragCurve()
                self.updateHouseResultsTable()
                self.updateConnectionTable_with_results()
                self.updateConnectionTypePlots()
                self.updateHeatmap()
                self.updateWaterIngressPlot()
                self.updateBreachPlot()
                self.updateCpiPlot()
                self.has_run = True

        except IOError:
            msg = unicode('A report file is still open by another program, '
                          'unable to run simulation.')
            QMessageBox.warning(self, 'VAWS Program Warning', msg)
            self.statusBar().showMessage(unicode(''))
        except Exception as err:
            self.statusBar().showMessage(
                unicode('Fatal Error Occurred: {}'.format(err)))
            raise
        finally:
            self.statusProgressBar.hide()
            self.ui.actionStop.setEnabled(False)
            self.ui.actionRun.setEnabled(True)
            self.ui.actionOpen_Scenario.setEnabled(True)
            self.ui.actionSave_Scenario.setEnabled(True)
            self.ui.actionSave_Scenario_As.setEnabled(True)
            self.statusBar().showMessage('Ready')

    def heatmap_house_change(self):

        if self.has_run:
            # idx = self.ui.comboBox_heatmap.findText(self.cfg.list_groups[0])
            # if self.ui.comboBox_heatmap.currentIndex() == idx:
            self.updateHeatmap()
            # else:
            #     self.ui.comboBox_heatmap.setCurrentIndex(idx)

    def load_connection_change(self):

        if self.has_run:
            # idx = self.ui.comboBox_heatmap.findText(self.cfg.list_groups[0])
            # if self.ui.comboBox_heatmap.currentIndex() == idx:
            self.updateLoadPlot()
            # else:
            #     self.ui.comboBox_heatmap.setCurrentIndex(idx)

    def updateHeatmap(self):

        # group_name = self.cfg.list_groups[
        #     self.ui.comboBox_heatmap.currentIndex()]

        group_name = str(self.ui.comboBox_heatmap.currentText())
        house_number = self.ui.spinBox_heatmap.value()

        grouped = self.cfg.connections.loc[
            self.cfg.connections.group_name == group_name]

        capacity = np.array([self.results_dict['connection']['capacity'][i]
                             for i in grouped.index])[:, 0, :]

        capacity[capacity == -1] = np.nan
        if house_number == 0:
            if len(capacity) > 0:
                mean_connection_capacity = np.nanmean(capacity, axis=1)
            else:
                mean_connection_capacity = np.zeros(1)
        else:
            mean_connection_capacity = capacity[:, house_number - 1]

        mean_connection_capacity = np.nan_to_num(mean_connection_capacity)

        red_v = self.ui.redV.value()
        blue_v = self.ui.blueV.value()
        vstep = self.ui.vStep.value()

        plot_damage_show(self.ui.mplsheeting, grouped,
                         mean_connection_capacity,
                         self.cfg.house['length'],
                         self.cfg.house['width'],
                         red_v, blue_v, vstep,
                         house_number)

    def updatePressurePlot(self):

        pressure_key = str(self.ui.comboBox_pressure.currentText())
        wind_dir_idx = int(self.ui.comboBox2_pressure.currentIndex())

        coords = [(key, value['coords'], value['centroid'])
                  for key, value in self.cfg.zones.items()]

        if pressure_key == 'cpi_alpha':
            pressure_value = np.array([value[pressure_key]
                                       for _, value in self.cfg.zones.items()])
        else:
            pressure_value = np.array([value[pressure_key][wind_dir_idx]
                                       for _, value in self.cfg.zones.items()])

        if pressure_key == 'edge':
            v_min = 0.0
            v_max = 1.0
        else:
            _min, _max = min(pressure_value), max(pressure_value)

            if _min == _max:
                v_min = -1.0 + _min
                v_max = 1.0 + _max
            else:
                v_min = _min - np.sign(_min) * 0.05*_min
                v_max = _max + np.sign(_max) * 0.05*_max

        try:
            plot_pressure_show(self.ui.mplpressure,
                               coords,
                               pressure_value,
                               self.cfg.house['length'],
                               self.cfg.house['width'],
                               v_min=v_min,
                               v_max=v_max)
        except ValueError:
            self.logger.warning('Can not plot {}'.format(pressure_key))

    def cpi_plot_change(self):
        if self.has_run:
            house_number = self.ui.spinBox_cpi.value()
            self.updateCpiPlot(house_number)


        # wall_major_rows = 2
        # wall_major_cols = self.cfg.house['roof_cols']
        # wall_minor_rows = 2
        # wall_minor_cols = 8
        #
        # for group_name, grouped in self.cfg.connections.groupby('group_name'):
        #     if group_name not in ('wallcladding', 'wallcollapse'):
        #         continue
        #
        #     v_south_grid = ones((wall_major_rows, wall_major_cols), dtype=float32) * blue_v + 10.0
        #     v_north_grid = ones((wall_major_rows, wall_major_cols), dtype=float32) * blue_v + 10.0
        #     v_west_grid = ones((wall_minor_rows, wall_minor_cols), dtype=float32) * blue_v + 10.0
        #     v_east_grid = ones((wall_minor_rows, wall_minor_cols), dtype=float32) * blue_v + 10.0
        #
        #     # construct south grid
        #     for gridCol in range(0, wall_major_cols):
        #         for gridRow in range(0, wall_major_rows):
        #             colChar = chr(ord('A') + gridCol)
        #             loc = 'WS%s%d' % (colChar, gridRow + 1)
        #             conn = connByZoneTypeMap[loc].get(group_name)
        #
        #         if conn and conn.result_failure_v > 0:
        #             v_south_grid[gridRow][gridCol] = conn.result_failure_v
        #
        #     # construct north grid
        #     for gridCol in range(0, wall_major_cols):
        #         for gridRow in range(0, wall_major_rows):
        #             colChar = chr(ord('A') + gridCol)
        #             loc = 'WN%s%d' % (colChar, gridRow + 1)
        #             conn = connByZoneTypeMap[loc].get(group_name)
        #
        #         if conn and conn.result_failure_v > 0:
        #             v_north_grid[gridRow][gridCol] = conn.result_failure_v
        #
        #     # construct west grid
        #     for gridCol in range(0, wall_minor_cols):
        #         for gridRow in range(0, wall_minor_rows):
        #             loc = 'WW%d%d' % (gridCol + 2, gridRow + 1)
        #             conn = connByZoneTypeMap[loc].get(group_name)
        #
        #             if conn and conn.result_failure_v > 0:
        #                 v_west_grid[gridRow][gridCol] = conn.result_failure_v
        #
        #     # construct east grid
        #     for gridCol in range(0, wall_minor_cols):
        #         for gridRow in range(0, wall_minor_rows):
        #             loc = 'WE%d%d' % (gridCol + 2, gridRow + 1)
        #             conn = connByZoneTypeMap[loc].get(group_name)
        #
        #             if conn and conn.result_failure_v > 0:
        #                 v_east_grid[gridRow][gridCol] = conn.result_failure_v
        #
        #     plot_wall_damage_show(group_name,v_south_grid, v_north_grid, v_west_grid,
        #                           v_east_grid, wall_major_cols, wall_major_rows,
        #                           wall_minor_cols, wall_minor_rows,red_v, blue_v)

        # self.ui.damages_tab.setUpdatesEnabled(True)

    def updateInfluence(self):
        conn_name = int(self.ui.spinBox_influence.value())
        plot_influence(self.ui.mplinfluecnes, self.cfg, conn_name)

    def updateComboBox_patch(self):

        failed_conn_name = int(self.ui.spinBox_patch.value())

        try:
            sub_list = sorted(self.cfg.influence_patches[failed_conn_name].keys())
        except KeyError:
            self.logger.info('patch is not defined for conn {}'.format(failed_conn_name))
            self.ui.comboBox_patch.clear()
        else:
            sub_list = [str(x) for x in sub_list]
            self.disconnect(self.ui.comboBox_patch,
                            SIGNAL("currentIndexChanged(QString)"),
                            self.updatePatch)
            self.ui.comboBox_patch.clear()
            self.ui.comboBox_patch.addItems(sub_list)
            self.connect(self.ui.comboBox_patch,
                         SIGNAL("currentIndexChanged(QString)"),
                         self.updatePatch)

            idx = self.ui.comboBox_patch.findText(sub_list[0])
            if self.ui.comboBox_patch.currentIndex() == idx:
                self.updatePatch()
            else:
                self.ui.comboBox_patch.setCurrentIndex(idx)

    def updatePatch(self):
        failed_conn_name = self.ui.spinBox_patch.value()
        try:
            conn_name = int(self.ui.comboBox_patch.currentText())
        except ValueError:
            pass
        else:
            plot_influence_patch(self.ui.mplpatches, self.cfg, failed_conn_name,
                             conn_name)

    def updateWaterIngressPlot(self):
        self.statusBar().showMessage('Plotting Water Ingress')

        _array = self.results_dict['house']['water_ingress_cost']
        wi_means = _array.mean(axis=1)

        self.ui.wateringress_plot.axes.hold(True)
        self.ui.wateringress_plot.axes.scatter(
            self.cfg.wind_speeds[:, np.newaxis] * np.ones(shape=(1, self.cfg.no_models)),
            _array, c='k', s=8, marker='+', label='_nolegend_')
        self.ui.wateringress_plot.axes.plot(self.cfg.wind_speeds, wi_means, c='b', marker='o')
        self.ui.wateringress_plot.axes.set_title('Water Ingress By Wind Speed')
        self.ui.wateringress_plot.axes.set_xlabel('Impact Wind speed (m/s)')
        self.ui.wateringress_plot.axes.set_ylabel('Water Ingress Cost')
        self.ui.wateringress_plot.axes.figure.canvas.draw()
        self.ui.wateringress_plot.axes.set_xlim(self.cfg.wind_speeds[0],
                                                self.cfg.wind_speeds[-1])
        # self.ui.wateringress_plot.axes.set_ylim(0)

    def updateCpiPlot(self, house_number=0):
        self.statusBar().showMessage('Plotting Cpi')

        _array = self.results_dict['house']['cpi']
        _means = _array.mean(axis=1)

        self.ui.cpi_plot.axes.plot(self.cfg.wind_speeds, _means, c='b', marker='o', label='mean')

        if house_number:
            self.ui.cpi_plot.axes.hold(True)
            self.ui.cpi_plot.axes.plot(self.cfg.wind_speeds, _array[:, house_number-1],
                                       c='r', marker='+', label='{:d}'.format(house_number))

        self.ui.cpi_plot.axes.set_title('Internal Pressure Coefficient')
        self.ui.cpi_plot.axes.set_xlabel('Wind speed (m/s)')
        self.ui.cpi_plot.axes.set_ylabel('Cpi')
        self.ui.cpi_plot.axes.legend(loc=0, scatterpoints=1)
        self.ui.cpi_plot.axes.set_xlim(self.cfg.wind_speeds[0],
                                       self.cfg.wind_speeds[-1])
        self.ui.cpi_plot.axes.figure.canvas.draw()
        self.ui.cpi_plot.axes.hold(False)
        # self.ui.wateringress_plot.axes.set_ylim(0)

    def updateBreachPlot(self):
        self.statusBar().showMessage('Plotting Debris Results')
        self.ui.breaches_plot.axes.figure.clf()
        # we'll have three seperate y axis running at different scales

        breaches = self.results_dict['house']['window_breached'].sum(axis=1) / self.cfg.no_models
        nv_means = self.results_dict['house']['no_debris_impacts'].mean(axis=1)
        num_means = self.results_dict['house']['no_debris_items'].mean(axis=1)

        host = SubplotHost(self.ui.breaches_plot.axes.figure, 111)
        par1 = host.twinx()
        par2 = host.twinx()
        par2.axis['right'].set_visible(False)
        offset = 60, 0
        new_axisline = par2.get_grid_helper().new_fixed_axis
        par2.axis['right2'] = new_axisline(loc='right', axes=par2, offset=offset)
        par2.axis['right2'].label.set_visible(True)
        par2.axis['right2'].set_label('Supply')
            
        self.ui.breaches_plot.axes.figure.add_axes(host)
        self.ui.breaches_plot.axes.figure.subplots_adjust(right=0.75)

        host.set_ylabel('Perc Breached')
        host.set_xlabel('Wind Speed (m/s)')
        par1.set_ylabel('Impacts')
        host.set_xlim(self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])
                
        p1, = host.plot(self.cfg.wind_speeds, breaches, label='Breached', c='b')
        p2, = par1.plot(self.cfg.wind_speeds, nv_means, label='Impacts', c='g')
        p2b = par1.plot(self.cfg.wind_speeds, nv_means.cumsum(), label='_nolegend_', c='g')
        p3, = par2.plot(self.cfg.wind_speeds, num_means, label='Supply', c='r')
        p3b = par2.plot(self.cfg.wind_speeds, num_means.cumsum(), label='_nolegend_', c='r')
        
        host.axis['left'].label.set_color(p1.get_color())
        par1.axis['right'].label.set_color(p2.get_color())
        par2.axis['right2'].label.set_color(p3.get_color())
        
        host.legend(loc=2)
        self.ui.breaches_plot.axes.figure.canvas.draw()

    def updateStrengthPlot(self):
        self.ui.connection_type_plot.axes.hold(False)
        
        xlabels = []
        for _id, (type_name, grouped) in enumerate(self.cfg.connections.groupby('type_name')):
            _array = np.array([self.results_dict['connection']['strength'][i] for i in grouped.index]).flatten()

            self.ui.connection_type_plot.axes.scatter(
                _id * np.ones_like(_array), _array, s=8, marker='+')
            self.ui.connection_type_plot.axes.hold(True)
            self.ui.connection_type_plot.axes.scatter(_id, _array.mean(), s=20,
                                                      c='r', marker='o')
            xlabels.append(type_name)

        self.ui.connection_type_plot.axes.set_xticks(range(_id + 1))
        self.ui.connection_type_plot.axes.set_xticklabels(xlabels, rotation='vertical')
        self.ui.connection_type_plot.axes.set_title('Connection Type Strengths')
        self.ui.connection_type_plot.axes.set_position([0.05, 0.20, 0.9, 0.75])
        self.ui.connection_type_plot.axes.figure.canvas.draw()     
        self.ui.connection_type_plot.axes.set_xlim((-0.5, len(xlabels)))

    def updateTypeDamagePlot(self):

        self.ui.connection_type_damages_plot.axes.hold(False)

        xlabels = []
        for _id, (type_name, grouped) in enumerate(self.cfg.connections.groupby('type_name')):
            _array = np.array([self.results_dict['connection']['capacity'][i] for i in grouped.index]).flatten()
            _array[_array == -1] = np.nan

            self.ui.connection_type_damages_plot.axes.scatter(
                _id * np.ones_like(_array), _array, s=8, marker='+')
            self.ui.connection_type_damages_plot.axes.hold(True)
            self.ui.connection_type_damages_plot.axes.scatter(_id,
                                                              np.nanmean(_array),
                                                              s=20,
                                                              c='r',
                                                              marker='o')
            xlabels.append(type_name)

        self.ui.connection_type_damages_plot.axes.set_xticks(range(_id + 1))
        self.ui.connection_type_damages_plot.axes.set_xticklabels(xlabels, rotation='vertical')
        self.ui.connection_type_damages_plot.axes.set_title('Connection Type Damage Speeds')
        self.ui.connection_type_damages_plot.axes.set_position([0.05, 0.20, 0.9, 0.75])
        self.ui.connection_type_damages_plot.axes.figure.canvas.draw()     
        self.ui.connection_type_damages_plot.axes.set_xlim((-0.5, len(xlabels)))

    def updateLoadPlot(self):

        self.statusBar().showMessage('updating connection load')

        group_name = str(self.ui.comboBox_load.currentText())
        house_number = self.ui.spinBox_load.value()
        ispeed = np.argmin(np.abs(self.cfg.wind_speeds-self.ui.doubleSpinBox_load.value()))

        grouped = self.cfg.connections.loc[
            self.cfg.connections.group_name == group_name]

        load = np.array([self.results_dict['connection']['load'][i]
                         for i in grouped.index])[:, ispeed, house_number-1]

        _min, _max = min(load), max(load)
        v_min = _min - np.sign(_min) * 0.05 * _min
        v_max = _max + np.sign(_max) * 0.05 * _max

        load[load == 0.0] = -1.0 * np.inf

        try:
            plot_load_show(self.ui.mplload, grouped,
                           load,
                           self.cfg.house['length'],
                           self.cfg.house['width'],
                           v_min=v_min,
                           v_max=v_max)
        except ValueError:
            self.logger.warning('Can not plot connection load')

    def updateConnectionTypePlots(self):
        self.statusBar().showMessage('Plotting Connection Types')
        self.updateStrengthPlot()
        self.updateTypeDamagePlot()
        self.updateLoadPlot()

    def updateConnectionTable_with_results(self):
        self.statusBar().showMessage('Updating Connections Table')

        connections_damaged = self.results_dict['connection']['damaged']
        for irow, item in enumerate(self.cfg.connections.iterrows()):
            conn_id = item[0]
            failure_count = np.count_nonzero(connections_damaged[conn_id])
            failure_mean = np.mean(connections_damaged[conn_id])
            self.ui.connections.setItem(irow, 4, QTableWidgetItem('{:.3}'.format(failure_mean)))
            self.ui.connections.setItem(irow, 5, QTableWidgetItem('{}'.format(failure_count)))

    # def determine_capacities(self):
    #
    #     _array = np.array([self.results_dict['connection']['capacity'][i]
    #                     for i in self.cfg.list_connections])
    #
    #     _array[_array == -1] = np.nan
    #
    #     return np.nanmean(_array, axis=1)

    def updateHouseResultsTable(self):

        self.statusBar().showMessage('Updating Zone Results')
        self.ui.zoneResults.clear()

        wind_dir = [self.results_dict['house']['wind_dir_index'][i]
                    for i in range(self.cfg.no_models)]
        construction_level = [self.results_dict['house']['construction_level'][i]
                              for i in range(self.cfg.no_models)]

        for i in range(self.cfg.no_models):
            parent = QTreeWidgetItem(self.ui.zoneResults,
                                     ['#{} ({}/{})'.format(i + 1,
                                                           WIND_DIR[wind_dir[i]],
                                                           construction_level[i]), '', '', ''])
            for zr_key in self.cfg.zones:
                QTreeWidgetItem(parent, ['',
                                         zr_key,
                                         '{:.3}'.format(self.results_dict['zone']['cpe'][zr_key][0, i]),
                                         '{:.3}'.format(self.results_dict['zone']['cpe_str'][zr_key][0, i]),
                                         '{:.3}'.format(self.results_dict['zone']['cpe_eave'][zr_key][0, i])])
                
        self.ui.zoneResults.resizeColumnToContents(0)
        self.ui.zoneResults.resizeColumnToContents(1)
        self.ui.zoneResults.resizeColumnToContents(2)
        self.ui.zoneResults.resizeColumnToContents(3)
        self.ui.zoneResults.resizeColumnToContents(4)
        self.ui.zoneResults.resizeColumnToContents(5)
                
        self.statusBar().showMessage('Updating Connection Results')

        self.ui.connectionResults.clear()
        for i in range(self.cfg.no_models):
            parent = QTreeWidgetItem(self.ui.connectionResults,
                                     ['#{} ({}/{})'.format(i + 1,
                                                           WIND_DIR[wind_dir[i]],
                                                           construction_level[i]), '', '', ''])

            for _id, value in self.cfg.connections.iterrows():

                capacity = self.results_dict['connection']['capacity'][_id][0, i]
                # load = self.results_dict['connection']['load'][_id][house_num]
                dead_load = self.results_dict['connection']['dead_load'][_id][0, i]
                strength = self.results_dict['connection']['strength'][_id][0, i]

                load_last = self.results_dict['connection']['load'][_id][-1, i]
                if load_last:
                    load_desc = '{:.3}'.format(load_last)
                else:
                    load_desc = 'collapse'

                conn_parent = QTreeWidgetItem(parent, ['{}_{}'.format(value['type_name'], _id),
                                                       '{:.3}'.format(capacity if capacity > 0 else 'NA'),
                                                       '{:.3}'.format(strength),
                                                       '{:.3}'.format(dead_load),
                                                       '{}'.format(load_desc)])

                # TODO for infl_dict in damage_report.get('infls', {}):
                #     QTreeWidgetItem(conn_parent, ['%.3f(%s) = %.3f(i) * %.3f(area) * %.3f(pz)' % (infl_dict['load'],
                #                                                                                   infl_dict['name'],
                #                                                                                   infl_dict['infl'],
                #                                                                                   infl_dict['area'],
                #                                                                                   infl_dict['pz'])])
        self.ui.connectionResults.resizeColumnToContents(1)
        self.ui.connectionResults.resizeColumnToContents(2)
        self.ui.connectionResults.resizeColumnToContents(3)
        self.ui.connectionResults.resizeColumnToContents(4)
        header_view = self.ui.connectionResults.header()
        header_view.resizeSection(0, 350)

    def open_scenario(self, config_file=None):
        settings = QSettings()
        # Set the path to the default
        scenario_path = SCENARIOS_DIR
        if config_file:
            scenario_path = config_file
        elif settings.contains("ScenarioFolder"):
            # we have a saved scenario path so use that
            scenario_path = unicode(settings.value("ScenarioFolder").toString())

        filename = QFileDialog.getOpenFileName(self,
                                               "Scenarios",
                                               scenario_path,
                                               "Scenarios (*.cfg)")
        if not filename:
            return

        config_file = '{}'.format(filename)
        if os.path.isfile(config_file):
            self.file_load(config_file)
            settings.setValue("ScenarioFolder", QVariant(QString(os.path.dirname(config_file))))
        else:
            msg = 'Unable to load scenario: {}\nFile not found.'.format(config_file)
            QMessageBox.warning(self, "VAWS Program Warning", unicode(msg))

    def save_scenario(self):
        self.update_config_from_ui()
        self.cfg.save_config()
        self.ui.statusbar.showMessage('Saved to file {}'.format(self.cfg.file_cfg))
        # self.update_ui_from_config()
        
    def save_as_scenario(self):
        # TODO: check
        self.update_config_from_ui()
        current_parent, _ = os.path.split(self.cfg.path_cfg)

        fname = unicode(QFileDialog.getSaveFileName(self, "VAWS - Save Scenario",
                                                    current_parent, CONFIG_TEMPL))
        if len(fname) > 0:
            if "." not in fname:
                fname += ".cfg"
            # check we have a directory for the scenario
            path_cfg, file_suffix_name = os.path.split(fname)
            file_name, ext = os.path.splitext(file_suffix_name)
            if file_name not in path_cfg:
                # we need to create a directory for the scenario
                path_cfg = os.path.join(path_cfg, file_name)

            if not os.path.isdir(path_cfg):
                os.mkdir(path_cfg)

            if not os.path.isdir(os.path.join(path_cfg, INPUT_DIR)):
                default_input = os.path.join(self.cfg.path_cfg, INPUT_DIR)
                shutil.copytree(default_input,
                                os.path.join(path_cfg, INPUT_DIR))

            if not os.path.isdir(os.path.join(path_cfg, OUTPUT_DIR)):
                os.mkdir(os.path.join(path_cfg, OUTPUT_DIR))

            settings = QSettings()
            settings.setValue("ScenarioFolder", QVariant(QString(path_cfg)))
            self.cfg.path_cfg = path_cfg
            self.cfg.file_cfg = os.path.join(path_cfg, file_suffix_name)
            self.cfg.output_path = os.path.join(path_cfg, OUTPUT_DIR)

            self.cfg.save_config()
            set_logger(self.cfg)
            self.update_ui_from_config()
        else:
            msg = 'No scenario name entered. Action cancelled'
            QMessageBox.warning(self, "VAWS Program Warning", unicode(msg))

#    def set_scenario(self, s=None):
        # if s:
        #     self.cfg = s
        #     self.cfg.process_config()

        # self.update_ui_from_config()
        #
        # self.statusBar().showMessage('Ready')

    def closeEvent(self, event):
        if self.okToContinue():
            if self.cfg.file_cfg and QFile.exists(self.cfg.file_cfg):
                settings = QSettings()
                filename = QVariant(QString(self.cfg.file_cfg))
                settings.setValue("LastFile", filename)
            self.storeSizePosToSettings()
        else:
            event.ignore()
        
    def update_ui_from_config(self):
        if self.cfg:
            self.statusBar().showMessage('Updating', 1000)

            # Scenario
            self.ui.numHouses.setText('{:d}'.format(self.cfg.no_models))
            self.ui.houseName.setText(self.cfg.model_name)
            self.ui.seedRandom.setText(str(self.cfg.random_seed))

            idx = self.ui.terrainCategory.findText(self.cfg.file_wind_profiles)
            if self.ui.terrainCategory.currentIndex() == idx:
                self.updateTerrainCategoryTable()
            else:
                self.ui.terrainCategory.setCurrentIndex(idx)

            self.ui.regionalShielding.setText('{:.1f}'.format(
                self.cfg.regional_shielding_factor))
            self.ui.windMin.setValue(self.cfg.wind_speed_min)
            self.ui.windMax.setValue(self.cfg.wind_speed_max)
            self.ui.windIncrement.setText('{:.3f}'.format(self.cfg.wind_speed_increment))
            self.ui.windDirection.setCurrentIndex(self.cfg.wind_dir_index)

            # Debris
            idx = self.ui.debrisRegion.findText(self.cfg.region_name)
            if self.ui.debrisRegion.currentIndex() == idx:
                self.updateDebrisRegionsTable()
            else:
                self.ui.debrisRegion.setCurrentIndex(idx)

            self.ui.buildingSpacing.setCurrentIndex(
                self.ui.buildingSpacing.findText(
                    '{:.1f}'.format(self.cfg.building_spacing)))
            self.ui.debrisRadius.setValue(self.cfg.debris_radius)
            self.ui.debrisAngle.setValue(self.cfg.debris_angle)
            self.ui.sourceItems.setValue(self.cfg.source_items)
            self.ui.debrisBoundary.setText(
                '{:.3f}'.format(self.cfg.boundary_radius))
            # self.ui.flighttimeMean.setText(
            #     '{:.3f}'.format(self.cfg.flight_time_mean))
            # self.ui.flighttimeStddev.setText(
            #     '{:.3f}'.format(self.cfg.flight_time_stddev))
            self.ui.staggeredDebrisSources.setChecked(self.cfg.staggered_sources)
            self.ui.debris.setChecked(self.cfg.flags['debris'])

            # construction levels
            # self.ui.constructionEnabled.setChecked(self.cfg.flags['construction_levels'])
            self.ui.constLevels.setText(
                ', '.join(self.cfg.construction_levels_levels))
            self.ui.constProbs.setText(
                ', '.join([str(x) for x in self.cfg.construction_levels_probs]))
            self.ui.constMeans.setText(
                ', '.join([str(x) for x in self.cfg.construction_levels_mean_factors]))
            self.ui.constCovs.setText(
                ', '.join([str(x) for x in self.cfg.construction_levels_cv_factors]))

            # water ingress
            self.ui.waterThresholds.setText(
                ', '.join([str(x) for x in self.cfg.water_ingress_i_thresholds]))
            self.ui.waterSpeed0.setText(
                ', '.join([str(x) for x in self.cfg.water_ingress_i_speed_at_zero_wi]))
            self.ui.waterSpeed1.setText(
                ', '.join([str(x) for x in self.cfg.water_ingress_i_speed_at_full_wi]))
            self.ui.waterEnabled.setChecked(self.cfg.flags.get('water_ingress'))

            # options
            self.ui.fragilityStates.setText(
                ', '.join(self.cfg.fragility_i_states))
            self.ui.fragilityThresholds.setText(
                ', '.join([str(x) for x in self.cfg.fragility_i_thresholds]))
            self.ui.diffShielding.setChecked(self.cfg.flags.get('differential_shielding'))

            self.ui.redV.setValue(self.cfg.heatmap_vmin)
            self.ui.blueV.setValue(self.cfg.heatmap_vmax)
            self.ui.vStep.setValue(self.cfg.heatmap_vstep)

            self.update_house_panel()
            # self.updateGlobalData()

            self.ui.actionRun.setEnabled(True)

            if self.cfg.file_cfg:
                self.statusBarScenarioLabel.setText(
                    'Scenario: {}'.format(os.path.basename(self.cfg.file_cfg)))
        else:
            self.statusBarScenarioLabel.setText('Scenario: None')
        
    def update_config_from_ui(self):
        new_cfg = self.cfg

        # Scenario section
        new_cfg.no_models = int(self.ui.numHouses.text())
        new_cfg.model_name = str(self.ui.houseName.text())
        new_cfg.file_wind_profiles = str(self.ui.terrainCategory.currentText())
        new_cfg.regional_shielding_factor = float(self.ui.regionalShielding.text())
        new_cfg.wind_speed_min = self.ui.windMin.value()
        new_cfg.wind_speed_max = self.ui.windMax.value()
        new_cfg.wind_speed_increment = float(self.ui.windIncrement.text())
        # new_cfg.set_wind_speeds()
        new_cfg.wind_direction = WIND_DIR[
            self.ui.windDirection.currentIndex()]

        # Debris section
        new_cfg.set_region_name(str(self.ui.debrisRegion.currentText()))
        new_cfg.building_spacing = float(self.ui.buildingSpacing.currentText())
        new_cfg.debris_radius = self.ui.debrisRadius.value()
        new_cfg.debris_angle = self.ui.debrisAngle.value()
        new_cfg.source_items = self.ui.sourceItems.value()
        # if self.ui.flighttimeMean.text():
        #     new_cfg.flight_time_mean = float(self.ui.flighttimeMean.text())
        # if self.ui.flighttimeStddev.text():
        #     new_cfg.flight_time_stddev = float(self.ui.flighttimeStddev.text())
        if self.ui.debrisBoundary.text():
            new_cfg.boundary_radius = float(self.ui.debrisBoundary.text())
        new_cfg.staggered_sources = self.ui.staggeredDebrisSources.isChecked()

        new_cfg.flags['water_ingress'] = self.ui.waterEnabled.isChecked()
        new_cfg.flags['differential_shielding'] = self.ui.diffShielding.isChecked()
        new_cfg.flags['debris'] = self.ui.debris.isChecked()

        # option section
        new_cfg.random_seed = int(self.ui.seedRandom.text())
        new_cfg.heatmap_vmin = float(self.ui.redV.value())
        new_cfg.heatmap_vmax = float(self.ui.blueV.value())
        new_cfg.heatmap_vstep = float(self.ui.vStep.value())

        # construction section
        # new_cfg.flags['construction_levels'] = self.ui.constructionEnabled.isChecked()
        new_cfg.construction_levels_levels = [
            x.strip() for x in str(self.ui.constLevels.text()).split(',')]
        new_cfg.construction_levels_probs = [
            float(x) for x in unicode(self.ui.constProbs.text()).split(',')]
        new_cfg.construction_levels_mean_factors = [
            float(x) for x in unicode(self.ui.constMeans.text()).split(',')]
        new_cfg.construction_levels_cov_factors = [
            float(x) for x in unicode(self.ui.constCovs.text()).split(',')]

        # fragility section
        new_cfg.fragility_i_thresholds = [
            float(x) for x in unicode(self.ui.fragilityThresholds.text()).split(',')]
        new_cfg.fragility_i_states = [
            x.strip() for x in unicode(self.ui.fragilityStates.text()).split(',')]

        # water ingress
        new_cfg.flags['water_ingress'] = self.ui.waterEnabled.isChecked()
        new_cfg.water_ingress_i_thresholds = [
            float(x) for x in unicode(self.ui.waterThresholds.text()).split(',')]
        new_cfg.water_ingress_i_zero_wi = [
            float(x) for x in unicode(self.ui.waterSpeed0.text()).split(',')]
        new_cfg.water_ingress_i_full_wi = [
            float(x) for x in unicode(self.ui.waterSpeed1.text()).split(',')]

        # house / groups section
        # for irow, (index, ctg) in enumerate(self.cfg.groups.items()):
        #     cellWidget = self.ui.connGroups.cellWidget(irow, 4)
        #     new_cfg.flags['conn_type_group_{}'.format(index)] = True \
        #         if cellWidget.checkState() == Qt.Checked else False

    def file_load(self, fname):
        try:
            path_cfg = os.path.dirname(os.path.realpath(fname))
            set_logger(path_cfg)
            self.cfg = Config(fname)
            self.init_terrain_category()
            self.init_debris_region()
            self.init_influence_and_patch()
            self.update_ui_from_config()
            self.statusBar().showMessage('Ready')
        except Exception as excep:
            self.logger.exception("Loading configuration caused exception")

            msg = 'Unable to load previous scenario: %s\nLoad cancelled.' % fname
            QMessageBox.warning(self, "VAWS Program Warning", unicode(msg))

    def okToContinue(self):
        self.update_config_from_ui()
        if self.dirty_scenario:
            reply = QMessageBox.question(self,
                                         "WindSim - Unsaved Changes",
                                         "Save unsaved changes?",
                                         QMessageBox.Yes | QMessageBox.No |
                                         QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return False
            elif reply == QMessageBox.Yes:
                self.save_scenario()
        return True
    
    def testDebrisSettings(self):
        self.update_config_from_ui()
        self.cfg.process_config()

        shape_type = {'Compact': 'c', 'Sheet': 'g', 'Rod': 'r'}

        wind_speed, ok = QInputDialog.getInteger(
            self, "Debris Test", "Wind speed (m/s):", 50, 10, 200)

        if ok:

            rnd_state = np.random.RandomState(1)
            incr_speed = self.cfg.wind_speeds[1] - self.cfg.wind_speeds[0]

            damage_incr = vulnerability_weibull_pdf(
                x=wind_speed, **VUL_DIC[self.cfg.region_name]) * incr_speed

            mean_no_debris_items = np.rint(self.cfg.source_items * damage_incr)

            debris_items = generate_debris_items(
                rnd_state=rnd_state,
                wind_speed=wind_speed,
                cfg=self.cfg,
                mean_no_debris_items=mean_no_debris_items)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            # source_x, source_y = [], []
            # for source in self.cfg.debris_sources:
            #     source_x.append(source.x)
            #     source_y.append(source.y)
            # ax.scatter(source_x, source_y, label='source', color='b')
            # ax.scatter(0, 0, label='target', color='r')

            # add footprint
            # _array = np.array(house.debris.footprint.exterior.xy).T
            #
            # ax.add_patch(patches.Polygon(_array, alpha=0.5))
            # ax.add_patch(patches.Polygon(self.cfg.impact_boundary.exterior, alpha=0.5))

            for item in debris_items:
                _x, _y = item.trajectory.xy[0][1], item.trajectory.xy[1][1]
                ax.scatter(_x, _y, color=shape_type[item.type], alpha=0.2)

            for source in self.cfg.debris_sources:
                ax.scatter(source.x, source.y, color='b', label='source')
            ax.scatter(0, 0, label='target', color='r')

            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc=2, scatterpoints=1)

            title_str = 'Debris samples at {0:.3f} m/s in region of {1}'.format(
                wind_speed, self.cfg.region_name)
            ax.set_title(title_str)

            ax.axes.set_xlim(-0.5*self.cfg.debris_radius, self.cfg.debris_radius)
            ax.axes.set_ylim(-1.0*self.cfg.debris_radius, self.cfg.debris_radius)
            # fig.canvas.draw()
            fig.show()

    def testConstructionLevels(self):
        self.update_config_from_ui()
        self.cfg.process_config()

        selected_type, ok = QInputDialog.getItem(self, "Construction Test", "Connection Type:",
            self.cfg.types.keys(), 0, False)

        if ok:

            house = House(self.cfg, seed=1)
            lognormal_strength = self.cfg.types['{}'.format(selected_type)]['lognormal_strength']
            mu, std = compute_arithmetic_mean_stddev(*lognormal_strength)
            mu *= house.mean_factor
            std *= house.cv_factor * house.mean_factor

            x = []
            n = 50000
            for i in xrange(n):
                rv = sample_lognorm_given_mean_stddev(mu, std, house.rnd_state)
                x.append(rv)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
            _mean = self.cfg.types['{}'.format(selected_type)]['strength_mean']
            _std = self.cfg.types['{}'.format(selected_type)]['strength_std']
            title_str = 'Sampled strength of {0} \n ' \
                        'construction level: {1}, mean: {2:.2f}, std: {3:.2f}'.format(
                selected_type, house.construction_level, _mean, _std)
            ax.set_title(title_str)
            fig.show()

    def testWaterIngress(self):

        self.update_config_from_ui()
        self.cfg.process_config()

        di_array = []
        dic_thresholds = {}
        for i, value in enumerate(self.cfg.water_ingress.index):
            if i == 0:
                dic_thresholds[i] = (0.0, value)
            else:
                dic_thresholds[i] = (self.cfg.water_ingress.index[i-1], value)

            di_array.append(0.5*(dic_thresholds[i][0] + dic_thresholds[i][1]))

        a = np.zeros((len(di_array), len(self.cfg.wind_speeds)))

        for i, di in enumerate(di_array):
            for j, speed in enumerate(self.cfg.wind_speeds):
                a[i, j] = 100.0 * compute_water_ingress_given_damage(
                    di, speed, self.cfg.water_ingress)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for j in range(a.shape[0]):
            ax.plot(self.cfg.wind_speeds, a[j, :],
                    label='{:.1f} <= DI < {:.1f}'.format(*dic_thresholds[j]))

        ax.legend(loc=1)
        ax.set_xlabel('Wind speed (m/s)')
        ax.set_ylabel('Water ingress (%)')
        fig.show()


def run_gui():
    parser = process_commandline()

    (options, args) = parser.parse_args()

    if not options.config_file:
        initial_scenario = DEFAULT_SCENARIO
    else:
        initial_scenario = options.config_file

    path_cfg = os.path.dirname(os.path.realpath(initial_scenario))
    if options.verbose:
        set_logger(path_cfg, logging_level=options.verbose)
    else:
        set_logger(path_cfg)

    initial_config = Config(file_cfg=initial_scenario)

    app = QApplication(sys.argv)
    app.setOrganizationName("Geoscience Australia")
    app.setOrganizationDomain("ga.gov.au")
    app.setApplicationName("VAWS")
    splash_image = QPixmap()

    splash_file = os.path.join(SOURCE_DIR, 'images/splash/splash.png')
    if not splash_image.load(splash_file):
        raise Exception('Could not load splash image')

    app.processEvents()

    global my_app
    my_app = MyForm(init_scenario=initial_config)

    splash = QSplashScreen(my_app, splash_image)
    splash.show()

    my_app.show()

    time.sleep(5)
    splash.finish(my_app)

    sys.exit(app.exec_())

if __name__ == '__main__':
    run_gui()
