#!/usr/bin/env python
# adjust python path so we may import things from peer packages
import sys
import shutil
import time
import os.path
import logging
import warnings
import h5py
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from mpl_toolkits.axisartist.parasite_axes import ParasiteAxes
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QSettings, QVariant, QFile
from PyQt5.QtWidgets import QProgressBar, QLabel, QMainWindow, QApplication, QTableWidget, \
                        QTableWidgetItem, QFileDialog, \
                        QMessageBox, QTreeWidgetItem, QInputDialog, QSplashScreen

from vaws.model.constants import WIND_DIR, DEBRIS_TYPES_KEYS, VUL_DIC, BLDG_SPACING, DEBRIS_VULNERABILITY, DEBRIS_TYPES_CLRS
from vaws.model.debris import generate_debris_items
from vaws.gui.house import HouseViewer
from vaws.model.curve import vulnerability_lognorm, vulnerability_weibull, vulnerability_weibull_pdf

from vaws.gui.main_ui import Ui_main
from vaws.model.main import process_commandline, set_logger, \
    simulate_wind_damage_to_houses
from vaws.model.config import Config, INPUT_DIR, OUTPUT_DIR, FILE_RESULTS, DEBRIS_ITEMS, WATER_INGRESS_ITEMS
from vaws.model.version import VERSION_DESC
from vaws.model.damage_costing import compute_water_ingress_given_damage

from vaws.gui.output import plot_wind_event_damage, plot_wind_event_mean, \
                        plot_wind_event_show, plot_fitted_curve, \
                        plot_fragility_show, plot_damage_show, plot_influence, \
                        plot_influence_patch, plot_load_show, plot_pressure_show

from vaws.gui.mixins import PersistSizePosMixin, setupTable, finiTable

my_app = None

SOURCE_DIR = os.path.dirname(__file__)
VAWS_DIR = os.sep.join(SOURCE_DIR.split(os.sep)[:-2])
SCENARIOS_DIR = os.sep.join(SOURCE_DIR.split(os.sep)[:-1])
SCENARIOS_DIR = os.path.join(SCENARIOS_DIR, 'scenarios')
CONFIG_TEMPL = "Scenarios (*.cfg)"
VUL_OUTPUT_TEMPL = "Output file (*.csv)"
DEFAULT_SCENARIO = os.path.join(SCENARIOS_DIR, 'default', 'default.cfg')

PRESSURE_KEYS = ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'cpi_alpha',
                 'edge']
WIND_4_TEST_WATER_INGRESS = (0, 200)
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
        self.setWindowTitle(windowTitle)

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
        self.ui.actionOpen_Scenario.triggered.connect(self.open_scenario)
        self.ui.actionRun.triggered.connect(self.runScenario)
        self.ui.actionStop.triggered.connect(self.stopScenario)
        self.ui.actionSave_Scenario.triggered.connect(self.save_scenario)
        self.ui.actionSave_Scenario_As.triggered.connect(self.save_as_scenario)
        self.ui.actionHouse_Info.triggered.connect(self.showHouseInfoDlg)
        self.ui.actionLoad_Results.triggered.connect(self.load_results)
        # TODO: actionNew missing

        # test panel
        self.ui.testDebrisButton.clicked.connect(self.testDebrisSettings)
        self.ui.testWaterIngressButton.clicked.connect(self.testWaterIngress)

        # Scenario panel
        self.ui.windMin.valueChanged.connect(
            lambda x: self.onSliderChanged(self.ui.windMinLabel, x))
        self.ui.windMax.valueChanged.connect(
            lambda x: self.onSliderChanged(self.ui.windMaxLabel, x))

        # debris panel
        self.ui.debrisRadius.valueChanged.connect(
                     lambda x: self.onSliderChanged(self.ui.debrisRadiusLabel, x))
        self.ui.debrisAngle.valueChanged.connect(
                     lambda x: self.onSliderChanged(self.ui.debrisAngleLabel, x))
        self.ui.sourceItems.valueChanged.connect(
                     lambda x: self.onSliderChanged(self.ui.sourceItemsLabel, x))
        self.ui.vuln_pushButton.clicked.connect(self.save_vuln_file)

        # options
        self.ui.redV.valueChanged.connect(
                     lambda x: self.onSliderChanged(self.ui.redVLabel, x))
        self.ui.blueV.valueChanged.connect(
                     lambda x: self.onSliderChanged(self.ui.blueVLabel, x))
        self.ui.vStep.valueChanged.connect(
                     lambda x: self.onSliderChanged(self.ui.vStepLabel, x))
        self.ui.applyDisplayChangesButton.clicked.connect(
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

        # init combobox
        self.init_debrisvuln()

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
        self.ui.spinBox_cost.valueChanged.connect(self.cost_plot_change)

        self.ui.ln_checkBox.stateChanged.connect(self.updateVulnCurve)
        self.ui.wb_checkBox.stateChanged.connect(self.updateVulnCurve)

        self.update_ui_from_config()
        # QTimer.singleShot(0, self.set_scenario)

        self.statusBar().showMessage('Ready')

    def init_terrain_category(self):
        try:
            self.ui.terrainCategory.currentTextChanged.disconnect(
                self.updateTerrainCategoryTable)
        except TypeError:
            pass

        self.ui.terrainCategory.clear()
        self.ui.terrainCategory.addItems(os.listdir(self.cfg.path_wind_profiles))
        self.ui.terrainCategory.currentTextChanged.connect(
            self.updateTerrainCategoryTable)

    def init_debris_region(self):

        try:
            self.ui.debrisRegion.currentIndexChanged.disconnect(
                self.updateDebrisRegionsTable)
        except TypeError:
            pass

        self.ui.debrisRegion.clear()
        self.ui.debrisRegion.addItems(self.cfg.debris_regions.keys())
        self.ui.debrisRegion.currentIndexChanged.connect(self.updateDebrisRegionsTable)

    def init_heatmap_group(self):

        try:
            self.ui.comboBox_heatmap.currentIndexChanged.disconnect(
                self.heatmap_house_change)
        except TypeError:
            pass

        self.ui.comboBox_heatmap.clear()
        self.ui.comboBox_heatmap.addItems(self.cfg.list_groups)
        self.ui.comboBox_heatmap.currentIndexChanged.connect(self.heatmap_house_change)
        self.ui.spinBox_heatmap.valueChanged.connect(self.heatmap_house_change)

    def init_pressure(self):

        self.ui.mplpressure.figure.clear()
        self.ui.mplpressure.axes.cla()
        self.ui.mplpressure.axes.figure.canvas.draw()

        # init combobox
        try:
            self.ui.comboBox_pressure.currentIndexChanged.disconnect(
                self.updatePressurePlot)
        except TypeError:
            pass
        self.ui.comboBox_pressure.clear()
        self.ui.comboBox_pressure.addItems(PRESSURE_KEYS)
        self.ui.comboBox_pressure.currentIndexChanged.connect(self.updatePressurePlot)

        if str(self.ui.comboBox_pressure.currentText()) == PRESSURE_KEYS[0]:
            self.updatePressurePlot()
        else:
            self.ui.comboBox_pressure.setValue(PRESSURE_KEYS[0])

        # init combobox2
        try:
            self.ui.comboBox2_pressure.currentIndexChanged.disconnect(
                self.updatePressurePlot)
        except TypeError:
            pass
        self.ui.comboBox2_pressure.clear()
        self.ui.comboBox2_pressure.addItems(WIND_DIR[:-1])
        self.ui.comboBox2_pressure.currentIndexChanged.connect(self.updatePressurePlot)

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

        try:
            self.ui.comboBox_load.currentIndexChanged.disconnect(
                self.load_connection_change)
        except TypeError:
            pass
        self.ui.comboBox_load.clear()
        self.ui.comboBox_load.addItems(self.cfg.list_groups)
        self.ui.comboBox_load.currentIndexChanged.connect(self.load_connection_change)

        self.ui.spinBox_load.valueChanged.connect(self.load_connection_change)

        self.ui.doubleSpinBox_load.valueChanged.connect(self.load_connection_change)

    def onSliderChanged(self, label, x):
        label.setText(f'{x:d}')

    def updateDisplaySettings(self):
        if self.has_run:
            self.heatmap_house_change()
        else:
            msg = 'Simulation results unavailable'
            QMessageBox.warning(self, 'VAWS Program Warning', msg)


    def showHouseInfoDlg(self):
        dlg = HouseViewer(self.cfg)
        dlg.exec_()

    def init_debrisvuln(self):

        try:
            self.ui.comboBox_debrisVul.currentIndexChanged.disconnect(
                self.updateDebrisVuln)
        except TypeError:
            pass
        self.ui.comboBox_debrisVul.clear()
        self.ui.comboBox_debrisVul.addItems(DEBRIS_VULNERABILITY)
        self.ui.comboBox_debrisVul.currentIndexChanged.connect(self.updateDebrisVuln)

    def updateDebrisVuln(self):

        self.ui.debrisVul_param1.clear()
        self.ui.debrisVul_param2.clear()


    def updateDebrisRegionsTable(self):

        self.cfg.set_region_name(str(self.ui.debrisRegion.currentText()))

        # load up debris regions
        _debris_region = self.cfg.debris_regions[self.cfg.region_name]
        setupTable(self.ui.debrisRegions, _debris_region)

        no_values = len(self.cfg.debris_types[list(self.cfg.debris_types)[0]])

        for i, key in enumerate(DEBRIS_TYPES_KEYS):

            _list = [(k, v) for k, v in _debris_region.items() if key in k]

            _list.sort()

            for j, (k, v) in enumerate(_list):

                self.ui.debrisRegions.setItem(no_values*i + j, 0,
                                              QTableWidgetItem(k))
                self.ui.debrisRegions.setItem(no_values*i + j, 1,
                                              QTableWidgetItem(f'{v:.3f}'))

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
                irow, 0, QTableWidgetItem(f'{head_col:.3f}'))

        for key, _list in self.cfg.wind_profiles.items():
            for irow, value in enumerate(_list):
                self.ui.boundaryProfile.setItem(
                    irow, key, QTableWidgetItem(f'{value:.3f}'))

        self.ui.boundaryProfile.resizeColumnsToContents()

    def updateConnectionTypeTable(self):
        # load up connection types grid
        setupTable(self.ui.connectionsTypes, self.cfg.types)
        for irow, (index, ctype) in enumerate(self.cfg.types.items()):
            self.ui.connectionsTypes.setItem(
                irow, 0, QTableWidgetItem(index))
            self.ui.connectionsTypes.setItem(
                irow, 1, QTableWidgetItem(f"{ctype['strength_mean']:.3f}"))
            self.ui.connectionsTypes.setItem(
                irow, 2, QTableWidgetItem(f"{ctype['strength_std']:.3f}"))
            self.ui.connectionsTypes.setItem(
                irow, 3, QTableWidgetItem(f"{ctype['dead_load_mean']:.3f}"))
            self.ui.connectionsTypes.setItem(
                irow, 4, QTableWidgetItem(f"{ctype['dead_load_std']:.3f}"))
            self.ui.connectionsTypes.setItem(
                irow, 5, QTableWidgetItem(ctype['group_name']))
            self.ui.connectionsTypes.setItem(
                irow, 6, QTableWidgetItem(f"{ctype['costing_area']:.3f}"))
        finiTable(self.ui.connectionsTypes)

    def updateZonesTable(self):
        setupTable(self.ui.zones, self.cfg.zones)

        for irow, (index, z) in enumerate(self.cfg.zones.items()):
            self.ui.zones.setItem(
                irow, 0, QTableWidgetItem(index))
            self.ui.zones.setItem(
                irow, 1, QTableWidgetItem(f"{z['area']:.3f}"))
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
            self.ui.connGroups.setItem(irow, 1, QTableWidgetItem(f"{ctg['dist_order']:d}"))
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
                f'{_inst.surface_area:.3f}'))
            self.ui.damageScenarios.setItem(irow, 2, QTableWidgetItem(
                f'{_inst.envelope_repair_rate:.3f}'))
            self.ui.damageScenarios.setItem(irow, 3, QTableWidgetItem(
                f'{_inst.internal_repair_rate:.3f}'))
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
        ok = self.update_config_from_ui()
        if ok:
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
            self.ui.spinBox_cost.setRange(0, self.cfg.no_models)
            # self.ui.cpi_house.setValue(0)
            # self.ui.cpi_houseLabel.setText('{:d}'.format(0))

            self.ui.cpi_plot.axes.cla()
            self.ui.cpi_plot.axes.figure.canvas.draw()

            self.ui.cost_plot.axes.cla()
            self.ui.cost_plot.axes.figure.canvas.draw()

            self.ui.wateringress_plot.axes.cla()
            self.ui.wateringress_plot.axes.figure.canvas.draw()

            self.ui.wateringress_prop_plot.axes.cla()
            self.ui.wateringress_prop_plot.axes.figure.canvas.draw()

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
                        f'Simulation complete in {run_time:0.3f}')

                    # self.results_dict = bucket
                    self.updateVulnCurve()
                    self.updateFragCurve()
                    self.updateHouseResultsTable()
                    self.updateConnectionTable_with_results()
                    self.updateConnectionTypePlots()
                    self.updateHeatmap()
                    self.updateWaterIngressPlot()
                    self.updateWaterIngressPropPlot()
                    self.updateBreachPlot()
                    self.updateCpiPlot()
                    self.updateCostPlot()
                    self.has_run = True

            except IOError:
                msg = 'A report file is still open by another program unable to run simulation.'
                QMessageBox.warning(self, 'VAWS Program Warning', msg)
                self.statusBar().showMessage('')
            except Exception as err:
                self.statusBar().showMessage(f'Fatal Error Occurred: {str}')
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

        try:
            capacity = np.array([self.results_dict['connection']['capacity'][i]
                             for i in grouped.index])[:, 0, :]
        except KeyError:
            capacity = np.array([self.results_dict['connection']['capacity'][str(i)]
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
                msg = f'Cannot plot {pressure_key}'
                QMessageBox.warning(self, "VAWS Program Warning", msg)

    def cpi_plot_change(self):
        if self.has_run:
            house_number = self.ui.spinBox_cpi.value()
            self.updateCpiPlot(house_number)

    def cost_plot_change(self):
        if self.has_run:
            house_number = self.ui.spinBox_cost.value()
            self.updateCostPlot(house_number)

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
            self.logger.info(f'patch is not defined for conn {failed_conn_name}')
            self.ui.comboBox_patch.clear()
        else:
            sub_list = [str(x) for x in sub_list]
            # self.disconnect(self.ui.comboBox_patch,
            #                 SIGNAL("currentIndexChanged(QString)"),
            #                 self.updatePatch)
            self.ui.comboBox_patch.clear()
            self.ui.comboBox_patch.addItems(sub_list)
            self.ui.comboBox_patch.currentIndexChanged.connect(
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

    def updateVulnCurve(self):

        try:
            _array = self.results_dict['house']['di']
        except TypeError:
            pass
            #msg = 'Simulation results unavailable'
            #QMessageBox.warning(self, "VAWS Program Warning", msg)
            #self.ui.wb_checkBox.setChecked(False)
            #self.ui.ln_checkBox.setChecked(False)
        else:
            self.ui.mplvuln.axes.figure.clf()
            ax = self.ui.mplvuln.axes.figure.add_subplot(111)

            plot_wind_event_show(ax, self.cfg.no_models,
                                 self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])

            mean_cols = _array.T.mean(axis=0)
            plot_wind_event_mean(ax, self.cfg.wind_speeds, mean_cols)

            # plot the individuals
            for col in _array.T:
                plot_wind_event_damage(ax, self.cfg.wind_speeds, col)

            try:
                param1 = self.results_dict['vulnerability']['weibull']['param1']
                param2 = self.results_dict['vulnerability']['weibull']['param2']
            except KeyError as msg:
                self.logger.warning(msg)
            else:
                self.ui.wb_coeff_1.setText(f'{param1:.3f}')
                self.ui.wb_coeff_2.setText(f'{param2:.3f}')

                if self.ui.wb_checkBox.isChecked():
                    y = vulnerability_weibull(self.cfg.wind_speeds, param1, param2)
                    plot_fitted_curve(ax, self.cfg.wind_speeds, y, col='r',
                                      label="Weibull")

            try:
                param1 = self.results_dict['vulnerability']['lognorm']['param1']
                param2 = self.results_dict['vulnerability']['lognorm']['param2']
            except KeyError as msg:
                self.logger.warning(msg)
            else:
                self.ui.ln_coeff_1.setText(f'{param1:.3f}')
                self.ui.ln_coeff_2.setText(f'{param2:.3f}')

                if self.ui.ln_checkBox.isChecked():
                    y = vulnerability_lognorm(self.cfg.wind_speeds, param1, param2)
                    plot_fitted_curve(ax, self.cfg.wind_speeds, y, col='b',
                                          label="Lognormal")

            ax.legend(loc=2, fancybox=True, shadow=True, fontsize='small')

            self.ui.mplvuln.axes.figure.canvas.draw()

    def convert_h5_results(self, fid):

        self.results_dict = {}

        def h_dic(d, results_dict):
            for k, v in d.items():
                if isinstance(v, h5py._hl.group.Group):
                    results_dict[k] = {}
                    h_dic(v, results_dict[k])
                else:
                    # clean up value
                    results_dict[k] = v.value
        h_dic(fid, self.results_dict)

        try:
            column_names = fid['fragility']['counted'].attrs['column_names'].split(',')
        except KeyError:
            self.logger.warning('Fragility curves are not available')
        else:
            self.results_dict['fragility']['counted'] = pd.DataFrame(self.results_dict['fragility']['counted'], columns=column_names)

        if 'input' in self.results_dict:
            for item in ['no_models', 'model_name', 'random_seed', 'wind_speed_min',
                         'wind_speed_max', 'wind_speed_increment', 'file_wind_profiles',
                         'regional_shielding_factor', 'wind_direction']:
                setattr(self.cfg, item, self.results_dict['input'][item])

            self.cfg.flags.update(self.results_dict['input']['flags'])

            for item in ['vmin', 'vmax', 'vstep']:
                setattr(self.cfg, f'heatmap_{item}',
                        self.results_dict['input'][f'heatmap_{item}'])

            for item in ['states', 'thresholds']:
                _item = f'fragility_{item}'
                _value = self.results_dict['input'][_item].split(',')
                try:
                    _value = [float(x) for x in _value]
                except ValueError:
                    setattr(self.cfg, _item, _value)
                else:
                    setattr(self.cfg, _item, _value)

            if self.cfg.flags['debris']:
                for item in DEBRIS_ITEMS:
                    setattr(self.cfg, item,
                            self.results_dict['input'][item])

            if self.cfg.flags['water_ingress']:
                for item in WATER_INGRESS_ITEMS:
                    _item = f'water_ingress_{item}'
                    _value = self.results_dict['input'][f'water_ingress_{item}'].split(',')
                    try:
                        _value = [float(x) for x in _value]
                    except ValueError:
                        setattr(self.cfg, _item, _value)
                    else:
                        setattr(self.cfg, _item, _value)

            if self.cfg.flags['debris_vulnerability']:
                for item in ['function' , 'param1' , 'param2']:
                    value = self.results_dict['input']['debris_vuln_input'][item]
                    try:
                        value = float(value)
                    except ValueError:
                        self.cfg.debris_vuln_input[item] = value
                    else:
                        self.cfg.debris_vuln_input[item] = value

        return self.results_dict

    def updateFragCurve(self):

        self.ui.mplfrag.axes.figure.clf()
        ax = self.ui.mplfrag.axes.figure.add_subplot(111)

        plot_fragility_show(ax, self.cfg.no_models,
                            self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])

        try:
            df_counted = self.results_dict['fragility']['counted']

        except KeyError:
            self.logger.warning('Fragility curve can not be constructed')

        else:

            for i, (ds, value) in enumerate(self.cfg.fragility.iterrows(), 1):

                ax.plot(self.cfg.wind_speeds,
                        df_counted[f'pe{i}'].values,
                        f"{value['color']}o")

                try:
                    param1 = self.results_dict['fragility']['MLE'][ds]['param1']
                    param2 = self.results_dict['fragility']['MLE'][ds]['param2']
                except KeyError:
                    try:
                        param1 = self.results_dict['fragility']['OLS'][ds]['param1']
                        param2 = self.results_dict['fragility']['OLS'][ds]['param2']
                    except KeyError:
                        self.logger.warning(f'Value of {ds} can not be determined')
                    else:
                        y = vulnerability_lognorm(self.cfg.wind_speeds, param1, param2)

                        plot_fitted_curve(ax, self.cfg.wind_speeds, y,
                                          col=value['color'], label=ds)
                else:
                    y = vulnerability_lognorm(self.cfg.wind_speeds, param1, param2)

                    plot_fitted_curve(ax, self.cfg.wind_speeds, y,
                                      col=value['color'], label=ds)

            ax.legend(loc=2, fancybox=True, shadow=True, fontsize='small')

        self.ui.mplfrag.axes.figure.canvas.draw()

    def updateWaterIngressPlot(self):

        self.statusBar().showMessage('Plotting Water Ingress')

        self.ui.wateringress_plot.axes.figure.clf()
        ax = self.ui.wateringress_plot.axes.figure.add_subplot(111)

        _array = self.results_dict['house']['water_ingress_cost']
        wi_means = _array.mean(axis=1)

        ax.scatter(
            self.cfg.wind_speeds[:, np.newaxis] * np.ones(shape=(1, self.cfg.no_models)),
            _array, c='k', s=8, marker='+', label='_nolegend_')
        ax.plot(self.cfg.wind_speeds, wi_means, c='b', marker='o')

        ax.set_title('Water Ingress By Wind Speed')
        ax.set_xlabel('Impact Wind speed (m/s)')
        ax.set_ylabel('Water Ingress Cost')
        ax.set_xlim(self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])
        ax.set_ylim(0)

        self.ui.wateringress_plot.axes.figure.canvas.draw()

    def updateWaterIngressPropPlot(self):

        self.statusBar().showMessage('Plotting Water Ingress Prop.')

        self.ui.wateringress_prop_plot.axes.figure.clf()
        host = SubplotHost(self.ui.wateringress_prop_plot.axes.figure, 111)

        par1 = ParasiteAxes(host, sharex=host)

        # host
        host.parasites.append(par1)
        host.set_xlabel('Impact Wind speed (m/s)')
        host.set_ylabel('Proportion of houses damaged due to water ingress')
        host.set_xlim(self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])

        # par1
        host.axis["right"].set_visible(False)
        par1.axis["right"].set_visible(True)
        par1.set_ylabel('Water Ingress (%)')
        par1.axis["right"].major_ticklabels.set_visible(True)
        par1.axis["right"].label.set_visible(True)
        par1.set_ylim(0)
        self.ui.wateringress_prop_plot.axes.figure.add_axes(host)

        _array = self.results_dict['house']['water_ingress_perc']
        par1.scatter(
            self.cfg.wind_speeds[:, np.newaxis] * np.ones(shape=(1, self.cfg.no_models)),
            _array, c='k', s=8, marker='+', label='_nolegend_')

        tmp = _array.cumsum(axis=0)
        tmp[tmp > 0] = 1
        host.plot(self.cfg.water_ingress_ref_prop_v, self.cfg.water_ingress_ref_prop, c='r', linestyle='-', label='Target')
        host.plot(self.cfg.wind_speeds, tmp.sum(axis=1)/self.cfg.no_models, c='b', linestyle='-.', label='Simulation')

        host.set_ylim(0)
        host.legend(loc=0, scatterpoints=1)

        self.ui.wateringress_prop_plot.axes.figure.canvas.draw()

    def updateCpiPlot(self, house_number=0):

        self.statusBar().showMessage('Plotting Cpi')
        ax = self.ui.cpi_plot.axes.figure.add_subplot(111)
        ax.clear()

        _array = self.results_dict['house']['cpi']
        _means = _array.mean(axis=1)

        ax.plot(self.cfg.wind_speeds, _means, c='b', marker='o', label='mean')

        if house_number:
            ax.plot(self.cfg.wind_speeds, _array[:, house_number-1],
                                       c='r', marker='+', label=f'{house_number:d}')

        ax.set_title('Internal Pressure Coefficient')
        ax.set_xlabel('Wind speed (m/s)')
        ax.set_ylabel('Cpi')
        ax.legend(loc=0, scatterpoints=1)
        ax.set_xlim(self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])
        ax.set_ylim(0)

        self.ui.cpi_plot.axes.figure.canvas.draw()

    def updateCostPlot(self, house_number=0):

        self.statusBar().showMessage('Plotting repair cost')
        self.ui.cost_plot.axes.figure.clf()

        #ax = self.ui.cost_plot.axes.figure.add_subplot(111)
        #ax.clear()

        host = SubplotHost(self.ui.cost_plot.axes.figure, 111)

        par1 = ParasiteAxes(host, sharex=host)

        host.parasites.append(par1)

        # host
        host.set_ylabel('Repair cost')
        host.set_xlabel('Wind Speed (m/s)')
        host.set_xlim(self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])

        # par1
        host.axis["right"].set_visible(False)
        par1.axis["right"].set_visible(True)
        par1.set_ylabel('Repair cost/replacement cost')
        par1.axis["right"].major_ticklabels.set_visible(True)
        par1.axis["right"].label.set_visible(True)


        self.ui.cost_plot.axes.figure.add_axes(host)

        # mean
        try:
            _dic = self.results_dict['house']['repair_cost_by_scenario']
        except KeyError:
            msg = f'Unable to load repair cost'
            self.statusBar().showMessage(msg)
        else:
            for key, value in _dic.items():
                _value = value.mean(axis=1)
                host.plot(self.cfg.wind_speeds, _value, label=f'{key}:mean')
                par1.plot(self.cfg.wind_speeds, _value/self.cfg.house['replace_cost'], linestyle='')

        if self.cfg.flags['water_ingress']:
            _value = self.results_dict['house']['water_ingress_cost'].mean(axis=1)
            host.plot(self.cfg.wind_speeds, _value, label='Water ingress:mean')
            par1.plot(self.cfg.wind_speeds, _value/self.cfg.house['replace_cost'], linestyle='')

        # house instance
        if house_number:
            try:
                _dic = self.results_dict['house']['repair_cost_by_scenario']
            except KeyError:
                msg = f'Unable to load repair cost'
                self.statusBar().showMessage(msg)
            else:
                for key, value in _dic.items():
                    _value = value[:, house_number-1]
                    host.plot(self.cfg.wind_speeds, _value, label=f'{key}:{house_number:d}', linestyle='dashed')
                    par1.plot(self.cfg.wind_speeds, _value/self.cfg.house['replace_cost'], linestyle='')

            if self.cfg.flags['water_ingress']:
                _value = self.results_dict['house']['water_ingress_cost'][:, house_number-1]
                host.plot(self.cfg.wind_speeds, _value, label=f'Water ingress:{house_number:d}', linestyle='dashed')
                par1.plot(self.cfg.wind_speeds, _value/self.cfg.house['replace_cost'], linestyle='')


        host.legend(loc=2, scatterpoints=1)

        self.ui.cost_plot.axes.figure.canvas.draw()

    def updateBreachPlot(self):

        self.statusBar().showMessage('Plotting Debris Results')
        self.ui.breaches_plot.axes.figure.clf()
        # we'll have three seperate y axis running at different scales

        breaches = self.results_dict['house']['window_breached'].sum(axis=1) / self.cfg.no_models
        nv_means = self.results_dict['house']['no_debris_impacts'].mean(axis=1)
        num_means = self.results_dict['house']['no_debris_items'].mean(axis=1)

        host = SubplotHost(self.ui.breaches_plot.axes.figure, 111)

        par1 = ParasiteAxes(host, sharex=host)
        par2 = ParasiteAxes(host, sharex=host)

        host.parasites.append(par1)
        host.parasites.append(par2)

        # host
        host.set_ylabel('Perc Breached')
        host.set_xlabel('Wind Speed (m/s)')
        host.set_xlim(self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])

        # par1
        host.axis["right"].set_visible(False)
        par1.axis["right"].set_visible(True)
        par1.set_ylabel('Impacts')

        par1.axis["right"].major_ticklabels.set_visible(True)
        par1.axis["right"].label.set_visible(True)

        #  par2
        par2.axis['right'].set_visible(False)
        offset = 60, 0
        new_axisline = par2.get_grid_helper().new_fixed_axis
        par2.axis['right2'] = new_axisline(loc='right', axes=par2, offset=offset)
        par2.axis['right2'].label.set_visible(True)
        par2.axis['right2'].set_label('Supply')

        self.ui.breaches_plot.axes.figure.add_axes(host)
        self.ui.breaches_plot.axes.figure.subplots_adjust(right=0.75)

        p1, = host.plot(self.cfg.wind_speeds, breaches, label='Breached', c='b')
        p2, = par1.plot(self.cfg.wind_speeds, nv_means, label='Impacts', c='g')
        _, = par1.plot(self.cfg.wind_speeds, nv_means.cumsum(), label='_nolegend_', c='g', ls='--')
        p3, = par2.plot(self.cfg.wind_speeds, num_means, label='Supply', c='r')
        _, = par2.plot(self.cfg.wind_speeds, num_means.cumsum(), label='_nolegend_', c='r', ls='--')

        host.axis['left'].label.set_color(p1.get_color())
        par1.axis['right'].label.set_color(p2.get_color())
        par2.axis['right2'].label.set_color(p3.get_color())

        host.legend(loc=2)
        self.ui.breaches_plot.axes.figure.canvas.draw()

    def updateStrengthPlot(self):

        self.ui.connection_type_plot.axes.figure.clf()
        ax = self.ui.connection_type_plot.axes.figure.add_subplot(111)

        xlabels = []
        for _id, (type_name, grouped) in enumerate(self.cfg.connections.groupby('type_name')):
            try:
                _array = np.array([self.results_dict['connection']['strength'][i] for i in grouped.index]).flatten()
            except KeyError:
                _array = np.array([self.results_dict['connection']['strength'][str(i)] for i in grouped.index]).flatten()
            ax.scatter(
                _id * np.ones_like(_array), _array, s=8, marker='+')
            # ax.hold(True)
            ax.scatter(_id, _array.mean(), s=20,
                                                      c='r', marker='o')
            xlabels.append(type_name)

        ax.set_xticks(range(_id + 1))
        ax.set_xticklabels(xlabels, rotation='vertical')
        ax.set_title('Connection Type Strengths')
        ax.set_position([0.05, 0.20, 0.9, 0.75])
        ax.figure.canvas.draw()
        ax.set_xlim((-0.5, len(xlabels)))

        self.ui.connection_type_plot.figure.canvas.draw()

    def updateTypeDamagePlot(self):

        self.ui.connection_type_damages_plot.axes.figure.clf()
        ax = self.ui.connection_type_damages_plot.axes.figure.add_subplot(111)

        xlabels = []
        for _id, (type_name, grouped) in enumerate(self.cfg.connections.groupby('type_name')):
            try:
                _array = np.array(
                [self.results_dict['connection']['capacity'][i] for i in grouped.index]).flatten()
            except KeyError:
                _array = np.array(
                [self.results_dict['connection']['capacity'][str(i)] for i in grouped.index]).flatten()

            _array[_array == -1] = np.nan
            ax.scatter(_id * np.ones_like(_array), _array, s=8, marker='+')
            ax.scatter(_id, np.nanmean(_array), s=20, c='r', marker='o')
            xlabels.append(type_name)

        ax.set_xticks(range(_id + 1))
        ax.set_xticklabels(xlabels, rotation='vertical')
        ax.set_title('Connection Type Damage Speeds')
        ax.set_ylabel('Wind speed (m/s)')
        ax.set_position([0.05, 0.20, 0.9, 0.75])
        ax.set_xlim((-0.5, len(xlabels)))
        ax.figure.canvas.draw()

    def updateLoadPlot(self):

        self.statusBar().showMessage('updating connection load')

        group_name = str(self.ui.comboBox_load.currentText())
        house_number = self.ui.spinBox_load.value()
        ispeed = np.argmin(np.abs(self.cfg.wind_speeds-self.ui.doubleSpinBox_load.value()))

        grouped = self.cfg.connections.loc[
            self.cfg.connections.group_name == group_name]

        #load = np.array([self.results_dict['connection']['load'][i]
        #                 for i in grouped.index])[:, ispeed, house_number-1]

        # maintain vmin, vmax for all speed and house
        try:
            tmp = np.array([self.results_dict['connection']['load'][i]
                         for i in grouped.index])
        except KeyError:
            tmp = np.array([self.results_dict['connection']['load'][str(i)]
                         for i in grouped.index])

        _min, _max = tmp.min(), tmp.max()
        v_min = _min - np.sign(_min) * 0.05 * _min
        v_max = _max + np.sign(_max) * 0.05 * _max

        load = tmp[:, ispeed, house_number-1]
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
            try:
                failure_count = np.count_nonzero(connections_damaged[conn_id])
            except KeyError:
                failure_count = np.count_nonzero(connections_damaged[str(conn_id)])

            try:
                failure_mean = np.mean(connections_damaged[conn_id])
            except KeyError:
                failure_mean = np.mean(connections_damaged[str(conn_id)])
            self.ui.connections.setItem(irow, 4, QTableWidgetItem(f'{failure_mean:.3}'))
            self.ui.connections.setItem(irow, 5, QTableWidgetItem(f'{failure_count}'))

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
        # construction_level = [self.results_dict['house']['construction_level'][i]
        #                       for i in range(self.cfg.no_models)]

        for i in range(self.cfg.no_models):
            parent = QTreeWidgetItem(self.ui.zoneResults,
                                     [f'#{i+1} {WIND_DIR[wind_dir[i]]}', '', '', ''])
            for zr_key in self.cfg.zones:
                QTreeWidgetItem(parent, ['',
                                         zr_key,
                                         f"{self.results_dict['zone']['cpe'][zr_key][0, i]:.3}",
                                         f"{self.results_dict['zone']['cpe_str'][zr_key][0, i]:.3}",
                                         f"{self.results_dict['zone']['cpe_eave'][zr_key][0, i]:.3}"])

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
                                     [f'#{i+1} {WIND_DIR[wind_dir[i]]}', '', '', ''])

            for _id, value in self.cfg.connections.iterrows():

                try:
                    capacity = self.results_dict['connection']['capacity'][_id][0, i]
                except KeyError:
                    capacity = self.results_dict['connection']['capacity'][str(_id)][0, i]

                # load = self.results_dict['connection']['load'][_id][house_num]
                try:
                    dead_load = self.results_dict['connection']['dead_load'][_id][0, i]
                except KeyError:
                    dead_load = self.results_dict['connection']['dead_load'][str(_id)][0, i]

                try:
                    strength = self.results_dict['connection']['strength'][_id][0, i]
                except KeyError:
                    strength = self.results_dict['connection']['strength'][str(_id)][0, i]

                try:
                    load_last = self.results_dict['connection']['load'][_id][-1, i]
                except KeyError:
                    load_last = self.results_dict['connection']['load'][str(_id)][-1, i]

                if load_last:
                    load_desc = f'{load_last:.3}'
                else:
                    load_desc = 'collapse'

                conn_parent = QTreeWidgetItem(parent, [f"{value['type_name']}_{_id}",
                                                       '{:.3}'.format(capacity if capacity > 0 else 'NA'),
                                                       f'{strength:.3}',
                                                       f'{dead_load:.3}',
                                                       f'{load_desc}'])

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
            scenario_path = settings.value("ScenarioFolder")

        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Scenarios",
                                                  scenario_path,
                                                  "Scenarios (*.cfg)")
        if not filename:
            return

        config_file = '{}'.format(filename)
        if os.path.isfile(config_file):
            self.file_load(config_file)
            settings.setValue("ScenarioFolder", QVariant(os.path.dirname(config_file)))
        else:
            msg = f'Unable to load scenario: {config_file}\nFile not found.'
            QMessageBox.warning(self, "VAWS Program Warning", msg)

    def load_results(self, config_file=None):
        settings = QSettings()
        # Set the path to the default
        scenario_path = SCENARIOS_DIR
        if config_file:
            scenario_path = config_file
        elif settings.contains("ScenarioFolder"):
            # we have a saved scenario path so use that
            scenario_path = settings.value("ScenarioFolder")

        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Scenarios",
                                                  scenario_path,
                                                  "Scenarios (*.cfg)")
        if not filename:
            return

        config_file = '{}'.format(filename)
        h5_file = os.path.join(os.path.dirname(config_file), OUTPUT_DIR, FILE_RESULTS)
        if os.path.exists(h5_file):
            self.file_load(config_file)
            #settings.setValue("ScenarioFolder", QVariant(os.path.dirname(config_file)))


            self.ui.spinBox_heatmap.setRange(0, self.cfg.no_models)

            self.ui.doubleSpinBox_load.setRange(self.cfg.wind_speeds[0], self.cfg.wind_speeds[-1])
            self.ui.spinBox_load.setRange(1, self.cfg.no_models)

            msg = f'Load previous results'
            self.has_run = True
            self.statusBar().showMessage(msg)
            with h5py.File(h5_file, 'r') as fid:

                self.results_dict = self.convert_h5_results(fid)

                try:
                    assert self.cfg.no_models == self.results_dict['house']['di'].shape[1]
                    assert np.array_equal(self.cfg.wind_speeds, self.results_dict['wind_speeds'])
                except AssertionError:
                    self.logger.critical('Saved results are not compatible with the input file')
                else:
                    self.updateVulnCurve()
                    self.updateFragCurve()
                    self.updateHouseResultsTable()
                    self.updateConnectionTable_with_results()
                    self.updateConnectionTypePlots()
                    self.updateHeatmap()
                    self.updateWaterIngressPlot()
                    self.updateWaterIngressPropPlot()
                    self.updateBreachPlot()
                    self.updateCpiPlot()
                    self.updateCostPlot()

        else:
            msg = f'Unable to load resutls: {config_file}\nFile not found.'
            QMessageBox.warning(self, "VAWS Program Warning", msg)

    def save_scenario(self):
        ok = self.update_config_from_ui()
        if ok:
            self.cfg.save_config()
            self.ui.statusbar.showMessage(f'Saved to file {self.cfg.file_cfg}')
        # self.update_ui_from_config()

    def save_vuln_file(self):

        try:
            _array = (self.results_dict['house']['di']).T.mean(axis=0)
        except TypeError:
            msg = 'Simultation results unavailable'
            QMessageBox.warning(self, "VAWS Program Warning", msg)
        else:
            fname, _ = QFileDialog.getSaveFileName(self, "VAWS - Save mean vul curve",
                                               self.cfg.path_output, VUL_OUTPUT_TEMPL)
            if len(fname) > 0:
                if "." not in fname:
                    fname += ".csv"
                combined = np.array([self.cfg.wind_speeds, _array]).T
                np.savetxt(fname, combined, delimiter=',', header='wind_speed,mean_di')
            else:
                msg = 'No file name entered. Action cancelled'
                QMessageBox.warning(self, "VAWS Program Warning", msg)

    def save_as_scenario(self):
        # TODO: check
        ok = self.update_config_from_ui()
        if ok:
            current_parent, _ = os.path.split(self.cfg.path_cfg)

            fname, _ = QFileDialog.getSaveFileName(self, "VAWS - Save Scenario",
                                                current_parent, CONFIG_TEMPL)
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
                settings.setValue("ScenarioFolder", QVariant(path_cfg))
                self.cfg.path_cfg = path_cfg
                self.cfg.file_cfg = os.path.join(path_cfg, file_suffix_name)
                self.cfg.output_path = os.path.join(path_cfg, OUTPUT_DIR)

                self.cfg.save_config()
                set_logger(self.cfg)
                self.update_ui_from_config()
            else:
                msg = 'No scenario name entered. Action cancelled'
                QMessageBox.warning(self, "VAWS Program Warning", msg)

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
                filename = QVariant(self.cfg.file_cfg)
                settings.setValue("LastFile", filename)
            self.storeSizePosToSettings()
        else:
            event.ignore()

    def update_ui_from_config(self):
        if self.cfg:
            self.statusBar().showMessage('Updating', 1000)

            # Scenario
            self.ui.numHouses.setText(f'{self.cfg.no_models:d}')
            self.ui.houseName.setText(self.cfg.model_name)
            self.ui.seedRandom.setText(str(self.cfg.random_seed))

            idx = self.ui.terrainCategory.findText(self.cfg.file_wind_profiles)
            if self.ui.terrainCategory.currentIndex() == idx:
                self.updateTerrainCategoryTable()
            else:
                self.ui.terrainCategory.setCurrentIndex(idx)

            self.ui.regionalShielding.setText(f'{self.cfg.regional_shielding_factor:.1f}')
            self.ui.windMin.setValue(self.cfg.wind_speed_min)
            self.ui.windMax.setValue(self.cfg.wind_speed_max)
            self.ui.windIncrement.setText(f'{self.cfg.wind_speed_increment:.3f}')
            self.ui.windDirection.setCurrentIndex(self.cfg.wind_dir_index)

            # Debris
            idx = self.ui.debrisRegion.findText(self.cfg.region_name)
            if self.ui.debrisRegion.currentIndex() == idx:
                self.updateDebrisRegionsTable()
            else:
                self.ui.debrisRegion.setCurrentIndex(idx)

            self.ui.buildingSpacing.setCurrentIndex(
                self.ui.buildingSpacing.findText(
                    f'{self.cfg.building_spacing:.1f}'))
            self.ui.debrisRadius.setValue(self.cfg.debris_radius)
            self.ui.debrisAngle.setValue(self.cfg.debris_angle)
            self.ui.sourceItems.setValue(self.cfg.source_items)
            self.ui.debrisBoundary.setText(
                f'{self.cfg.boundary_radius:.3f}')
            self.ui.staggeredDebrisSources.setChecked(self.cfg.staggered_sources)
            self.ui.debris.setChecked(self.cfg.flags['debris'])

            self.ui.checkBox_debrisVul.setChecked(self.cfg.flags['debris_vulnerability'])
            if self.cfg.flags['debris_vulnerability']:
                idx = self.ui.comboBox_debrisVul.findText(
                    self.cfg.debris_vuln_input['function'].capitalize())
                if self.ui.comboBox_debrisVul.currentIndex() == idx:
                    self.updateDebrisVuln()
                else:
                    self.ui.comboBox_debrisVul.setCurrentIndex(idx)

                self.ui.debrisVul_param1.setText(
                    f"{self.cfg.debris_vuln_input['param1']}")
                self.ui.debrisVul_param2.setText(
                    f"{self.cfg.debris_vuln_input['param2']}")

            # water ingress
            self.ui.waterThresholds.setText(
                ', '.join([str(x) for x in self.cfg.water_ingress_thresholds]))
            self.ui.waterSpeed0.setText(
                ', '.join([str(x) for x in self.cfg.water_ingress_speed_at_zero_wi]))
            self.ui.waterSpeed1.setText(
                ', '.join([str(x) for x in self.cfg.water_ingress_speed_at_full_wi]))
            self.ui.ref_prop_v.setText(
                ', '.join([str(x) for x in self.cfg.water_ingress_ref_prop_v]))
            self.ui.ref_prop.setText(
                ', '.join([str(x) for x in self.cfg.water_ingress_ref_prop]))
            self.ui.di_threshold_wi.setText(f'{self.cfg.water_ingress_di_threshold_wi:f}')
            self.ui.waterEnabled.setChecked(self.cfg.flags['water_ingress'])

            # wall collapse
            if self.cfg.wall_collapse:
                self.ui.typeName.setText(
                    ', '.join([str(x) for x in self.cfg.wall_collapse['type_name']]))
                self.ui.roofDamage.setText(
                    ', '.join([str(x) for x in self.cfg.wall_collapse['roof_damage']]))
                self.ui.wallDamage.setText(
                    ', '.join([str(x) for x in self.cfg.wall_collapse['wall_damage']]))
                self.ui.wallCollapseEnabled.setChecked(self.cfg.flags['wall_collapse'])

            # options
            self.ui.fragilityStates.setText(
                ', '.join(self.cfg.fragility_states))
            self.ui.fragilityThresholds.setText(
                ', '.join([str(x) for x in self.cfg.fragility_thresholds]))
            self.ui.diffShielding.setChecked(self.cfg.flags['differential_shielding'])

            self.ui.redV.setValue(min(self.cfg.wind_speeds[0], self.cfg.heatmap_vmin))
            self.ui.blueV.setValue(max(self.cfg.wind_speeds[-1], self.cfg.heatmap_vmax))
            self.ui.vStep.setValue(self.cfg.heatmap_vstep)

            self.update_house_panel()
            # self.updateGlobalData()

            self.ui.actionRun.setEnabled(True)

            if self.cfg.file_cfg:
                self.statusBarScenarioLabel.setText(
                   f'Scenario: {os.path.basename(self.cfg.file_cfg)}')
        else:
            self.statusBarScenarioLabel.setText('Scenario: None')

    def update_config_from_ui(self):
        new_cfg = self.cfg
        ok = True

        # Scenario section
        new_cfg.no_models = int(self.ui.numHouses.text())
        new_cfg.model_name = self.ui.houseName.text()
        new_cfg.file_wind_profiles = self.ui.terrainCategory.currentText()
        new_cfg.regional_shielding_factor = float(self.ui.regionalShielding.text())
        new_cfg.wind_speed_min = self.ui.windMin.value()
        new_cfg.wind_speed_max = self.ui.windMax.value()
        new_cfg.wind_speed_increment = float(self.ui.windIncrement.text())
        # new_cfg.set_wind_speeds()
        new_cfg.wind_direction = WIND_DIR[
            self.ui.windDirection.currentIndex()]

        # Debris section
        new_cfg.flags['debris'] = self.ui.debris.isChecked()
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

        if self.ui.checkBox_debrisVul.isChecked():
            new_cfg.flags['debris_vulnerability'] = True
            new_cfg.debris_vuln_input['function'] = self.ui.comboBox_debrisVul.currentText()
            msg = 'Invalid vulnerability parameter value(s)'
            try:
                new_cfg.debris_vuln_input['param1'] = float(self.ui.debrisVul_param1.text())
            except ValueError:
                QMessageBox.warning(self, 'VAWS Program Warning', msg)
                ok = False
            else:
                try:
                    new_cfg.debris_vuln_input['param2'] = float(self.ui.debrisVul_param2.text())
                except ValueError:
                    QMessageBox.warning(self, 'VAWS Program Warning', msg)
                    ok = False
        else:
            new_cfg.flags['debris_vulnerability'] = False
            new_cfg.debris_vuln_input = {}

        # options 
        new_cfg.flags['differential_shielding'] = self.ui.diffShielding.isChecked()

        # option section
        new_cfg.random_seed = int(self.ui.seedRandom.text())
        new_cfg.heatmap_vmin = float(self.ui.redV.value())
        new_cfg.heatmap_vmax = float(self.ui.blueV.value())
        new_cfg.heatmap_vstep = float(self.ui.vStep.value())

        # fragility section
        new_cfg.fragility_thresholds = [
            float(x) for x in self.ui.fragilityThresholds.text().split(',')]
        new_cfg.fragility_states = [
            x.strip() for x in self.ui.fragilityStates.text().split(',')]

        # water ingress
        new_cfg.flags['water_ingress'] = self.ui.waterEnabled.isChecked()
        new_cfg.water_ingress_thresholds = [
            float(x) for x in self.ui.waterThresholds.text().split(',')]
        new_cfg.water_ingress_speed_at_zero_wi = [
            float(x) for x in self.ui.waterSpeed0.text().split(',')]
        new_cfg.water_ingress_speed_at_full_wi = [
            float(x) for x in self.ui.waterSpeed1.text().split(',')]
        new_cfg.water_ingress_ref_prop_v = [
            float(x) for x in self.ui.ref_prop_v.text().split(',')]
        new_cfg.water_ingress_ref_prop = [
            float(x) for x in self.ui.ref_prop.text().split(',')]
        new_cfg.water_ingress_di_threshold_wi = float(self.ui.di_threshold_wi.text())

        # wall collapse
        new_cfg.flags['wall_collapse'] = self.ui.wallCollapseEnabled.isChecked()
        new_cfg.wall_collapse['type_name'] = [
            x.strip() for x in self.ui.typeName.text().split(',')]
        new_cfg.wall_collapse['roof_damage'] = [
            float(x) for x in self.ui.roofDamage.text().split(',')]
        new_cfg.wall_collapse['wall_damage'] = [
            float(x) for x in self.ui.wallDamage.text().split(',')]

        # house / groups section
        # for irow, (index, ctg) in enumerate(self.cfg.groups.items()):
        #     cellWidget = self.ui.connGroups.cellWidget(irow, 4)
        #     new_cfg.flags['conn_type_group_{}'.format(index)] = True \
        #         if cellWidget.checkState() == Qt.Checked else False
        return ok

    def file_load(self, fname):
        try:
            path_cfg = os.path.dirname(os.path.realpath(fname))
            set_logger(path_cfg)
            self.cfg = Config(fname)
            self.init_terrain_category()
            self.init_debris_region()
            self.init_debrisvuln()
            self.init_pressure()
            self.init_influence_and_patch()
            self.update_ui_from_config()
            self.statusBar().showMessage('Ready')
        except Exception as excep:
            self.logger.exception("Loading configuration caused exception")

            msg = 'Unable to load previous scenario: %s\nLoad cancelled.' % fname
            QMessageBox.warning(self, "VAWS Program Warning", msg)

    def okToContinue(self):
        ok = self.update_config_from_ui()
        if ok and self.dirty_scenario:
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

    def testDebrisRun(self, wind_speed, vuln_input):

        rnd_state = np.random.RandomState(1)
        incr_speed = self.cfg.wind_speeds[1] - self.cfg.wind_speeds[0]

        if self.ui.comboBox_debrisVul.currentText() == 'Weibull':
            damage_incr = vulnerability_weibull_pdf(
                x=wind_speed, alpha=vuln_input['param1'], beta=vuln_input['param2']) * incr_speed
        else:
            damage_incr = vulnerability_lognorm_pdf(
                x=wind_speed, med=vuln_input['param1'], std=vuln_input['param2']) * incr_speed

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
        #_array = np.array(house.debris.footprint.exterior.xy).T
        #ax.add_patch(patches.Polygon(_array, alpha=0.5))
        #ax.add_patch(patches.Polygon(self.cfg.impact_boundary.exterior, alpha=0.5))

        for item in debris_items:
            _x, _y = item.trajectory.xy[0][1], item.trajectory.xy[1][1]
            ax.scatter(_x, _y, color=DEBRIS_TYPES_CLRS[item.type], alpha=0.2)

        for source in self.cfg.debris_sources:
            ax.scatter(source.x, source.y, color='b', label='source')
        ax.scatter(0, 0, label='target', color='r')

        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc=2, scatterpoints=1)

        title_str = f'Debris samples at {wind_speed:.3f} m/s in region of {self.cfg.region_name}'
        ax.set_title(title_str)

        ax.axes.set_xlim(-0.5*self.cfg.debris_radius, self.cfg.debris_radius)
        ax.axes.set_ylim(-1.0*self.cfg.debris_radius, self.cfg.debris_radius)
        # fig.canvas.draw()
        fig.show()


    def testDebrisSettings(self):
        try:
            vuln_input = {'param1': float(self.ui.debrisVul_param1.text()),
                          'param2': float(self.ui.debrisVul_param2.text())}
        except ValueError:
            msg = 'Invalid vulnerability parameter value(s)'
            QMessageBox.warning(self, 'VAWS Program Warning', msg)
        else:
            ok = self.update_config_from_ui()
            if ok:
                self.cfg.process_config()
                wind_speed, ok = QInputDialog.getInt(
                    self, "Debris Test", "Wind speed (m/s):", 50, 10, 200)
                if ok:
                    self.testDebrisRun(wind_speed, vuln_input)

    def testWaterIngress(self):
        new_cfg = self.cfg
        new_cfg.flags['water_ingress'] = self.ui.waterEnabled.isChecked()
        new_cfg.water_ingress_thresholds = [
            float(x) for x in self.ui.waterThresholds.text().split(',')]
        new_cfg.water_ingress_speed_at_zero_wi = [
            float(x) for x in self.ui.waterSpeed0.text().split(',')]
        new_cfg.water_ingress_speed_at_full_wi = [
            float(x) for x in self.ui.waterSpeed1.text().split(',')]
        new_cfg.water_ingress_ref_prop_v = [
            float(x) for x in self.ui.ref_prop_v.text().split(',')]
        new_cfg.water_ingress_ref_prop = [
            float(x) for x in self.ui.ref_prop.text().split(',')]
        #new_cfg.water_ingress_di_threshold_wi = float(self.ui.di_threshold_wi.text())

        new_cfg.wind_speed_min = min(WIND_4_TEST_WATER_INGRESS[0], self.ui.windMin.value())
        new_cfg.wind_speed_max = max(WIND_4_TEST_WATER_INGRESS[1], self.ui.windMax.value())
        new_cfg.wind_speed_increment = float(self.ui.windIncrement.text())
        new_cfg.set_water_ingress()
        new_cfg.set_wind_speeds()

        di_array = []
        dic_thresholds = {}
        for i, value in enumerate(new_cfg.water_ingress.index):
            if i == 0:
                dic_thresholds[i] = (0.0, value)
            else:
                dic_thresholds[i] = (new_cfg.water_ingress.index[i-1], value)

            di_array.append(0.5*(dic_thresholds[i][0] + dic_thresholds[i][1]))

        a = np.zeros((len(di_array), len(new_cfg.wind_speeds)))
        for i, di in enumerate(di_array):
            for j, speed in enumerate(new_cfg.wind_speeds):
                a[i, j] = 100.0 * compute_water_ingress_given_damage(
                    di, speed, new_cfg.water_ingress)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for j in range(a.shape[0]):
            ax.plot(new_cfg.wind_speeds, a[j, :],
                    label='{:.1f} <= DI < {:.1f}'.format(*dic_thresholds[j]))

        ax.legend(loc=1)
        ax.set_xlabel('Wind speed (m/s)')
        ax.set_ylabel('Water ingress (%)')
        fig.show()


def run_gui():
    parser = process_commandline()

    args = parser.parse_args()

    if not args.config_file:
        initial_scenario = DEFAULT_SCENARIO
    else:
        initial_scenario = args.config_file

    path_cfg = os.path.dirname(os.path.realpath(initial_scenario))
    if args.verbose:
        set_logger(path_cfg, logging_level=args.verbose)
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
