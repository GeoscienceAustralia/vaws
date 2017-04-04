#!/usr/bin/env python
# adjust python path so we may import things from peer packages
import sys
import shutil
import time
import os.path
import logging

from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
from PyQt4.QtCore import SIGNAL, QTimer, Qt, QSettings, QVariant, QString, QFile
from PyQt4.QtGui import QProgressBar, QLabel, QMainWindow, QApplication, QTableWidget, QPixmap,\
                        QTableWidgetItem, QDialog, QCheckBox, QFileDialog, QIntValidator,\
                        QDoubleValidator, QMessageBox, QTreeWidgetItem, QInputDialog, QSplashScreen
import numpy
import pandas as pd
from vaws.curve import vulnerability_lognorm, vulnerability_weibull

from main_ui import Ui_main
from vaws.simulation import process_commandline, set_logger, \
    simulate_wind_damage_to_houses
from vaws.config import Config, INPUT_DIR
from vaws.version import VERSION_DESC
from vaws.stats import compute_logarithmic_mean_stddev

from mixins import PersistSizePosMixin, setupTable, finiTable

my_app = None

SOURCE_DIR = os.path.dirname(__file__)
VAWS_DIR = os.sep.join(SOURCE_DIR.split(os.sep)[:-2])
SCENARIOS_DIR = os.path.join(VAWS_DIR, 'scenarios')
CONFIG_TEMPL = "Scenarios (*.cfg)"
DEFAULT_SCENARIO = os.path.join(SCENARIOS_DIR,'default/default.cfg')

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
        
        self.ui = Ui_main()
        self.ui.setupUi(self)

        windowTitle = VERSION_DESC
        self.setWindowTitle(unicode(windowTitle))

        self.cfg = init_scenario

        # scenario section
        self.ui.numHouses.setValidator(QIntValidator(1, 10000, self.ui.numHouses))
        self.ui.houseName.addItems(self.cfg.df_house['name'].values)
        self.ui.houseName.setCurrentIndex(-1)
        self.ui.terrainCategory.addItems(self.cfg.terrain_categories)
        self.ui.terrainCategory.setCurrentIndex(-1)
        self.ui.windDirection.addItems(self.cfg.wind_dir)

        # debris
        self.ui.debrisRegion.addItems(self.cfg.region_names)
        self.ui.debrisRegion.setCurrentIndex(-1)
        self.ui.buildingSpacing.addItems(('20.0', '40.0'))
        self.ui.flighttimeMean.setValidator(
            QDoubleValidator(0.0, 100.0, 3, self.ui.flighttimeMean))
        self.ui.flighttimeStddev.setValidator(QDoubleValidator(
            0.0, 100.0, 3, self.ui.flighttimeStddev))
        # self.ui.debrisExtension.setValidator(QDoubleValidator(
        #     0.0, 100.0, 3, self.ui.debrisExtension))
      
        self.statusProgressBar = QProgressBar()
        self.statusProgressBar.setMinimum(0)
        self.statusProgressBar.setMaximum(100)
        self.statusBar().addPermanentWidget(self.statusProgressBar)
        self.statusProgressBar.hide()        
        self.statusBarScenarioLabel = QLabel()
        self.statusBarScenarioLabel.setText('Scenario: None')
        self.statusBar().addPermanentWidget(self.statusBarScenarioLabel)
        
        self.dirty_scenario = False         # means scenario has changed
        self.dirty_conntypes = False        # means connection_types have been modified
        self.has_run = False
        self.initSizePosFromSettings()

        # top panel
        self.connect(self.ui.actionOpen_Scenario, SIGNAL("triggered()"), self.open_scenario)
        self.connect(self.ui.actionRun, SIGNAL("triggered()"), self.runScenario)
        self.connect(self.ui.actionStop, SIGNAL("triggered()"), self.stopScenario)
        self.connect(self.ui.actionSave_Scenario, SIGNAL("triggered()"), self.save_scenario)
        self.connect(self.ui.actionSave_Scenario_As, SIGNAL("triggered()"), self.save_as_scenario)
        self.connect(self.ui.actionHouse_Info, SIGNAL("triggered()"), self.showHouseInfoDlg)

        # test panel
        self.connect(self.ui.testDebrisButton, SIGNAL("clicked()"), self.testDebrisSettings)
        self.connect(self.ui.testConstructionButton, SIGNAL("clicked()"), self.testConstructionLevels)

        # Scenario panel
        self.connect(self.ui.terrainCategory, SIGNAL("currentIndexChanged(QString)"), self.updateTerrainCategoryTable)
        self.connect(self.ui.windMin, SIGNAL("valueChanged(int)"), lambda x: self.onSliderChanged(self.ui.windMinLabel, x))
        self.connect(self.ui.windMax, SIGNAL("valueChanged(int)"), lambda x: self.onSliderChanged(self.ui.windMaxLabel, x))
        self.connect(self.ui.windSteps, SIGNAL("valueChanged(int)"), lambda x: self.onSliderChanged(self.ui.windStepsLabel, x))

        # debris panel
        self.connect(self.ui.debrisRegion, SIGNAL("currentIndexChanged(QString)"), self.updateDebrisRegionsTable)
        self.connect(self.ui.debrisRadius, SIGNAL("valueChanged(int)"), lambda x: self.onSliderChanged(self.ui.debrisRadiusLabel, x))
        self.connect(self.ui.debrisAngle, SIGNAL("valueChanged(int)"), lambda x: self.onSliderChanged(self.ui.debrisAngleLabel, x))
        self.connect(self.ui.sourceItems, SIGNAL("valueChanged(int)"), lambda x: self.onSliderChanged(self.ui.sourceItemsLabel, x))

        # options
        self.connect(self.ui.redV, SIGNAL("valueChanged(int)"), lambda x: self.onSliderChanged(self.ui.redVLabel, x))
        self.connect(self.ui.blueV, SIGNAL("valueChanged(int)"), lambda x: self.onSliderChanged(self.ui.blueVLabel, x))
        self.connect(self.ui.applyDisplayChangesButton, SIGNAL("clicked()"), self.updateDisplaySettings)

        # plot panel
        self.connect(self.ui.fitLognormalCurve, SIGNAL("clicked()"), self.updateVulnCurveSettings)

        self.statusBar().showMessage('Loading')
        self.house_results = []
        self.stopTriggered = False
        self.selected_zone = None
        self.selected_conn = None
        self.selected_plotKey = None

        self.updateGlobalData()
        self.ui.sourceItems.setValue(-1)

        QTimer.singleShot(0, self.set_scenario)

    def onZoneSelected(self, z, plotKey):
        self.selected_zone = z
        self.selected_plotKey = plotKey

    def onSelectConnection(self, connection_name):
        for irow in range(len(self.cfg.house.connections)):
            if unicode(self.ui.connections.item(irow, 0).text()) == connection_name:
                self.ui.connections.setCurrentCell(irow, 0)
                break
        
    def onSelectZone(self, zoneLoc):
        for irow in range(len(self.cfg.house.zones)):
            if unicode(self.ui.zones.item(irow, 0).text()) == zoneLoc:
                self.ui.zones.setCurrentCell(irow, 0)
                break

    def onConnectionsDoubleClicked(self, row, col):
        connection_name = unicode(self.ui.connections.item(row, 0).text())
        self.selected_conn = house.connByNameMap[connection_name]
        self.showConnectionViewer() 
        
    def onSliderChanged(self, label, x):
        label.setText('{:d}'.format(x))
        
    def updateVulnCurveSettings(self):
        if self.has_run:
            self.update_config_from_ui()
            self.updateVulnCurve()
            self.statusBar().showMessage(unicode('Curve Fit Updated'))
            
    def updateVulnCurve(self):
        # self.show_results(None, self.ui.redV.value(), self.ui.blueV.value())

        df_dmg_idx = pd.read_hdf(self.cfg.file_house, 'house')['di']
        df_fitted_curves = pd.read_csv(self.cfg.file_curve,
                                       names=['key', 'error', 'param1',
                                              'param2'], skiprows=1,
                                       index_col='key')

        self.ui.mplfrag.axes.hold(True)
        self.ui.mplfrag.axes.cla()
        self.ui.mplfrag.axes.figure.canvas.draw()
        for _, col in df_dmg_idx.iteritems():
            self.ui.mplfrag.axes.plot(self.cfg.speeds, col.values, 'o', 0.3)

        for ds, value in self.cfg.fragility_thresholds.iterrows():
            try:
                y = vulnerability_lognorm(self.cfg.speeds,
                                          df_fitted_curves.at[ds, 'param1'],
                                          df_fitted_curves.at[ds, 'param2'])
            except KeyError as msg:
                logging.warning(msg)
            else:
                self.ui.mplfrag.axes.plot(self.cfg.speeds, y,
                                          color=value['color'],
                                          linestyle='-', label=ds)
        self.ui.mplfrag.axes.legend()

        # self.ui.mplvuln.axes.cla()
        # self.ui.mplvuln.axes.figure.canvas.draw()
        # for _, col in df_dmg_idx.iteritems():
        #     self.ui.mplvuln.axes.plot(self.cfg.speeds, col.values, '-', 0.3)

        # self.ui.mplvuln.axes.plot(self.cfg.speeds,result[0][''])
        # self.ui.sumOfSquares.setText('%f' % self.simulator.ss)
        # self.ui.coeff_1.setText('%f' % self.simulator.A_final[0])
        # self.ui.coeff_2.setText('%f' % self.simulator.A_final[1])

    def updateDisplaySettings(self):
        if self.has_run:
            self.simulator.plot_connection_damage(self.ui.redV.value(), self.ui.blueV.value())
            
    def updateGlobalData(self):

        # load up debris types
        setupTable(self.ui.debrisTypes, self.cfg.debris_types)
        for i, (key, value) in enumerate(self.cfg.debris_types.iteritems()):
            self.ui.debrisTypes.setItem(i, 0, QTableWidgetItem(key))
            self.ui.debrisTypes.setItem(i, 1, QTableWidgetItem(
                '{:.2f}'.format(value['cdav'])))
        finiTable(self.ui.debrisTypes)

        # load up debris regions
        _debris_region = self.cfg.dic_debris_regions[self.cfg.region_name]
        setupTable(self.ui.debrisRegions, _debris_region)

        for i, key1 in enumerate(self.cfg.debris_types):

            _str = '{}_ratio'.format(key1)
            _value = _debris_region[_str]
            self.ui.debrisRegions.setItem(5 * i, 0, QTableWidgetItem(_str))
            self.ui.debrisRegions.setItem(5 * i, 1,
                                          QTableWidgetItem(
                                              '{:.3f}'.format(_value)))

            for j, key2 in enumerate(['frontalarea', 'mass'], 1):
                _mean_str = '{}_{}_mean'.format(key1, key2)
                _std_str = '{}_{}_stddev'.format(key1, key2)
                _mean = _debris_region[_mean_str]
                _std = _debris_region[_std_str]

                self.ui.debrisRegions.setItem(5 * i + 2 * j - 1, 0,
                                              QTableWidgetItem(_mean_str))
                self.ui.debrisRegions.setItem(5 * i + 2 * j - 1, 1,
                                              QTableWidgetItem(
                                                  '{:.3f}'.format(_mean)))

                self.ui.debrisRegions.setItem(5 * i + 2 * j, 0,
                                              QTableWidgetItem(_std_str))
                self.ui.debrisRegions.setItem(5 * i + 2 * j, 1,
                                              QTableWidgetItem(
                                                  '{:.3f}'.format(_std)))

        finiTable(self.ui.debrisRegions)

    def showHouseInfoDlg(self):
        from gui import house as gui_house
        dlg = gui_house.HouseViewer(self.cfg)
        dlg.exec_()

    def updateDebrisRegionsTable(self):

        self.cfg.region_name = unicode(self.ui.debrisRegion.currentText())
        self.cfg.set_debris_types()
        self.updateGlobalData()

    def updateTerrainCategoryTable(self):

        self.cfg.terrain_category = unicode(self.ui.terrainCategory.currentText())

        self.cfg.set_wind_profile()

        self.ui.boundaryProfile.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ui.boundaryProfile.setRowCount(len(self.cfg.profile_heights))
        self.ui.boundaryProfile.setSelectionBehavior(QTableWidget.SelectRows)
        self.ui.boundaryProfile.clearContents()

        for irow, head_col in enumerate(self.cfg.profile_heights):
            self.ui.boundaryProfile.setItem(irow, 0, QTableWidgetItem('{:.3f}'.format(head_col)))

        for key, _list in self.cfg.wind_profile.iteritems():
            for irow, value in enumerate(_list):
                self.ui.boundaryProfile.setItem(irow, key, QTableWidgetItem('{:.3f}'.format(value)))

        self.ui.boundaryProfile.resizeColumnsToContents()

    def updateConnectionTypeTable(self):
        # load up connection types grid
        setupTable(self.ui.connectionsTypes, self.cfg.df_types)
        for irow, (index, ctype) in enumerate(self.cfg.df_types.iterrows()):
            self.ui.connectionsTypes.setItem(irow, 0, QTableWidgetItem(index))
            self.ui.connectionsTypes.setItem(irow, 1, QTableWidgetItem('{:.3f}'.format(ctype['lognormal_strength'][0])))
            self.ui.connectionsTypes.setItem(irow, 2, QTableWidgetItem('{:.3f}'.format(ctype['lognormal_strength'][1])))
            self.ui.connectionsTypes.setItem(irow, 3, QTableWidgetItem('{:.3f}'.format(ctype['lognormal_dead_load'][0])))
            self.ui.connectionsTypes.setItem(irow, 4, QTableWidgetItem('{:.3f}'.format(ctype['lognormal_dead_load'][1])))
            self.ui.connectionsTypes.setItem(irow, 5, QTableWidgetItem(ctype['group_name']))
            self.ui.connectionsTypes.setItem(irow, 6, QTableWidgetItem('{:.3f}'.format(ctype['costing_area'])))
        finiTable(self.ui.connectionsTypes)
        
    def updateZonesTable(self):
        setupTable(self.ui.zones, self.cfg.df_zones)

        for irow, (index, z) in enumerate(self.cfg.df_zones.iterrows()):
            self.ui.zones.setItem(irow, 0, QTableWidgetItem(index))
            self.ui.zones.setItem(irow, 1, QTableWidgetItem('{:.3f}'.format(z['area'])))
            self.ui.zones.setItem(irow, 2, QTableWidgetItem('{:.3f}'.format(z['cpi_alpha'])))
            for dir in range(8):
                self.ui.zones.setItem(irow, 3+dir, QTableWidgetItem('{:.3f}'.format(self.cfg.df_zones_cpe_mean[dir][index])))
            for dir in range(8):
                self.ui.zones.setItem(irow, 11+dir, QTableWidgetItem('{:.3f}'.format(self.cfg.df_zones_cpe_str_mean[dir][index])))
            for dir in range(8):
                self.ui.zones.setItem(irow, 19+dir, QTableWidgetItem('{:.3f}'.format(self.cfg.df_zones_cpe_eave_mean[dir][index])))
        finiTable(self.ui.zones)
    
    def updateConnectionGroupTable(self):
        setupTable(self.ui.connGroups, self.cfg.df_groups)
        for irow, (index, ctg) in enumerate(self.cfg.df_groups.iterrows()):
            self.ui.connGroups.setItem(irow, 0, QTableWidgetItem(index))
            self.ui.connGroups.setItem(irow, 1, QTableWidgetItem('{:d}'.format(ctg['dist_order'])))
            self.ui.connGroups.setItem(irow, 2, QTableWidgetItem(ctg['dist_dir']))
            self.ui.connGroups.setItem(irow, 3, QTableWidgetItem(index))

            checked = self.cfg.flags.get('conn_type_group_{}'.format(index), False)
            cellWidget = QCheckBox()
            cellWidget.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.ui.connGroups.setCellWidget(irow, 4, cellWidget)
        finiTable(self.ui.connGroups)

    def onHouseChanged(self, selectedHouseName):
        # called whenever a house is selected (including by initial scenario)
        self.cfg.house_name = unicode(selectedHouseName)
        self.update_house_panel()

    def update_house_panel(self):
        self.updateConnectionGroupTable()
        self.updateConnectionTypeTable()
        
        # load up damage scenarios grid
        setupTable(self.ui.damageScenarios, self.cfg.dic_costings)

        for irow, (name, _inst) in enumerate(self.cfg.dic_costings.iteritems()):
            self.ui.damageScenarios.setItem(irow, 0, QTableWidgetItem(name))
            self.ui.damageScenarios.setItem(irow, 1, QTableWidgetItem('{:.3f}'.format(_inst.surface_area)))
            self.ui.damageScenarios.setItem(irow, 2, QTableWidgetItem('{:.3f}'.format(_inst.envelope_repair_rate)))
            self.ui.damageScenarios.setItem(irow, 3, QTableWidgetItem('{:.3f}'.format(_inst.internal_repair_rate)))

        finiTable(self.ui.damageScenarios)
        
        # load up connections grid
        setupTable(self.ui.connections, self.cfg.df_connections)
        for irow, (index, c) in enumerate(self.cfg.df_connections.iterrows()):
            self.ui.connections.setItem(irow, 0, QTableWidgetItem(c['group_name']))
            self.ui.connections.setItem(irow, 1, QTableWidgetItem(c['type_name']))
            self.ui.connections.setItem(irow, 2, QTableWidgetItem(c['zone_loc']))
            # edge = 'False'
            # if c['edge'] != 0:
            #     edge = 'True'
            # self.ui.connections.setItem(irow, 3, QTableWidgetItem(edge))
        finiTable(self.ui.connections)
        self.ui.connections.horizontalHeader().resizeSection(2, 70)
        self.ui.connections.horizontalHeader().resizeSection(1, 110)
        self.updateZonesTable()
        
    def updateModelFromUI(self):
        if self.dirty_conntypes:
            irow = 0
            for ctype in self.cfg.house.conn_types:
                sm = float(unicode(self.ui.connectionsTypes.item(irow, 1).text())) 
                ss = float(unicode(self.ui.connectionsTypes.item(irow, 2).text())) 
                dm = float(unicode(self.ui.connectionsTypes.item(irow, 3).text())) 
                ds = float(unicode(self.ui.connectionsTypes.item(irow, 4).text()))
                ctype.set_strength_params(sm, ss) 
                ctype.set_deadload_params(dm, ds)
                irow += 1
            self.dirty_conntypes = False
            
        self.cfg.updateModel()
    
    def stopScenario(self):
        self.stopTriggered = True
        
    def runScenario(self):
        self.statusBar().showMessage('Running Scenario')
        self.statusProgressBar.show()
        self.update_config_from_ui()
        self.ui.mplsheeting.axes.cla()
        self.ui.mplsheeting.axes.figure.canvas.draw()
        self.ui.mplbatten.axes.cla()
        self.ui.mplbatten.axes.figure.canvas.draw()
        self.ui.mplrafter.axes.cla()
        self.ui.mplrafter.axes.figure.canvas.draw()
        self.ui.connection_type_plot.axes.cla()
        self.ui.connection_type_plot.axes.figure.canvas.draw()
        self.ui.breaches_plot.axes.cla()
        self.ui.breaches_plot.axes.figure.canvas.draw()
        self.ui.wateringress_plot.axes.cla()
        self.ui.wateringress_plot.axes.figure.canvas.draw()
        
        # run simulation with progress bar
        self.ui.actionStop.setEnabled(True)
        self.ui.actionRun.setEnabled(False)
        self.ui.actionOpen_Scenario.setEnabled(False)
        self.ui.actionSave_Scenario.setEnabled(False)
        self.ui.actionSave_Scenario_As.setEnabled(False)

        # attempt to run the simulator, being careful with exceptions...
        try:
            run_time, list_results = simulate_wind_damage_to_houses(self.cfg,
                                                                    call_back=progress_callback)

            if run_time is not None:
                self.statusBar().showMessage(unicode('Simulation '
                                                     'complete in {:0.3f}'.format(run_time)))
                self.updateVulnCurve()
                # TODO Finish drawing results
                # self.updateHouseResultsTable(list_results)
                # self.updateConnectionTable(list_results)
                # self.updateConnectionTypePlots(list_results)
                # self.updateBreachPlot(list_results)
                # self.updateWaterIngressPlot(list_results)
                self.has_run = True

        except IOError as err:
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

    def updateWaterIngressPlot(self):
        self.statusBar().showMessage('Plotting Water Ingress')
        speeds, samples = self.simulator.get_windresults_samples_perc_water_ingress()
        wi_means = []
        xarr = []
        obsarr = []
        for v in speeds:
            wi_means.append( numpy.mean( numpy.array( samples[v] ) ) )
            for obs in samples[v]:
                obsarr.append(obs)
                xarr.append(v)
        self.ui.wateringress_plot.axes.hold(True)
        self.ui.wateringress_plot.axes.scatter(xarr, obsarr, s=8, marker='+', label='_nolegend_')
        self.ui.wateringress_plot.axes.plot(speeds, wi_means, c='b', marker='o')
        self.ui.wateringress_plot.axes.set_title('Water Ingress By Wind Speed')
        self.ui.wateringress_plot.axes.set_xlabel('Impact Wind speed (m/s)')
        self.ui.wateringress_plot.axes.set_ylabel('Water Ingress Cost')
        self.ui.wateringress_plot.axes.figure.canvas.draw()
        self.ui.wateringress_plot.axes.set_xlim((speeds[0], speeds[len(speeds)-1]))
        self.ui.wateringress_plot.axes.set_ylim((0))
        
    def updateBreachPlot(self):
        self.statusBar().showMessage('Plotting Debris Results')        
        self.ui.breaches_plot.axes.figure.clf()
                
        # we'll have three seperate y axis running at different scales
        cumimpact_arr = []
        cumsupply_arr = []
        speeds, breaches = self.simulator.get_windresults_perc_houses_breached()
        speeds, nv_samples = self.simulator.get_windresults_samples_nv()
        speeds, num_samples = self.simulator.get_windresults_samples_num_items()
        nv_means = []
        num_means = []
        for i, v in enumerate(speeds):
            nv_means.append(numpy.mean(numpy.array(nv_samples[v])))
            cumimpact_arr.append(sum(nv_means[:i+1]))            
            num_means.append(numpy.mean(numpy.array(num_samples[v])))
            cumsupply_arr.append(sum(num_means[:i+1]))
        
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
        host.set_xlim(speeds[0], speeds[len(speeds)-1])
                
        p1, = host.plot(speeds, breaches, label='Breached', c='b')
        p2, = par1.plot(speeds, nv_means, label='Impacts', c='g')
        p2b = par1.plot(speeds, cumimpact_arr, label='_nolegend_', c='g')
        p3, = par2.plot(speeds, num_means, label='Supply', c='r')
        p3b = par2.plot(speeds, cumsupply_arr, label='_nolegend_', c='r')
        
        host.axis['left'].label.set_color(p1.get_color())
        par1.axis['right'].label.set_color(p2.get_color())
        par2.axis['right2'].label.set_color(p3.get_color())
        
        host.legend(loc=2)
        self.ui.breaches_plot.axes.figure.canvas.draw()
                
    def updateStrengthPlot(self):
        self.ui.connection_type_plot.axes.hold(False)
        
        obs_dict = {}
        for ct in self.cfg.house.conn_types:
            obs_dict[ct.connection_type] = []
        for hr in self.house_results:
            for cr in hr[4]:
                connection_type_name = cr[0]
                sampled_strength = cr[3]
                obs_dict[connection_type_name].append(sampled_strength)
        
        for (ct_num, ct) in enumerate(self.cfg.house.conn_types):
            obs_arr = obs_dict[ct.connection_type]
            if len(obs_arr) > 0:                
                self.ui.connection_type_plot.axes.scatter([ct_num]*len(obs_arr), obs_arr, s=8, marker='+')
                self.ui.connection_type_plot.axes.hold(True)
                self.ui.connection_type_plot.axes.scatter([ct_num], numpy.mean(obs_arr), s=20, c='r', marker='o')
        
        xlabels = []
        xticks = []
        for (ct_num, ct) in enumerate(self.cfg.house.conn_types):
            xlabels.append(ct.connection_type)
            xticks.append(ct_num)
        
        self.ui.connection_type_plot.axes.set_xticks(xticks)
        self.ui.connection_type_plot.axes.set_xticklabels(xlabels, rotation='vertical')
        self.ui.connection_type_plot.axes.set_title('Connection Type Strengths')
        self.ui.connection_type_plot.axes.set_position([0.05, 0.20, 0.9, 0.75])
        self.ui.connection_type_plot.axes.figure.canvas.draw()     
        self.ui.connection_type_plot.axes.set_xlim((-0.5, len(xlabels)))
        
    def updateTypeDamagePlot(self):
        self.ui.connection_type_damages_plot.axes.hold(False)
        
        obs_dict = {}
        for ct in self.cfg.house.conn_types:
            obs_dict[ct.connection_type] = []
        for hr in self.house_results:
            for cr in hr[4]:
                connection_type_name = cr[0]
                broke_at_v = cr[2]
                obs_dict[connection_type_name].append(broke_at_v)
        
        for (ct_num, ct) in enumerate(self.cfg.house.conn_types):
            obs_arr = obs_dict[ct.connection_type]
            if len(obs_arr) > 0:                
                self.ui.connection_type_damages_plot.axes.scatter([ct_num]*len(obs_arr), obs_arr, s=8, marker='+')
                self.ui.connection_type_damages_plot.axes.hold(True)
                self.ui.connection_type_damages_plot.axes.scatter([ct_num], numpy.mean(obs_arr), s=20, c='r', marker='o')
        
        xlabels = []
        xticks = []
        for (ct_num, ct) in enumerate(self.cfg.house.conn_types):
            xlabels.append(ct.connection_type)
            xticks.append(ct_num)
        
        self.ui.connection_type_damages_plot.axes.set_xticks(xticks)
        self.ui.connection_type_damages_plot.axes.set_xticklabels(xlabels, rotation='vertical')
        self.ui.connection_type_damages_plot.axes.set_title('Connection Type Damage Speeds')
        self.ui.connection_type_damages_plot.axes.set_position([0.05, 0.20, 0.9, 0.75])
        self.ui.connection_type_damages_plot.axes.figure.canvas.draw()     
        self.ui.connection_type_damages_plot.axes.set_xlim((-0.5, len(xlabels)))
        
    def updateConnectionTypePlots(self):
        self.statusBar().showMessage('Plotting Connection Types')
        self.updateStrengthPlot()
        self.updateTypeDamagePlot()

    def updateConnectionTable(self):
        self.statusBar().showMessage('Updating Connections Table')
        for irow, c in enumerate(self.cfg.house.connections):
            self.ui.connections.setItem(irow, 4, QTableWidgetItem('{:.3f}'.format(c.result_failure_v)))
            self.ui.connections.setItem(irow, 5, QTableWidgetItem("%d" % c.result_failure_v_i))
        
    def updateHouseResultsTable(self):
        self.statusBar().showMessage('Updating Zone Results')
        self.ui.zoneResults.clear()
        for (house_num, hr) in enumerate(self.house_results):
            parent = QTreeWidgetItem(self.ui.zoneResults, ['H%d (%s/%.3f/%s)'%(house_num+1, config.Config.dirs[hr[2]], hr[3], hr[5]),
                                                           '', '', '', ''])
            zone_results_dict = hr[0]      
            for zr_key in sorted(zone_results_dict):
                zr = zone_results_dict[zr_key]
                QTreeWidgetItem(parent, ['', zr[0], '%.3f'%zr[1], '%.3f'%zr[2], '%.3f'%zr[3]])
                
        self.ui.zoneResults.resizeColumnToContents(0)
        self.ui.zoneResults.resizeColumnToContents(1)
        self.ui.zoneResults.resizeColumnToContents(2)
        self.ui.zoneResults.resizeColumnToContents(3)
        self.ui.zoneResults.resizeColumnToContents(4)
        self.ui.zoneResults.resizeColumnToContents(5)
                
        self.statusBar().showMessage('Updating Connection Results')
        self.ui.connectionResults.clear()
        for (house_num, hr) in enumerate(self.house_results):
            parent = QTreeWidgetItem(self.ui.connectionResults, ['H%d (%s/%.3f/%s)'%(house_num+1, scenario.Scenario.dirs[hr[2]], hr[3], hr[5])])
            for cr in hr[4]:
                damage_report = cr[5]
                load = damage_report.get('load', 0)
                
                load_desc = ''
                if load == 99.9:
                    load_desc = 'collapse'
                elif load != 0:
                    load_desc = '%.3f' % load
                
                conn_parent = QTreeWidgetItem(parent, ['%s_%s'%(cr[0], cr[1]), 
                                                       '%.3f'%cr[2] if cr[2] != 0 else '', 
                                                       '%.3f'%cr[3], 
                                                       '%.3f'%cr[4],
                                                       load_desc]) 
                for infl_dict in damage_report.get('infls', {}):
                    QTreeWidgetItem(conn_parent, ['%.3f(%s) = %.3f(i) * %.3f(area) * %.3f(pz)' % (infl_dict['load'], 
                                                                                                  infl_dict['name'],
                                                                                                  infl_dict['infl'],
                                                                                                  infl_dict['area'],
                                                                                                  infl_dict['pz'])])
        self.ui.connectionResults.resizeColumnToContents(1)
        self.ui.connectionResults.resizeColumnToContents(2)
        self.ui.connectionResults.resizeColumnToContents(3)
        self.ui.connectionResults.resizeColumnToContents(4)
        header_view = self.ui.connectionResults.header()
        header_view.resizeSection(0, 350)
                            
    def open_scenario(self):
        settings = QSettings()
        # Set the path to the default
        scenario_path = SCENARIOS_DIR
        if settings.contains("ScenarioFolder"):
            # we have a saved scenario path so use that
            scenario_path = unicode(settings.value("ScenarioFolder").toString())

        filename = QFileDialog.getOpenFileName(self, "Scenarios", scenario_path, "Scenarios (*.cfg)")
        if not filename:
            return

        config_file = '%s' % (filename)
        if os.path.isfile(config_file):
            self.file_load(config_file)
            settings.setValue("ScenarioFolder", QVariant(QString(os.path.dirname(config_file))))
        else:
            msg = 'Unable to load scenario: {}\nFile not found.'.format(config_file)
            QMessageBox.warning(self, "VAWS Program Warning", unicode(msg))

    def save_scenario(self):
        self.update_config_from_ui()
        self.cfg.save_config()
        self.ui.statusbar.showMessage('Saved to file %s' % self.cfg.cfg_file)
        self.update_ui_from_config()
        
    def save_as_scenario(self):
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

            settings = QSettings()
            settings.setValue("ScenarioFolder", QVariant(QString(path_cfg)))
            self.cfg.path_cfg = path_cfg
            self.cfg.cfg_file = os.path.join(path_cfg, file_suffix_name)

            self.cfg.save_config()
            self.update_ui_from_config()
        else:
            msg = 'No scenario name entered. Action cancelled'
            QMessageBox.warning(self, "VAWS Program Warning", unicode(msg))

    def set_scenario(self, s=None):
        if s:
            self.cfg = s

        self.update_ui_from_config()

        self.statusBar().showMessage('Ready')

    def closeEvent(self, event):
        if self.okToContinue():
            if self.cfg.cfg_file and QFile.exists(self.cfg.cfg_file):
                settings = QSettings()
                filename = QVariant(QString(self.cfg.cfg_file))
                settings.setValue("LastFile", filename)
            self.storeSizePosToSettings()
        else:
            event.ignore()
        
    def update_ui_from_config(self):
        if self.cfg:
            self.statusBar().showMessage('Updating', 1000)

            # Scenario
            self.ui.numHouses.setText('{:d}'.format(self.cfg.no_sims))
            self.ui.houseName.setCurrentIndex(self.ui.houseName.findText(self.cfg.house_name))
            self.ui.terrainCategory.setCurrentIndex(
                    self.ui.terrainCategory.findText(self.cfg.terrain_category))
            self.ui.regionalShielding.setText('{:.1f}'.format(
                self.cfg.regional_shielding_factor))
            self.ui.windMin.setValue(self.cfg.wind_speed_min)
            self.ui.windMax.setValue(self.cfg.wind_speed_max)
            self.ui.windSteps.setValue(self.cfg.wind_speed_steps)
            self.ui.windDirection.setCurrentIndex(self.cfg.wind_dir_index)

            # Debris
            self.ui.debris.setChecked(self.cfg.flags.get('debris'))
            if self.cfg.flags['debris']:
                self.ui.debrisRegion.setCurrentIndex(
                    self.ui.debrisRegion.findText(self.cfg.region_name))
                self.ui.buildingSpacing.setCurrentIndex(
                    self.ui.buildingSpacing.findText(
                        '{:.1f}'.format(self.cfg.building_spacing)))
                self.ui.sourceItems.setValue(self.cfg.source_items)
                self.ui.debrisRadius.setValue(self.cfg.debris_radius)
                self.ui.debrisAngle.setValue(self.cfg.debris_angle)
                self.ui.flighttimeMean.setText(
                    '{:.3f}'.format(self.cfg.flight_time_mean))
                self.ui.flighttimeStddev.setText(
                    '{:.3f}'.format(self.cfg.flight_time_stddev))
                self.ui.staggeredDebrisSources.setChecked(
                    self.cfg.flags.get('debris_staggered_sources'))

            # options
            self.ui.redV.setValue(self.cfg.red_v)
            self.ui.blueV.setValue(self.cfg.blue_v)
            self.ui.seedRandom.setChecked(self.cfg.flags.get('random_seed'))
            self.ui.diffShielding.setChecked(self.cfg.flags.get('diff_shielding'))
            self.ui.waterIngress.setChecked(self.cfg.flags.get('water_ingress'))

            self.ui.fitLognormalCurve.setChecked(self.cfg.flags.get('vul_fit_log'))
            # self.ui.distribution.setChecked(self.cfg.flags.get('dmg_distribute'))
            self.ui.actionRun.setEnabled(True)
            # self.ui.debrisExtension.setText('%f' % self.cfg.debris_extension)

            self.ui.constructionEnabled.setChecked(self.cfg.flags.get('construction_levels'))

            prob, mf, cf = self.cfg.getConstructionLevel('low')
            self.ui.lowProb.setValue(float(prob))
            self.ui.lowMean.setValue(float(mf))
            self.ui.lowCov.setValue(float(cf))

            prob, mf, cf = self.cfg.getConstructionLevel('medium')
            self.ui.mediumProb.setValue(float(prob))
            self.ui.mediumMean.setValue(float(mf))
            self.ui.mediumCov.setValue(float(cf))

            prob, mf, cf = self.cfg.getConstructionLevel('high')
            self.ui.highProb.setValue(float(prob))
            self.ui.highMean.setValue(float(mf))
            self.ui.highCov.setValue(float(cf))
            
            self.ui.slight.setValue(self.cfg.fragility_thresholds.loc['slight', 'threshold'])
            self.ui.medium.setValue(self.cfg.fragility_thresholds.loc['medium', 'threshold'])
            self.ui.severe.setValue(self.cfg.fragility_thresholds.loc['severe', 'threshold'])
            self.ui.complete.setValue(self.cfg.fragility_thresholds.loc['complete', 'threshold'])

            self.update_house_panel()

            if self.cfg.cfg_file:
                self.statusBarScenarioLabel.setText('Scenario: %s' % (os.path.basename(self.cfg.cfg_file)))
        else:
            self.statusBarScenarioLabel.setText('Scenario: None')
        
    def update_config_from_ui(self):
        new_cfg = self.cfg

        # Scenario section
        new_cfg.no_sims = int(self.ui.numHouses.text())
        # new_cfg.house_name = (unicode(self.ui.houseName.currentText()))
        new_cfg.terrain_category = unicode(self.ui.terrainCategory.currentText())
        new_cfg.regional_shielding_factor = float(unicode(self.ui.regionalShielding.text()))
        new_cfg.wind_speed_min = self.ui.windMin.value()
        new_cfg.wind_speed_max = self.ui.windMax.value()
        new_cfg.wind_speed_steps = self.ui.windSteps.value()
        new_cfg.wind_dir_index = self.ui.windDirection.currentIndex()

        # Debris section
        new_cfg.set_region_name(unicode(self.ui.debrisRegion.currentText()))
        new_cfg.flags['plot_fragility'] = True
        new_cfg.flags['plot_vul'] = True
        new_cfg.flags['random_seed'] = self.ui.seedRandom.isChecked()
        # new_cfg.flags['vul_fit_log'] = self.ui.fitLognormalCurve.isChecked())
        new_cfg.flags['water_ingress'] = self.ui.waterIngress.isChecked()
        # new_cfg.flags['dmg_distribute', self.ui.distribution.isChecked())
        new_cfg.flags['diff_shielding'] = self.ui.diffShielding.isChecked()
        new_cfg.flags['debris'] = self.ui.debris.isChecked()
        new_cfg.flags['debris_staggered_sources'] = self.ui.staggeredDebrisSources.isChecked()
        new_cfg.flags['construction_levels'] = self.ui.constructionEnabled.isChecked()
        new_cfg.source_items = self.ui.sourceItems.value()
        new_cfg.building_spacing = float(unicode(self.ui.buildingSpacing.currentText()))
        new_cfg.debris_radius = self.ui.debrisRadius.value()
        new_cfg.debris_angle = self.ui.debrisAngle.value()
        # new_cfg.debris_extension = float(self.ui.debrisExtension.text())
        new_cfg.flight_time_mean = float(unicode(self.ui.flighttimeMean.text()))
        new_cfg.flight_time_stddev = float(unicode(self.ui.flighttimeStddev.text()))

        new_cfg.flight_time_log_mu, new_cfg.flight_time_log_std = \
            compute_logarithmic_mean_stddev(new_cfg.flight_time_mean,
                                                  new_cfg.flight_time_stddev)

        new_cfg.red_v = float(unicode(self.ui.redV.value()))
        new_cfg.blue_v = float(unicode(self.ui.blueV.value()))
        
        new_cfg.setConstructionLevel('low',
                                float(unicode(self.ui.lowProb.value())), 
                                float(unicode(self.ui.lowMean.value())), 
                                float(unicode(self.ui.lowCov.value())))
        new_cfg.setConstructionLevel('medium',
                                float(unicode(self.ui.mediumProb.value())), 
                                float(unicode(self.ui.mediumMean.value())), 
                                float(unicode(self.ui.mediumCov.value())))
        new_cfg.setConstructionLevel('high',
                                float(unicode(self.ui.highProb.value())), 
                                float(unicode(self.ui.highMean.value())), 
                                float(unicode(self.ui.highCov.value())))
        
        new_cfg.fragility_thresholds['slight'] = float(self.ui.slight.value())
        new_cfg.fragility_thresholds['medium'] = float(self.ui.medium.value())
        new_cfg.fragility_thresholds['severe'] = float(self.ui.severe.value())
        new_cfg.fragility_thresholds['complete'] = float(self.ui.complete.value())

        for irow, (index, ctg) in enumerate(self.cfg.df_groups.iterrows()):

            cellWidget = self.ui.connGroups.cellWidget(irow, 4)
            new_cfg.flags['conn_type_group_{}'.format(index)] = True \
                if cellWidget.checkState() == Qt.Checked else False

        self.dirty_scenario = new_cfg != self.cfg
        if self.dirty_scenario:
            self.set_scenario(new_cfg)
        
    def file_load(self, fname):
        try:
            self.cfg = Config(fname)
            self.set_scenario(self.cfg)
            set_logger(self.cfg)
        except Exception as excep:
            logging.exception("Loading configuration caused exception")

            msg = 'Unable to load previous scenario: %s\nCreating new scenario.' % fname
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
        mgr = debris.DebrisManager(self.cfg.house,
                                  self.cfg.region,
                                  self.cfg.wind_speed_min, self.cfg.wind_speed_max, self.cfg.wind_speed_steps,
                                  self.cfg.getOpt_DebrisStaggeredSources(),
                                  self.cfg.debris_radius,
                                  self.cfg.debris_angle, 
                                  self.cfg.debris_extension,
                                  self.cfg.building_spacing,
                                  self.cfg.source_items,
                                  self.cfg.flighttime_mean,
                                  self.cfg.flighttime_stddev)
        
        v, ok = QInputDialog.getInteger(self, "Debris Test", "Wind Speed (m/s):", 50, 10, 200)
        if ok:
            mgr.set_wind_direction_index(self.cfg.getWindDirIndex())
            mgr.run(v, True)
            mgr.render(v)
    
    def testConstructionLevels(self):
        self.update_config_from_ui()
        items = []
        for ctype in self.cfg.house.conn_types:
            items.append(ctype.connection_type) 
        selected_ctype, ok = QInputDialog.getItem(self, "Construction Test", "Connection Type:", items, 0, False)
        if ok:
            import matplotlib.pyplot as plt
            for ctype in self.cfg.house.conn_types:
                if ctype.connection_type == selected_ctype:
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    x = []
                    n = 50000
                    for i in xrange(n):
                        level, mean_factor, cov_factor = self.cfg.sampleConstructionLevel()
                        rv = ctype.sample_strength(mean_factor, cov_factor)
                        x.append(rv)
                        
                    ax.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
                    fig.canvas.draw()
                    plt.show()    


def run_gui():
    parser = process_commandline()

    (options, args) = parser.parse_args()

    if not options.config_filename:
        settings = QSettings()
        if settings.contains("LastFile"):
            initial_scenario = unicode(settings.value("LastFile").toString())
        else:
            initial_scenario = DEFAULT_SCENARIO

        initial_config = Config(cfg_file=initial_scenario)

    else:
        initial_config = Config(cfg_file=options.config_filename)

    if options.verbose:
        initial_config.logging_level = options.verbose
        set_logger(initial_config)

    app = QApplication(sys.argv)
    app.setOrganizationName("Geoscience Australia")
    app.setOrganizationDomain("ga.gov.au")
    app.setApplicationName("WindSim")
    splash_image = QPixmap()

    source_dir = os.path.dirname(__file__)
    splash_file = os.path.join(source_dir, 'images/splash/splash.png')
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



