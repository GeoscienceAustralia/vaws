#!/usr/bin/env python
# adjust python path so we may import things from peer packages
import sys
import os.path
import logging

from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
from PyQt4.QtCore import SIGNAL, QTimer, Qt, QSettings, QVariant, QString, QFile
from PyQt4.QtGui import QProgressBar, QLabel, QMainWindow, QApplication, QTableWidget, QPixmap,\
                        QTableWidgetItem, QDialog, QCheckBox, QFileDialog, QIntValidator,\
                        QDoubleValidator, QMessageBox, QTreeWidgetItem, QInputDialog, QSplashScreen
import numpy

from main_ui import Ui_main
from vaws import simulation, scenario, house, version, debris, stats, output

from mixins import PersistSizePosMixin, setupTable, finiTable

my_app = None


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
        
        self.simulator = None

        self.ui = Ui_main()
        self.ui.setupUi(self)

        windowTitle = version.VERSION_DESC
        self.setWindowTitle(unicode(windowTitle))

        self.s = init_scenario

        self.ui.houseName.addItems(self.s.df_house['name'].values)
        self.ui.houseName.setCurrentIndex(-1)
        self.ui.buildingSpacing.addItems(('20', '40'))
        self.ui.terrainCategory.addItems(self.s.terrain_categories)
        self.ui.terrainCategory.setCurrentIndex(-1)
        self.ui.windDirection.addItems(self.s.wind_dir)
        self.ui.numHouses.setValidator(QIntValidator(1, 10000, self.ui.numHouses))
        self.ui.flighttimeMean.setValidator(QDoubleValidator(0.0, 100.0, 3, self.ui.flighttimeMean))
        self.ui.flighttimeStddev.setValidator(QDoubleValidator(0.0, 100.0, 3, self.ui.flighttimeStddev))
        self.ui.debrisExtension.setValidator(QDoubleValidator(0.0, 100.0, 3, self.ui.debrisExtension))
      
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
        self.filename = None
        self.initSizePosFromSettings()
        self.connect(self.ui.actionOpen_Scenario, SIGNAL("triggered()"), self.openScenario)
        self.connect(self.ui.actionRun, SIGNAL("triggered()"), self.runScenario)
        self.connect(self.ui.actionStop, SIGNAL("triggered()"), self.stopScenario)
        self.connect(self.ui.actionSave_Scenario, SIGNAL("triggered()"), self.saveScenario)
        self.connect(self.ui.actionSave_Scenario_As, SIGNAL("triggered()"), self.saveAsScenario)
        self.connect(self.ui.actionHouse_Info, SIGNAL("triggered()"), self.showHouseInfoDlg)
        self.connect(self.ui.testDebrisButton, SIGNAL("clicked()"), self.testDebrisSettings)
        self.connect(self.ui.testConstructionButton, SIGNAL("clicked()"), self.testConstructionLevels)
        self.connect(self.ui.houseName, SIGNAL("currentIndexChanged(QString)"), self.onHouseChanged)
        self.connect(self.ui.terrainCategory, SIGNAL("currentIndexChanged(QString)"), self.updateTerrainCategoryTable)
        self.connect(self.ui.windMin, SIGNAL("valueChanged(int)"), lambda val: self.onSliderChanged(self.ui.windMinLabel, val))
        self.connect(self.ui.windMax, SIGNAL("valueChanged(int)"), lambda val: self.onSliderChanged(self.ui.windMaxLabel, val))
        self.connect(self.ui.windSteps, SIGNAL("valueChanged(int)"), lambda val: self.onSliderChanged(self.ui.windStepsLabel, val))
        self.connect(self.ui.debrisRadius, SIGNAL("valueChanged(int)"), lambda val: self.onSliderChanged(self.ui.debrisRadiusLabel, val))
        self.connect(self.ui.debrisAngle, SIGNAL("valueChanged(int)"), lambda val: self.onSliderChanged(self.ui.debrisAngleLabel, val))
        self.connect(self.ui.sourceItems, SIGNAL("valueChanged(int)"), lambda val: self.onSliderChanged(self.ui.sourceItemsLabel, val))
        self.connect(self.ui.redV, SIGNAL("valueChanged(int)"), lambda val: self.onSliderChanged(self.ui.redVLabel, val))
        self.connect(self.ui.blueV, SIGNAL("valueChanged(int)"), lambda val: self.onSliderChanged(self.ui.blueVLabel, val))
        self.connect(self.ui.applyDisplayChangesButton, SIGNAL("clicked()"), self.updateDisplaySettings)
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
        for irow in range(len(self.s.house.connections)):
            if unicode(self.ui.connections.item(irow, 0).text()) == connection_name:
                self.ui.connections.setCurrentCell(irow, 0)
                break
        
    def onSelectZone(self, zoneLoc):
        for irow in range(len(self.s.house.zones)):
            if unicode(self.ui.zones.item(irow, 0).text()) == zoneLoc:
                self.ui.zones.setCurrentCell(irow, 0)
                break

    def onConnectionsDoubleClicked(self, row, col):
        connection_name = unicode(self.ui.connections.item(row, 0).text())
        self.selected_conn = house.connByNameMap[connection_name]
        self.showConnectionViewer() 
        
    def onSliderChanged(self, label, val):
        label.setText('%d' % val)
        
    def updateVulnCurveSettings(self):
        if self.has_run:
            self.updateScenarioFromUI()
            self.updateVulnCurve()
            self.statusBar().showMessage(unicode('Curve Fit Updated'))
            
    def updateVulnCurve(self, list_results):
        # self.show_results(None, self.ui.redV.value(), self.ui.blueV.value())
        self.ui.mplfrag.axes.cla()
        self.ui.mplfrag.axes.figure.canvas.draw()
        self.ui.mplvuln.axes.cla()
        self.ui.mplvuln.axes.figure.canvas.draw()

        # loop through each result in the list
        for result in list_results:
            self.ui.mplfrag.axes.plot(self.s.speeds,
                                      result['di']
                                      , '-', 0.3)
        # self.ui.mplvuln.axes.plot(self.s.speeds,result[0][''])
        # self.ui.sumOfSquares.setText('%f' % self.simulator.ss)
        # self.ui.coeff_1.setText('%f' % self.simulator.A_final[0])
        # self.ui.coeff_2.setText('%f' % self.simulator.A_final[1])

    def updateDisplaySettings(self):
        if self.has_run:
            self.simulator.plot_connection_damage(self.ui.redV.value(), self.ui.blueV.value())
            
    def updateGlobalData(self):
        # load up debris types
        debrisTypes = []
        setupTable(self.ui.debrisTypes, None)
        irow = 0
        for dt in debrisTypes:
            self.ui.debrisTypes.setItem(irow, 0, QTableWidgetItem(dt[0]))
            self.ui.debrisTypes.setItem(irow, 1, QTableWidgetItem("%0.3f" % dt[1]))
            irow += 1
        finiTable(self.ui.debrisTypes)
       
        # load up debris regions
        setupTable(self.ui.debrisRegions, self.s.debris_regions)
        # TODO complete debris model
        for irow, dr in enumerate([]):
            self.ui.debrisRegions.setItem(irow, 0, QTableWidgetItem(dr.name))
            self.ui.debrisRegions.setItem(irow, 1, QTableWidgetItem("%0.3f" % dr.alpha))
            self.ui.debrisRegions.setItem(irow, 2, QTableWidgetItem("%0.3f" % dr.beta))
            self.ui.debrisRegions.setItem(irow, 3, QTableWidgetItem("%0.3f" % dr.cr))
            self.ui.debrisRegions.setItem(irow, 4, QTableWidgetItem("%0.3f" % dr.cmm))
            self.ui.debrisRegions.setItem(irow, 5, QTableWidgetItem("%0.3f" % dr.cmc))
            self.ui.debrisRegions.setItem(irow, 6, QTableWidgetItem("%0.3f" % dr.cfm))
            self.ui.debrisRegions.setItem(irow, 7, QTableWidgetItem("%0.3f" % dr.cfc))
            self.ui.debrisRegions.setItem(irow, 8, QTableWidgetItem("%0.3f" % dr.rr))
            self.ui.debrisRegions.setItem(irow, 9, QTableWidgetItem("%0.3f" % dr.rmm))
            self.ui.debrisRegions.setItem(irow, 10, QTableWidgetItem("%0.3f" % dr.rmc))
            self.ui.debrisRegions.setItem(irow, 11, QTableWidgetItem("%0.3f" % dr.rfm))
            self.ui.debrisRegions.setItem(irow, 12, QTableWidgetItem("%0.3f" % dr.rfc))
            self.ui.debrisRegions.setItem(irow, 13, QTableWidgetItem("%0.3f" % dr.pr))
            self.ui.debrisRegions.setItem(irow, 14, QTableWidgetItem("%0.3f" % dr.pmm))
            self.ui.debrisRegions.setItem(irow, 15, QTableWidgetItem("%0.3f" % dr.pmc))
            self.ui.debrisRegions.setItem(irow, 16, QTableWidgetItem("%0.3f" % dr.pfm))
            self.ui.debrisRegions.setItem(irow, 17, QTableWidgetItem("%0.3f" % dr.pfc))
            self.ui.debrisRegion.addItem(dr.name)

        finiTable(self.ui.debrisRegions)
        
    def showHouseInfoDlg(self):
        from gui import house as gui_house
        dlg = gui_house.HouseViewer(self.s)
        dlg.exec_()
    
    def updateTerrainCategoryTable(self, tc):
        zarr = [3, 5, 7, 10, 12, 15, 17, 20, 25, 30]

        self.s.terrain_category = unicode(self.ui.terrainCategory.currentText())

        self.s.set_wind_profile(self.s.path_wind_profiles)

        self.ui.boundaryProfile.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ui.boundaryProfile.setRowCount(len(zarr))
        self.ui.boundaryProfile.setSelectionBehavior(QTableWidget.SelectRows)
        self.ui.boundaryProfile.clearContents()

        for irow, head_col in enumerate(zarr):
            self.ui.boundaryProfile.setItem(irow, 0, QTableWidgetItem("%0.3f" % head_col))

        for icol in range(1, 11):
            for irow in range(0, len(zarr)):
                self.ui.boundaryProfile.setItem(irow, icol,
                                                QTableWidgetItem("%0.3f" % self.s.wind_profile[icol][irow]))

        self.ui.boundaryProfile.resizeColumnsToContents()

    def updateConnectionTypeTable(self):
        # load up connection types grid
        setupTable(self.ui.connectionsTypes, self.s.df_types)
        for irow, (index, ctype) in enumerate(self.s.df_types.iterrows()):
            self.ui.connectionsTypes.setItem(irow, 0, QTableWidgetItem(index))
            self.ui.connectionsTypes.setItem(irow, 1, QTableWidgetItem("%0.3f" % ctype['lognormal_strength'][0]))
            self.ui.connectionsTypes.setItem(irow, 2, QTableWidgetItem("%0.3f" % ctype['lognormal_strength'][1]))
            self.ui.connectionsTypes.setItem(irow, 3, QTableWidgetItem("%0.3f" % ctype['lognormal_dead_load'][0]))
            self.ui.connectionsTypes.setItem(irow, 4, QTableWidgetItem("%0.3f" % ctype['lognormal_dead_load'][1]))
            self.ui.connectionsTypes.setItem(irow, 5, QTableWidgetItem(ctype['group_name']))
            self.ui.connectionsTypes.setItem(irow, 6, QTableWidgetItem("%0.3f" % ctype['costing_area']))
        finiTable(self.ui.connectionsTypes)
        
    def updateZonesTable(self):
        setupTable(self.ui.zones, self.s.df_zones)

        for irow, (index, z) in enumerate(self.s.df_zones.iterrows()):
            self.ui.zones.setItem(irow, 0, QTableWidgetItem(index))
            self.ui.zones.setItem(irow, 1, QTableWidgetItem("%0.3f" % z['area']))
            self.ui.zones.setItem(irow, 2, QTableWidgetItem("%0.3f" % z['cpi_alpha']))
            for dir in range(8):
                self.ui.zones.setItem(irow, 3+dir, QTableWidgetItem("%0.3f" % self.s.df_zones_cpe_mean[dir][index]))
            for dir in range(8):
                self.ui.zones.setItem(irow, 11+dir, QTableWidgetItem("%0.3f" % self.s.df_zones_cpe_str_mean[dir][index]))
            for dir in range(8):
                self.ui.zones.setItem(irow, 19+dir, QTableWidgetItem("%0.3f" % self.s.df_zones_cpe_eave_mean[dir][index]))
        finiTable(self.ui.zones)
    
    def updateConnectionGroupTable(self):
        setupTable(self.ui.connGroups, self.s.df_groups)
        for irow, (index, ctg) in enumerate(self.s.df_groups.iterrows()):
            self.ui.connGroups.setItem(irow, 0, QTableWidgetItem(index))
            self.ui.connGroups.setItem(irow, 1, QTableWidgetItem("%d" % ctg['dist_order']))
            self.ui.connGroups.setItem(irow, 2, QTableWidgetItem(ctg['dist_dir']))
            self.ui.connGroups.setItem(irow, 3, QTableWidgetItem(index))
            cellWidget = QCheckBox()
            checked = self.s.get_flag('conn_type_group_{}'.format(index), False)

            cellWidget.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.ui.connGroups.setCellWidget(irow, 4, cellWidget)            
        finiTable(self.ui.connGroups)
        
    def onHouseChanged(self, selectedHouseName):
        # called whenever a house is selected (including by initial scenario)
        self.s.house_name = unicode(selectedHouseName)
        self.updateConnectionGroupTable()
        self.updateConnectionTypeTable()
        
        # load up damage scenarios grid
        setupTable(self.ui.damageScenarios, self.s.df_damage_costing)

        for irow, (index, ds) in enumerate(self.s.df_damage_costing.iterrows()):
            self.ui.damageScenarios.setItem(irow, 0, QTableWidgetItem(ds['name']))
            self.ui.damageScenarios.setItem(irow, 1, QTableWidgetItem("%d" % ds['surface_area']))
            self.ui.damageScenarios.setItem(irow, 2, QTableWidgetItem("%0.3f" % ds['envelope_repair_rate']))
            self.ui.damageScenarios.setItem(irow, 3, QTableWidgetItem("%0.3f" % ds['internal_repair_rate']))

        finiTable(self.ui.damageScenarios)
        
        # load up connections grid
        setupTable(self.ui.connections, self.s.df_conns)
        for irow, (index, c) in enumerate(self.s.df_conns.iterrows()):
            self.ui.connections.setItem(irow, 0, QTableWidgetItem(c['group_name']))
            self.ui.connections.setItem(irow, 1, QTableWidgetItem(c['type_name']))
            self.ui.connections.setItem(irow, 2, QTableWidgetItem(c['zone_loc']))
            edge = 'False'
            if c['edge'] != 0:
                edge = 'True'
            self.ui.connections.setItem(irow, 3, QTableWidgetItem(edge))
        finiTable(self.ui.connections)
        self.ui.connections.horizontalHeader().resizeSection(2, 70)
        self.ui.connections.horizontalHeader().resizeSection(1, 110)
        self.updateZonesTable()
        
    def updateModelFromUI(self):
        if self.dirty_conntypes:
            irow = 0
            for ctype in self.s.house.conn_types:
                sm = float(unicode(self.ui.connectionsTypes.item(irow, 1).text())) 
                ss = float(unicode(self.ui.connectionsTypes.item(irow, 2).text())) 
                dm = float(unicode(self.ui.connectionsTypes.item(irow, 3).text())) 
                ds = float(unicode(self.ui.connectionsTypes.item(irow, 4).text()))
                ctype.set_strength_params(sm, ss) 
                ctype.set_deadload_params(dm, ds)
                irow += 1
            self.dirty_conntypes = False
            
        self.s.updateModel()
    
    def stopScenario(self):
        self.stopTriggered = True
        
    def runScenario(self):
        self.statusBar().showMessage('Running Scenario')
        self.statusProgressBar.show()
        self.updateScenarioFromUI()
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
            run_time, list_results = simulation.simulate_wind_damage_to_houses(self.s,
                                                                               call_back=progress_callback)

            if run_time is not None:
                self.statusBar().showMessage(unicode('Simulation '
                                                     'complete in {:0.3f}'.format(run_time)))
                self.updateVulnCurve(list_results)
                # TODO Finish drawing results
                # self.updateHouseResultsTable(list_results)
                # self.updateConnectionTable(list_results)
                # self.updateConnectionTypePlots(list_results)
                # self.updateBreachPlot(list_results)
                # self.updateWaterIngressPlot(list_results)
                self.has_run = True

        except IOError, err:
            QMessageBox.warning(self, 'VAWS Program Warning', unicode('A report file is still open by another program, unable to run simulation.'))
            self.statusBar().showMessage(unicode(''))
        except Exception, err:
            self.statusBar().showMessage(unicode('Fatal Error Occurred: %s' % err))
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
        for ct in self.s.house.conn_types:
            obs_dict[ct.connection_type] = []
        for hr in self.house_results:
            for cr in hr[4]:
                connection_type_name = cr[0]
                sampled_strength = cr[3]
                obs_dict[connection_type_name].append(sampled_strength)
        
        for (ct_num, ct) in enumerate(self.s.house.conn_types):
            obs_arr = obs_dict[ct.connection_type]
            if len(obs_arr) > 0:                
                self.ui.connection_type_plot.axes.scatter([ct_num]*len(obs_arr), obs_arr, s=8, marker='+')
                self.ui.connection_type_plot.axes.hold(True)
                self.ui.connection_type_plot.axes.scatter([ct_num], numpy.mean(obs_arr), s=20, c='r', marker='o')
        
        xlabels = []
        xticks = []
        for (ct_num, ct) in enumerate(self.s.house.conn_types):
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
        for ct in self.s.house.conn_types:
            obs_dict[ct.connection_type] = []
        for hr in self.house_results:
            for cr in hr[4]:
                connection_type_name = cr[0]
                broke_at_v = cr[2]
                obs_dict[connection_type_name].append(broke_at_v)
        
        for (ct_num, ct) in enumerate(self.s.house.conn_types):
            obs_arr = obs_dict[ct.connection_type]
            if len(obs_arr) > 0:                
                self.ui.connection_type_damages_plot.axes.scatter([ct_num]*len(obs_arr), obs_arr, s=8, marker='+')
                self.ui.connection_type_damages_plot.axes.hold(True)
                self.ui.connection_type_damages_plot.axes.scatter([ct_num], numpy.mean(obs_arr), s=20, c='r', marker='o')
        
        xlabels = []
        xticks = []
        for (ct_num, ct) in enumerate(self.s.house.conn_types):
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
        for irow, c in enumerate(self.s.house.connections):
            self.ui.connections.setItem(irow, 4, QTableWidgetItem("%0.3f" % c.result_failure_v))
            self.ui.connections.setItem(irow, 5, QTableWidgetItem("%d" % c.result_failure_v_i))
        
    def updateHouseResultsTable(self):
        self.statusBar().showMessage('Updating Zone Results')
        self.ui.zoneResults.clear()
        for (house_num, hr) in enumerate(self.house_results):
            parent = QTreeWidgetItem(self.ui.zoneResults, ['H%d (%s/%.3f/%s)'%(house_num+1, scenario.Scenario.dirs[hr[2]], hr[3], hr[5]), 
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
                            
    def openScenario(self):
        scenario_path = '../scenarios'
        settings = QSettings()
        if settings.contains("ScenarioFolder"):
            scenario_path = unicode(settings.value("ScenarioFolder").toString())
        self.filename = QFileDialog.getOpenFileName(self, "Scenarios", scenario_path, "Scenarios (*.csv)")
        if os.path.isfile(self.filename):
            fname = '%s' % (self.filename)
            self.file_load(fname)
            settings.setValue("ScenarioFolder", QVariant(QString(os.path.dirname(fname))))
            
    def saveScenario(self):
        if self.filename is None:
            self.saveAsScenario()
            self.ui.statusbar.showMessage('Saved as file %s' % self.filename)
        else:
            self.updateScenarioFromUI()
            self.fileSave()
            self.ui.statusbar.showMessage('Saved to file %s' % self.filename)
        self.updateUIFromScenario()
        
    def saveAsScenario(self):
        self.updateScenarioFromUI()
        fname = self.filename if self.filename is not None else "."
        fname = unicode(QFileDialog.getSaveFileName(self, "VAWS - Save Scenario", fname, "Scenarios (*.csv)"))
        if len(fname) > 0:
            if "." not in fname:
                fname += ".csv"
            self.filename = fname
            self.fileSave()
            self.updateUIFromScenario()
        
    def set_scenario(self, s=None):
        if s:
            self.s = s

        self.updateUIFromScenario()

        self.statusBar().showMessage('Ready')

    def closeEvent(self, event):
        if self.okToContinue():
            if self.filename is not None and QFile.exists(self.filename):
                settings = QSettings()
                filename = QVariant(QString(self.filename))
                settings.setValue("LastFile", filename)
            self.storeSizePosToSettings()
        else:
            event.ignore()
        
    def updateUIFromScenario(self):
        if self.s is not None:
            self.statusBar().showMessage('Updating', 1000)
            if self.s.house_name:
                self.ui.houseName.setCurrentIndex(self.ui.houseName.findText(self.s.house_name))
            self.ui.debrisRegion.setCurrentIndex(self.ui.debrisRegion.findText(""))
            self.ui.numHouses.setText('%d' % self.s.no_sims)
            self.ui.windMax.setValue(self.s.wind_speed_max)
            self.ui.windMin.setValue(self.s.wind_speed_min)
            self.ui.windSteps.setValue(self.s.wind_speed_steps)
            self.ui.windDirection.setCurrentIndex(self.s.wind_dir_index)
            if self.s.source_items:
                self.ui.sourceItems.setValue(self.s.source_items)
            self.ui.regionalShielding.setText('%f' % (self.s.get_flag('regional_shielding_factor', 0.0)))
            if self.s.building_spacing:
                self.ui.buildingSpacing.setCurrentIndex(
                    self.ui.buildingSpacing.findText('%d' % (self.s.building_spacing)))

            self.ui.seedRandom.setChecked(self.s.get_flag('sample_seed'))
            self.ui.fitLognormalCurve.setChecked(self.s.get_flag('vuln_fit_log'))
            self.ui.waterIngress.setChecked(self.s.get_flag('water_ingress'))
            self.ui.distribution.setChecked(self.s.get_flag('dmg_distribute'))
            self.ui.terrainCategory.setCurrentIndex(self.ui.terrainCategory.findText(self.s.terrain_category))
            self.ui.actionRun.setEnabled(True)
            self.ui.flighttimeMean.setText("{0:.3f}".format(self.s.flight_time_mean))
            self.ui.flighttimeStddev.setText("{0:.3f}".format(self.s.flight_time_stddev))
            self.ui.debrisAngle.setValue(self.s.debris_angle)
            self.ui.debrisRadius.setValue(self.s.debris_radius)
            self.ui.debrisExtension.setText('%f' % self.s.debris_extension)
            self.ui.diffShielding.setChecked(self.s.get_flag('diff_shielding'))
            self.ui.debris.setChecked(self.s.get_flag('debris'))
            self.ui.staggeredDebrisSources.setChecked(self.s.get_flag('debris_staggered_sources'))
            self.ui.redV.setValue(self.s.red_v)
            self.ui.blueV.setValue(self.s.blue_v)
          
            self.ui.constructionEnabled.setChecked(self.s.get_flag('constructionLevels'))
            prob, mf, cf = self.s.getConstructionLevel('low')
            self.ui.lowProb.setValue(float(prob))
            self.ui.lowMean.setValue(float(mf))
            self.ui.lowCov.setValue(float(cf))
            
            prob, mf, cf = self.s.getConstructionLevel('medium')
            self.ui.mediumProb.setValue(float(prob))
            self.ui.mediumMean.setValue(float(mf))
            self.ui.mediumCov.setValue(float(cf))
            
            prob, mf, cf = self.s.getConstructionLevel('high')
            self.ui.highProb.setValue(float(prob))
            self.ui.highMean.setValue(float(mf))
            self.ui.highCov.setValue(float(cf))
            
            self.ui.slight.setValue(self.s.fragility_thresholds.loc['slight', 'threshold'])
            self.ui.medium.setValue(self.s.fragility_thresholds.loc['medium', 'threshold'])
            self.ui.severe.setValue(self.s.fragility_thresholds.loc['severe', 'threshold'])
            self.ui.complete.setValue(self.s.fragility_thresholds.loc['complete', 'threshold'])
            
            self.updateConnectionGroupTable()
                      
        if self.filename:
            self.statusBarScenarioLabel.setText('Scenario: %s' % (os.path.basename(self.filename)))
        else:
            self.statusBarScenarioLabel.setText('Scenario: None')
        
    def updateScenarioFromUI(self):

        new_scenario = self.s
        new_scenario.house_name = (unicode(self.ui.houseName.currentText()))
        new_scenario.set_region_name(unicode(self.ui.debrisRegion.currentText()))
        new_scenario.set_flag('dmg_plot_fragility', True)
        new_scenario.set_flag('dmg_plot_vuln', True)
        new_scenario.set_flag('sample_seed', self.ui.seedRandom.isChecked())
        new_scenario.set_flag('vuln_fit_log', self.ui.fitLognormalCurve.isChecked())
        new_scenario.set_flag('water_ingress', self.ui.waterIngress.isChecked())
        new_scenario.set_flag('dmg_distribute', self.ui.distribution.isChecked())
        new_scenario.set_flag('diff_shielding', self.ui.diffShielding.isChecked())
        new_scenario.set_flag('debris', self.ui.debris.isChecked())
        new_scenario.set_flag('debris_staggered_sources', self.ui.staggeredDebrisSources.isChecked())
        new_scenario.set_flag('construction_levels', self.ui.constructionEnabled.isChecked())
        new_scenario.no_sims = int(self.ui.numHouses.text())
        new_scenario.wind_speed_max = self.ui.windMax.value()
        new_scenario.wind_speed_min = self.ui.windMin.value()
        new_scenario.wind_speed_steps = self.ui.windSteps.value()
        new_scenario.wind_dir_index = self.ui.windDirection.currentIndex()
        new_scenario.terrain_category = unicode(self.ui.terrainCategory.currentText())
        new_scenario.set_flag('regional_shielding_factor', float(unicode(self.ui.regionalShielding.text())))
        new_scenario.source_items = self.ui.sourceItems.value()
        new_scenario.building_spacing = int(unicode(self.ui.buildingSpacing.currentText()))
        new_scenario.debris_radius = self.ui.debrisRadius.value()
        new_scenario.debris_angle = self.ui.debrisAngle.value()
        new_scenario.debris_extension = float(self.ui.debrisExtension.text())
        new_scenario.flight_time_mean = float(unicode(self.ui.flighttimeMean.text()))
        new_scenario.flight_time_stddev = float(unicode(self.ui.flighttimeStddev.text()))

        new_scenario.flight_time_log_mu, new_scenario.flight_time_log_std = \
            stats.compute_logarithmic_mean_stddev(new_scenario.flight_time_mean,
                                                  new_scenario.flight_time_stddev)

        new_scenario.red_v = float(unicode(self.ui.redV.value()))
        new_scenario.blue_v = float(unicode(self.ui.blueV.value()))
        
        new_scenario.setConstructionLevel('low', 
                                float(unicode(self.ui.lowProb.value())), 
                                float(unicode(self.ui.lowMean.value())), 
                                float(unicode(self.ui.lowCov.value())))
        new_scenario.setConstructionLevel('medium', 
                                float(unicode(self.ui.mediumProb.value())), 
                                float(unicode(self.ui.mediumMean.value())), 
                                float(unicode(self.ui.mediumCov.value())))
        new_scenario.setConstructionLevel('high', 
                                float(unicode(self.ui.highProb.value())), 
                                float(unicode(self.ui.highMean.value())), 
                                float(unicode(self.ui.highCov.value())))
        
        new_scenario.fragility_thresholds['slight'] = float(self.ui.slight.value())
        new_scenario.fragility_thresholds['medium'] = float(self.ui.medium.value())
        new_scenario.fragility_thresholds['severe'] = float(self.ui.severe.value())
        new_scenario.fragility_thresholds['complete'] = float(self.ui.complete.value())   
        
        for irow, (index, ctg) in enumerate(self.s.df_groups.iterrows()):
            cellWidget = self.ui.connGroups.cellWidget(irow, 4)
            new_scenario.setOptCTGEnabled(index, True if cellWidget.checkState() == Qt.Checked else False)
            
        self.dirty_scenario = new_scenario != self.s
        if self.dirty_scenario:
            self.set_scenario(new_scenario)
        
    def file_load(self, fname):
        try:
            self.s = scenario.Scenario(fname)
            self.filename = fname
            self.set_scenario(self.s)
        except:
            msg = 'Unable to load previous scenario: %s\nCreating new scenario.' % fname
            QMessageBox.warning(self, "VAWS Program Warning", unicode(msg))

    def fileSave(self):
        self.s.save_config(self.filename)
        
    def fileSaveAs(self):
        self.s.save_config(self.filename)
    
    def okToContinue(self):
        self.updateScenarioFromUI()
        if self.dirty_scenario:
            reply = QMessageBox.question(self,
                            "WindSim - Unsaved Changes",
                            "Save unsaved changes?",
                            QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return False
            elif reply == QMessageBox.Yes:
                self.saveScenario()
        return True
    
    def testDebrisSettings(self):
        self.updateScenarioFromUI()
        mgr = debris.DebrisManager(self.s.house,
                                  self.s.region,
                                  self.s.wind_speed_min, self.s.wind_speed_max, self.s.wind_speed_steps,
                                  self.s.getOpt_DebrisStaggeredSources(),
                                  self.s.debris_radius,
                                  self.s.debris_angle, 
                                  self.s.debris_extension,
                                  self.s.building_spacing,
                                  self.s.source_items,
                                  self.s.flighttime_mean,
                                  self.s.flighttime_stddev)
        
        v, ok = QInputDialog.getInteger(self, "Debris Test", "Wind Speed (m/s):", 50, 10, 200)
        if ok:
            mgr.set_wind_direction_index(self.s.getWindDirIndex())
            mgr.run(v, True)
            mgr.render(v)
    
    def testConstructionLevels(self):
        self.updateScenarioFromUI()
        items = []
        for ctype in self.s.house.conn_types:
            items.append(ctype.connection_type) 
        selected_ctype, ok = QInputDialog.getItem(self, "Construction Test", "Connection Type:", items, 0, False)
        if ok:
            import matplotlib.pyplot as plt
            for ctype in self.s.house.conn_types:
                if ctype.connection_type == selected_ctype:
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    x = []
                    n = 50000
                    for i in xrange(n):
                        level, mean_factor, cov_factor = self.s.sampleConstructionLevel()
                        rv = ctype.sample_strength(mean_factor, cov_factor)
                        x.append(rv)
                        
                    ax.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
                    fig.canvas.draw()
                    plt.show()    
        
def run_gui():
    parser = simulation.process_commandline()

    (options, args) = parser.parse_args()

    if not options.scenario_filename:
        print '\nERROR: Must provide a scenario file to run simulator...\n'
        parser.print_help()
    else:

        if options.output_folder is None:
            path_, _ = os.path.split(sys.argv[0])
            options.output_folder = os.path.abspath(
                os.path.join(path_, '../outputs/output'))
        else:
            options.output_folder = os.path.abspath(
                os.path.join(os.getcwd(), options.output_folder))

        if options.verbose:
            file_logger = os.path.join(options.output_folder, 'log.txt')
            logging.basicConfig(filename=file_logger,
                                filemode='w',
                                level=logging.DEBUG,
                                format='%(levelname)s %(message)s')

        conf = scenario.Scenario(cfg_file=options.scenario_filename,
                                 output_path=options.output_folder)

        app = QApplication(sys.argv)
        app.setOrganizationName("Geoscience Australia")
        app.setOrganizationDomain("ga.gov.au")
        app.setApplicationName("WindSim")
        img = QPixmap()

        path_ = '/'.join(__file__.split('/')[:-1])
        img_file = os.path.join(path_, 'images/splash/splash.png')
        if not img.load(img_file):
            raise Exception('Could not load splash image')

        app.processEvents()

        global my_app
        my_app = MyForm(init_scenario=conf)

        splash = QSplashScreen(my_app, img)
        splash.show()
        my_app.show()
        import time
        time.sleep(5)
        splash.finish(my_app)
        sys.exit(app.exec_())


if __name__ == '__main__':
    run_gui()



