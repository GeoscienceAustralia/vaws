'''
    output.py - output module, postprocess and plot display engine
'''
from matplotlib import cm, mpl
from vaws import version, zone, house
mplDict = None
        

def hookupWidget(mpldict):
    global mplDict
    mplDict = mpldict
    mplDict['vulnerability'].axes.hold(True)
    mplDict['fragility'].axes.hold(True)

class PlotFlyoverCallback(object):
    def __init__(self, axes, data, statusbar, num_cols, num_rows):
        self.axes = axes
        self.data = data
        self.statusbar = statusbar
        self.num_rows = num_rows
        self.num_cols = num_cols
        
    def __call__(self, event):
        if event.inaxes != self.axes: 
            return
        col = int(event.xdata+0.5)
        if col > self.num_cols-1: 
            return
        row = int(event.ydata+0.5)
        if row > self.num_rows-1: 
            return
        locStr = 'Average damage @ %.3f m/sec for Zone %s' % (self.data[row,col], zone.getZoneLocFromGrid(col, row))
        self.statusbar.showMessage(locStr)

class PlotClickCallback(object):
    def __init__(self, figure, axes, plotKey, owner, aHouse, walldir=None):
        self.figure = figure
        self.axes = axes
        self.plotKey = plotKey
        self.house = aHouse
        self.owner = owner
        self.walldir = walldir
        self.selected, = axes.plot([0], [0], 'ws', ms=12, alpha=1.0, visible=False, markeredgecolor='black', markeredgewidth=2)
        
    def select(self, col, row):
        self.selected.set_visible(True)
        self.selected.set_data(col, row)
        self.figure.canvas.draw()
        
    def __call__(self, event):
        if event.inaxes != self.axes: 
            return
        
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        
        mplDict['south'].selected.set_visible(False)
        mplDict['north'].selected.set_visible(False)
        mplDict['east'].selected.set_visible(False)
        mplDict['west'].selected.set_visible(False)
        
        self.select(col, row)
    
        zoneLoc = zone.getZoneLocFromGrid(col, row)
        
        if self.walldir is not None:
            if self.walldir == 'south':
                zoneLoc = 'WS%s' % (zoneLoc)
            elif self.walldir == 'north':
                zoneLoc = 'WN%s' % (zoneLoc)
            elif self.walldir == 'east':
                zoneLoc = 'WE%d%d' % (col+2, row+1)
            elif self.walldir == 'west':
                zoneLoc = 'WW%d%d' % (col+2, row+1)
        
        self.owner.onSelectZone(zoneLoc)
        
        if zoneLoc in house.zoneByLocationMap:    
            z = house.zoneByLocationMap[zoneLoc]         
            self.owner.onZoneSelected(z, self.plotKey)
   
            ctg = house.ctgMap[self.plotKey]
            for c in z.located_conns:
                if c.ctype.group.id == ctg.id:
                    self.owner.onSelectConnection(c.connection_name)
                    break
            

def plot_wall_damage_show(plotKey, 
                          wall_south_V, 
                          wall_north_V,
                          wall_west_V,  
                          wall_east_V, 
                          num_major_cols, num_major_rows,
                          num_minor_cols, num_minor_rows, 
                          v_min, v_max):    

    # add the legend colorbar axes to base of plot
    fig = mplDict[plotKey].figure
    fig.clf()
    fig.canvas.draw()
    left = 0.1
    bottom = left
    width = (1.0 - left*2.0)
    height = 0.05
    axLegend = fig.add_axes([left, bottom, width, height])
    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    cb1 = mpl.colorbar.ColorbarBase(axLegend, cmap=cmap, norm=norm, orientation='horizontal')
    cb1.set_label('Wind Speed')

    # setup south ticks
    xticks = range(0, num_major_cols)
    xticklabels = []
    for tick in xticks:
        base = 'WS'
        letter = chr(ord('A')+tick)
        xticklabels.append(base + letter)
    yticks = range(0, num_major_rows)
    yticklabels = range(1, num_major_rows+1)

    # add the south wall
    bottom = 0.2
    height = 0.1
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(wall_south_V, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    mplDict['south'] = PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house'], 'south')
    fig.canvas.mpl_connect('motion_notify_event', PlotFlyoverCallback(axPlot, wall_south_V, mplDict['statusbar'], num_major_cols, num_major_rows))
    fig.canvas.mpl_connect('button_press_event', mplDict['south'])
    
    # setup north ticks
    xticklabels = []
    for tick in xticks:
        base = 'WN'
        letter = chr(ord('A')+tick)
        xticklabels.append(base + letter)
        
    # add the north wall
    bottom = 0.8
    height = 0.1
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(wall_north_V, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    mplDict['north'] = PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house'], 'north')
    fig.canvas.mpl_connect('motion_notify_event', PlotFlyoverCallback(axPlot, wall_north_V, mplDict['statusbar'], num_major_cols, num_major_rows))
    fig.canvas.mpl_connect('button_press_event', mplDict['north'])
    
    # setup west ticks
    xticks = range(0, num_minor_cols)
    xticklabels = []
    for tick in xticks:
        xticklabels.append('WW%d' % (tick+2))
    yticks = range(0, num_minor_rows)
    yticklabels = range(1, num_minor_rows+1)

    # add the west wall
    bottom = 0.4
    height = 0.1
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(wall_west_V, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    mplDict['west'] = PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house'], 'west')
    fig.canvas.mpl_connect('motion_notify_event', PlotFlyoverCallback(axPlot, wall_west_V, mplDict['statusbar'], num_minor_cols, num_minor_rows))
    fig.canvas.mpl_connect('button_press_event', mplDict['west'])
    
    # setup east ticks
    xticklabels = []
    for tick in xticks:
        xticklabels.append('WE%d' % (tick+2))
    
    # add the east wall
    bottom = 0.6
    height = 0.1
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(wall_east_V, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    mplDict['east'] = PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house'], 'east')
    fig.canvas.mpl_connect('motion_notify_event', PlotFlyoverCallback(axPlot, wall_east_V, mplDict['statusbar'], num_minor_cols, num_minor_rows))
    fig.canvas.mpl_connect('button_press_event', mplDict['east'])
    
    fig.canvas.draw()    
    

def plot_damage_show(plotKey, v_damaged_at, numCols, numRows, v_min, v_max):    
    xticks = range(0, numCols)
    xticklabels = []
    for tick in xticks:
        xticklabels.append(chr(ord('A')+tick))
    yticks = range(0, numRows)
    yticklabels = range(1, numRows+1)

    # add the legend colorbar axes
    fig = mplDict[plotKey].figure
    fig.clf()
    fig.canvas.draw()
    left = 0.1
    bottom = left
    width = (1.0 - left*2.0)
    height = 0.05
    axLegend = fig.add_axes([left, bottom, width, height])
    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    cb1 = mpl.colorbar.ColorbarBase(axLegend, cmap=cmap, norm=norm, orientation='horizontal')
    cb1.set_label('Wind Speed')

    # add the heatmap
    left = 0.1
    bottom = 0.2
    width = (1.0 - left*2.0)
    height = 0.75
    axPlot = fig.add_axes([left, bottom, width, height])
    axPlot.imshow(v_damaged_at, cmap=cm.jet_r, interpolation='nearest', origin='lower', vmin=v_min, vmax=v_max)
    axPlot.set_yticks(yticks)
    axPlot.set_yticklabels(yticklabels)
    axPlot.set_xticks(xticks)
    axPlot.set_xticklabels(xticklabels)
    axPlot.set_title('Wind Speed Damaged At Heatmap')
    axPlot.format_coord = format_coord
    
    fig.canvas.mpl_connect('motion_notify_event', PlotFlyoverCallback(axPlot, v_damaged_at, mplDict['statusbar'], numCols, numRows))
    fig.canvas.mpl_connect('button_press_event', PlotClickCallback(fig, axPlot, plotKey, mplDict['owner'], mplDict['house']))
    fig.canvas.draw()    


def plot_wind_event_damage(v, di):
    mplDict['vulnerability'].axes.scatter(v, di, s=8, marker='+', label='_nolegend_')


def plot_wind_event_mean(v, di):
    mplDict['vulnerability'].axes.scatter(v, di, s=20, c='r', marker='o', label="Means")


def plot_fitted_curve(v, di, label="Fitted Curve", alpha=1.0, col='b'):
    mplDict['vulnerability'].axes.plot(v, di, label=label, alpha=alpha, c=col)


def plot_model_curve(v, di, label="Model Curve"):
    mplDict['vulnerability'].axes.plot(v, di, label=label)


def plot_wind_event_show(num_iters, Vmin, Vmax, output_folder):
    mplDict['vulnerability'].axes.set_title('Vulnerability Curve (n = %d)' % (num_iters))
    mplDict['vulnerability'].axes.set_xlabel('Impact Wind speed (m/s)')
    mplDict['vulnerability'].axes.set_ylabel('Damage Index')
    mplDict['vulnerability'].axes.set_xlim((Vmin, Vmax))
    mplDict['vulnerability'].axes.set_ylim((0.0, 1.1))
    mplDict['vulnerability'].axes.figure.canvas.draw()
    

def plot_fragility_curve(v, fl, label, alpha=1.0, col='b'):
    mplDict['fragility'].axes.plot(v, fl, label=label, c=col, alpha=alpha)


def plot_fragility_show(num_iters, Vmin, Vmax, output_folder):
    mplDict['fragility'].axes.set_title('Fragility Curve (n = %d)' % num_iters)
    mplDict['fragility'].axes.set_xlabel('Impact Wind speed (m/s)')
    mplDict['fragility'].axes.set_ylabel('Probability of Damage State')
    mplDict['fragility'].axes.set_xlim((Vmin, Vmax))
    mplDict['fragility'].axes.set_ylim((0.0, 1.0))
    legend = mplDict['fragility'].axes.legend(loc=2, fancybox=True, shadow=True)
    for ltxt in legend.get_texts():
        ltxt.set_fontsize('small')
    mplDict['fragility'].axes.figure.canvas.draw()


def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    return zone.getZoneLocFromGrid(col, row)
