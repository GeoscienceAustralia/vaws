"""
    Import Module - import all our CSV data into a fresh database
        - should only run as part of 'packaging' process.
        - harvest house types found within subfolders
"""
import os.path
import sys
import datetime
import pandas as pd

import database
import debris
import house


def import_house(arg, dirname, names):
    if dirname.find('.svn') == -1:
        if 'house_data.csv' in names:
            print 'Importing House from folder: %s' % dirname
            house.importDataFromPath(dirname)
            

def loadTerrainProfileFromCSV(fileBase, tcat):
    db = database.db
    fileName = fileBase + tcat + '.csv'
    x = pd.read_csv(fileName, skiprows=1, header=None)
    for _, row in x.iterrows():
        for i, value in enumerate(row[1:], 1):
            ins = db.terrain_table.insert().values(tcat=tcat,
                                                   profile=i,
                                                   z=row[0],
                                                   m=value)
            db.session.execute(ins)
    db.session.commit()
    

def loadTerrainProfilesFrom(path):
    base = path + 'mzcat_terrain_'
    loadTerrainProfileFromCSV(base, '2')
    loadTerrainProfileFromCSV(base, '2.5')
    loadTerrainProfileFromCSV(base, '3')
    loadTerrainProfileFromCSV(base, '5')
    

def loadDebrisTypes(fileName):
    db = database.db
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        ins = db.debris_types_table.insert().values(name=row[0], cdav=row[1])
        db.session.execute(ins)
    db.session.commit()
    

def loadDebrisRegions(fileName):
    x = pd.read_csv(fileName)

    for _, row in x.iterrows():

        tmp = debris.DebrisRegion(name=row[0],
                                  cr=row[1],
                                  cmm=row[2],
                                  cmc=row[3],
                                  cfm=row[4],
                                  cfc=row[5],
                                  rr=row[6],
                                  rmm=row[7],
                                  rmc=row[8],
                                  rfm=row[9],
                                  rfc=row[10],
                                  pr=row[11],
                                  pmm=row[12],
                                  pmc=row[13],
                                  pfm=row[14],
                                  pfc=row[15],
                                  alpha=row[16],
                                  beta=row[17])

        database.db.session.add(tmp)
    database.db.session.commit()
    

def import_model(base_path, model_database, verbose=False):
    date_run = datetime.datetime.now()
    print 'Current Path: %s' % (os.getcwd())
    print 'Importing Wind Vulnerability Model Data into database from folder: %s' %(base_path)
    
    database.db.drop_tables()
    database.db.create_tables()
    print 'created new tables...'
    
    loadTerrainProfilesFrom(base_path)
    print 'imported terrain profiles'
    
    loadDebrisTypes(base_path + 'debris_types.csv')
    print 'imported debris types'
    
    loadDebrisRegions(base_path + 'debris_regions.csv')
    print 'imported debris regions'
    
    print 'Enumerating house-type subfolders...'
    os.path.walk(base_path + '/houses', import_house, None)
    
    #h = house.queryHouseWithName('Group 4 House')
    print 'Database has been imported in: %s' % (datetime.datetime.now() - date_run)
        
 
# unit tests
if __name__ == '__main__':
    def debugexcept(type, value, tb):
        if hasattr(sys, 'ps1') or not (sys.stderr.isatty() and sys.stdin.isatty()) or type == SyntaxError:
            sys.__excepthook__(type, value, tb)
        else:
            import traceback, pdb
            traceback.print_exception(type, value, tb)
            print
            pdb.pm()
            
    sys.excepthook = debugexcept
    
    import_model('../../data/', '../model.db')
