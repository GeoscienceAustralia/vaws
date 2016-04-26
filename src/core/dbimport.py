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
    # if dirname.find('.svn') == -1:
    if 'house_data.csv' in names:
        print 'Importing House from folder: %s' % dirname
        house.importDataFromPath(dirname)


def loadTerrainProfiles(path):
    for tcat in ['2', '2.5', '3', '5']:
        db = database.db
        fileName = os.path.join(path, 'mzcat_terrain_' + tcat + '.csv')
        x = pd.read_csv(fileName, skiprows=1, header=None)
        for _, row in x.iterrows():
            for i, value in enumerate(row[1:], 1):
                ins = db.terrain_table.insert().values(tcat=tcat,
                                                       profile=i,
                                                       z=int(row[0]),
                                                       m=float(value))
                db.session.execute(ins)
        db.session.commit()


def loadDebrisTypes(path):
    db = database.db
    fileName = os.path.join(path, 'debris_types.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        ins = db.debris_types_table.insert().values(name=row[0],
                                                    cdav=float(row[1]))
        db.session.execute(ins)
    db.session.commit()
    

def loadDebrisRegions(path):
    fileName = os.path.join(path, 'debris_regions.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = debris.DebrisRegion(name=row[0],
                                  cr=float(row[1]),
                                  cmm=float(row[2]),
                                  cmc=float(row[3]),
                                  cfm=float(row[4]),
                                  cfc=float(row[5]),
                                  rr=float(row[6]),
                                  rmm=float(row[7]),
                                  rmc=float(row[8]),
                                  rfm=float(row[9]),
                                  rfc=float(row[10]),
                                  pr=float(row[11]),
                                  pmm=float(row[12]),
                                  pmc=float(row[13]),
                                  pfm=float(row[14]),
                                  pfc=float(row[15]),
                                  alpha=float(row[16]),
                                  beta=float(row[17]))

        database.db.session.add(tmp)
    database.db.session.commit()
    

def import_model(base_path, model_database, verbose=False):
    date_run = datetime.datetime.now()
    # print 'Current Path: %s' % (os.getcwd())
    print 'Importing Wind Vulnerability Model Data into database' \
          'from folder: {}'.format(base_path)
    
    database.db.drop_tables()
    database.db.create_tables()
    print 'created new tables...'
    
    loadTerrainProfiles(base_path)
    print 'imported terrain profiles'
    
    loadDebrisTypes(base_path)
    print 'imported debris types'
    
    loadDebrisRegions(base_path)
    print 'imported debris regions'
    
    print 'Enumerating house-type subfolders...'
    os.path.walk(os.path.join(base_path,'houses'), import_house, None)
    
    #h = house.queryHouseWithName('Group 4 House')
    print 'Database has been imported in: %s' % (datetime.datetime.now() - date_run)
        
 
# unit tests
# if __name__ == '__main__':
#     def debugexcept(type, value, tb):
#         if hasattr(sys, 'ps1') or not (sys.stderr.isatty() and sys.stdin.isatty()) or type == SyntaxError:
#             sys.__excepthook__(type, value, tb)
#         else:
#             import traceback, pdb
#             traceback.print_exception(type, value, tb)
#             print
#             pdb.pm()
#
#     sys.excepthook = debugexcept
#
#     import_model('../../data/', '../model.db')
