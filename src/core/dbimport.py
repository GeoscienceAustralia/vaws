"""
    Import Module - import all our CSV data into a fresh database
        - should only run as part of 'packaging' process.
        - harvest house types found within subfolders
"""
import os.path
import sys
import datetime
import pandas as pd

import debris
import house
import database


def import_house(path, db):
    for path_, _, list_files in os.walk(path):

        if 'house_data.csv' in list_files and 'ignore' not in path_:
            house.importDataFromPath(path_, db)


def loadTerrainProfiles(path, db):
    for tcat in ['2', '2.5', '3', '5']:
        fileName = os.path.join(path, 'mzcat_terrain_' + tcat + '.csv')
        x = pd.read_csv(fileName, skiprows=1, header=None)
        for _, row in x.iterrows():
            for i, value in enumerate(row[1:], 1):
                ins = database.Terrain(tcat=tcat,
                                       profile=i,
                                       z=int(row[0]),
                                       m=float(value))
                db.session.add(ins)
        db.session.commit()


def loadDebrisTypes(path, db):
    fileName = os.path.join(path, 'debris_types.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        ins = database.DebrisType(name=row[0], cdav=float(row[1]))
        db.session.add(ins)
    db.session.commit()
    

def loadDebrisRegions(path, db):
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

        db.session.add(tmp)
    db.session.commit()
    

def import_model(base_path, db):
    date_run = datetime.datetime.now()
    # print 'Current Path: %s' % (os.getcwd())
    print 'Importing Wind Vulnerability Model Data into database' \
          'from folder: {}'.format(base_path)
    
    db.drop_tables()
    db.create_tables()
    print 'created new tables...'
    
    loadTerrainProfiles(base_path, db)
    print 'imported terrain profiles'
    
    loadDebrisTypes(base_path, db)
    print 'imported debris types'
    
    loadDebrisRegions(base_path, db)
    print 'imported debris regions'
    
    import_house(base_path, db)
    
    #h = house.queryHouseWithName('Group 4 House')
    print 'Database has been imported in: {}'.format(datetime.datetime.now()
                                                     - date_run)
        
 
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
