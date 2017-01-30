"""
    Import Module - import all our CSV data into a fresh database
        - should only run as part of 'packaging' process.
        - harvest house types found within subfolders
"""
import datetime
import logging
import pandas as pd
from os.path import join

from database import House, Patch, Connection, ConnectionTypeGroup, \
    ConnectionType, DamageFactoring, DamageCosting, Influence, \
    WaterIngressCosting, Wall, Coverage, CoverageType, Zone


def loadStructurePatchesFromCSV(path, db):
    """
    Load structural influence patches CSV -
    format is: damaged_conn, target_conn, zone, infl, zone, infl,....
    """
    fileName = join(path, 'influencefactorpatches.csv')
    logging.info('read {}'.format(fileName))

    lineCount = 0
    for line in open(fileName, 'r'):
        if lineCount != 0:
            damagedConn = None
            targetConn = None
            zone = None
            line = line.rstrip()
            fields = line.strip().split(",")
            for col, data in enumerate(fields):
                if len(data) == 0:  # excel sometimes puts extra ,,,, in when staggered cols
                    break
                if col == 0:
                    damagedConn = db.get_conn_by_name(data)
                elif col == 1:
                    targetConn = db.get_conn_by_name(data)
                elif (col % 2) == 0:
                    zone = db.get_zone_by_name(data)
                elif damagedConn is not None and targetConn is not None and zone is not None:
                    ins = Patch(damaged_connection_id=damagedConn.id,
                                target_connection_id=targetConn.id,
                                zone_id=zone.id,
                                coeff=float(data))
                    db.session.add(ins)
                    zone = None
        lineCount += 1
    db.session.commit()


def loadHouseFromCSV(path, db):
    fileName = join(path, 'house_data.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName).iloc[0]
    house = House(house_name=x[0],
                  replace_cost=float(x[1]),
                  height=float(x[2]),
                  cpe_V=float(x[3]),
                  cpe_k=float(x[4]),
                  cpe_struct_V=float(x[5]),
                  length=float(x[6]),
                  width=float(x[7]),
                  roof_columns=int(x[8]),
                  roof_rows=int(x[9]))
    db.session.add(house)
    db.session.commit()
    return house


def loadConnectionTypeGroupsFromCSV(path, house, db):
    fileName = join(path, 'conn_group.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = ConnectionTypeGroup(
            group_name=row[0],
            distribution_order=int(row[1]),
            distribution_direction=row[2],
            trigger_collapse_at=float(row[4]),
            patch_distribution=int(row[5]),
            set_zone_to_zero=int(row[6]),
            water_ingress_order=int(row[7]))
        tmp.costing = db.get_costing_by_name(row[3])
        house.conn_type_groups.append(tmp)
    db.session.commit()


def loadConnectionTypesFromCSV(path, house, db):
    fileName = join(path, 'conn_type.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = ConnectionType(row[0],
                             float(row[6]),
                             float(row[1]),
                             float(row[2]),
                             float(row[3]),
                             float(row[4]))
        tmp.group = db.get_conn_type_group_by_name(row[5])
        house.conn_types.append(tmp)
    db.session.commit()


def loadConnectionsFromCSV(path, house, db):
    fileName = join(path, 'connections.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = Connection(connection_name=row[0], edge=int(row[3]))
        tmp.ctype = db.get_conn_type_by_name(row[1])
        tmp.zone_id = db.get_zone_by_name(row[2]).id
        house.connections.append(tmp)
    db.session.commit()


def loadDamageCostingsFromCSV(path, house, db):
    fileName = join(path, 'damage_costing_data.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = DamageCosting(
            costing_name=row[0],
            area=float(row[1]),
            envelope_factor_formula_type=int(row[3]),
            envelope_repair_rate=float(row[2]),
            env_coeff_1=float(row[4]),
            env_coeff_2=float(row[5]),
            env_coeff_3=float(row[6]),
            internal_factor_formula_type=int(row[8]),
            internal_repair_rate=float(row[7]),
            int_coeff_1=float(row[9]),
            int_coeff_2=float(row[10]),
            int_coeff_3=float(row[11]))
        house.costings.append(tmp)
    db.session.commit()


def loadDamageFactoringsFromCSV(path, house, db):
    fileName = join(path, 'damage_factorings.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        parent_id = 0
        factor_id = 0
        for ctg in house.conn_type_groups:
            if ctg.group_name == row[0]:
                parent_id = ctg.id
            if ctg.group_name == row[1]:
                factor_id = ctg.id
        if parent_id == 0:
            raise LookupError(
                'Invalid connection group name given: {}'.format(row[0]))
        if factor_id == 0:
            raise LookupError(
                'Invalid connection group name given: {}'.format(row[1]))
        tmp = DamageFactoring(parent_id=parent_id,
                              factor_id=factor_id)
        house.factorings.append(tmp)
    db.session.commit()


def loadWaterCostingsFromCSV(path, house, db):
    fileName = join(path, 'water_ingress_costing_data.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = WaterIngressCosting(name=row[0],
                                  wi=float(row[1]),
                                  base_cost=float(row[2]),
                                  formula_type=int(row[3]),
                                  coeff1=float(row[4]),
                                  coeff2=float(row[5]),
                                  coeff3=float(row[6]))
        tmp.house_id = house.id
        db.session.add(tmp)
    db.session.commit()


def loadZoneFromCSV(path, house, db):
    fileName = join(path, 'zones.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = Zone(zone_name=row[0],
                   zone_area=float(row[1]),
                   coeff_N=float(row[6]),
                   coeff_NE=float(row[7]),
                   coeff_E=float(row[8]),
                   coeff_SE=float(row[9]),
                   coeff_S=float(row[2]),
                   coeff_SW=float(row[3]),
                   coeff_W=float(row[4]),
                   coeff_NW=float(row[5]),
                   struct_coeff_N=float(row[14]),
                   struct_coeff_NE=float(row[15]),
                   struct_coeff_E=float(row[16]),
                   struct_coeff_SE=float(row[17]),
                   struct_coeff_S=float(row[10]),
                   struct_coeff_SW=float(row[11]),
                   struct_coeff_W=float(row[12]),
                   struct_coeff_NW=float(row[13]),
                   eaves_coeff_N=float(row[22]),
                   eaves_coeff_NE=float(row[23]),
                   eaves_coeff_E=float(row[24]),
                   eaves_coeff_SE=float(row[25]),
                   eaves_coeff_S=float(row[18]),
                   eaves_coeff_SW=float(row[19]),
                   eaves_coeff_W=float(row[20]),
                   eaves_coeff_NW=float(row[21]),
                   leading_roof_N=int(row[30]),
                   leading_roof_NE=int(row[31]),
                   leading_roof_E=int(row[32]),
                   leading_roof_SE=int(row[33]),
                   leading_roof_S=int(row[26]),
                   leading_roof_SW=int(row[27]),
                   leading_roof_W=int(row[28]),
                   leading_roof_NW=int(row[29]),
                   cpi_alpha=float(row[34]),
                   wall_dir=int(row[35]))
        house.zones.append(tmp)
    db.session.commit()


def loadConnectionInfluencesFromCSV(path, db):
    fileName = join(path, 'connectioninfluences.csv')
    logging.info('read {}'.format(fileName))

    # input: connection_name, zone1_name, zone1_infl, (.....)
    lineCount = 0
    for line in open(fileName, 'r'):
        if lineCount != 0:
            parentLeft = None
            childRight = None
            line = line.rstrip()
            fields = line.strip().split(",")
            for col, data in enumerate(fields):
                if len(data) == 0:  # excel throws extra comma's in
                    break
                if col == 0:
                    parentLeft = db.get_conn_by_name(data)
                elif (col % 2) != 0:
                    childRight = data
                elif childRight is not None and parentLeft is not None:
                    inf = Influence(id=childRight, coeff=float(data))
                    # print 'adding influence from %s to %s of %f' % (
                    # parentLeft, childRight, float(data))
                    parentLeft.influences.append(inf)
                    childRight = None
        lineCount += 1
    db.session.commit()


def loadWallsFromCSV(path, house, db):
    fileName = join(path, 'walls.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = Wall(direction=int(row[0]), area=float(row[1]))
        house.walls.append(tmp)
    db.session.commit()


def loadCoverageTypesFromCSV(path, house, db):
    fileName = join(path, 'cov_types.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = CoverageType(name=row[0],
                           failure_momentum_mean=float(row[1]),
                           failure_momentum_stddev=float(row[2]))
        house.cov_types.append(tmp)
    db.session.commit()


def loadCoveragesFromCSV(path, house, db):
    fileName = join(path, 'coverages.csv')
    logging.info('read {}'.format(fileName))

    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = Coverage(description=row[0], area=float(row[2]))
        for covt in house.cov_types:
            if covt.name == row[3]:
                tmp.type = covt
                break
        for wall in house.walls:
            if wall.direction == int(row[1]):
                tmp.wall_id = wall.id
                wall.coverages.append(tmp)
                break
    db.session.commit()


def import_model(path, db):
    date_run = datetime.datetime.now()
    # print 'Current Path: %s' % (os.getcwd())
    logging.info('Importing data into database from folder {}'.format(path))
    
    db.drop_tables()
    db.create_tables()
    logging.info('create new tables')

    house_ = loadHouseFromCSV(path, db)

    loadWallsFromCSV(path, house_, db)

    loadCoverageTypesFromCSV(path, house_, db)

    loadCoveragesFromCSV(path, house_, db)

    loadDamageCostingsFromCSV(path, house_, db)

    loadWaterCostingsFromCSV(path, house_, db)

    loadConnectionTypeGroupsFromCSV(path, house_, db)

    loadConnectionTypesFromCSV(path, house_, db)

    loadDamageFactoringsFromCSV(path, house_, db)

    loadZoneFromCSV(path, house_, db)

    loadConnectionsFromCSV(path, house_, db)

    loadConnectionInfluencesFromCSV(path, db)

    loadStructurePatchesFromCSV(path, db)

    logging.info('Database has been imported in: {}'.format(
        datetime.datetime.now() - date_run))
