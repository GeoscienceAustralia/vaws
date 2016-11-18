"""
    House Module - reference storage for House type information.
        - loaded from database
        - imported from '../data/houses/subfolder' (so we need python constr)
"""
import os
from sqlalchemy import Integer, String, Float, Column, ForeignKey
from sqlalchemy.orm import relation
import pandas as pd

import database
import connection_type
import connection
import zone
import influence
import damage_costing
import wateringress

zoneByLocationMap = {}
zoneByIDMap = {}
ctgMap = {}
connByZoneTypeMap = {}
connByTypeGroupMap = {}
connByTypeMap = {}
connByIDMap = {}
connByNameMap = {}
inflZonesByConn = {}  # map conn.id --> map zone --> infl
wallByDirMap = {}


class Wall(database.Base):
    __tablename__ = 'walls'
    id = Column(Integer, primary_key=True)
    direction = Column(Integer)
    area = Column(Float)
    house_id = Column(Integer, ForeignKey('houses.id'))
    coverages = relation('Coverage')

    def __repr__(self):
        return 'Wall(dir: %d, area: %.3f): ' % (self.direction, self.area)


class CoverageType(database.Base):
    __tablename__ = 'coverage_types'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    failure_momentum_mean = Column(Float)
    failure_momentum_stddev = Column(Float)
    house_id = Column(Integer, ForeignKey('houses.id'))

    def __repr__(self):
        return 'CoverageType(name: %s): ' % (self.name)


class Coverage(database.Base):
    __tablename__ = 'coverages'
    id = Column(Integer, primary_key=True)
    description = Column(String)
    area = Column(Float)
    wall_id = Column(Integer, ForeignKey('walls.id'))
    type_id = Column(Integer, ForeignKey('coverage_types.id'), nullable=False)
    type = relation(CoverageType, uselist=False)

    def __repr__(self):
        return 'Coverage(desc: %s, area: %.3f)' % (self.description, self.area)


class House(database.Base):

    __tablename__ = 'houses'
    id = Column(Integer, primary_key=True)
    house_name = Column(String)
    replace_cost = Column(Float)
    height = Column(Float)

    # Cpe (external pressure coefficient) for roof assumed to
    # follow Type III GEV with V (cov) and k (shape factor)
    cpe_V = Column(Float)
    cpe_k = Column(Float)
    # Cpe for structure. Only cov is provided assuming k is the same as roof
    cpe_struct_V = Column(Float)
    length = Column(Float)
    width = Column(Float)
    roof_columns = Column(Integer)
    roof_rows = Column(Integer)
    walls = relation(Wall)
    cov_types = relation(CoverageType)
    costings = relation(damage_costing.DamageCosting)
    factorings = relation(damage_costing.DamageFactoring)
    conn_types = relation(connection_type.ConnectionType)
    conn_type_groups = relation(connection_type.ConnectionTypeGroup,
                                order_by=connection_type.ConnectionTypeGroup.distribution_order)
    water_groups = relation(connection_type.ConnectionTypeGroup,
                            order_by=connection_type.ConnectionTypeGroup.water_ingress_order)
    zones = relation(zone.Zone)
    connections = relation(connection.Connection)

    def __repr__(self):
        return '(%s, $%.2f)' % (self.house_name, self.replace_cost)

    def getConnectionByName(self, connName):
        for conn in self.connections:
            if conn.connection_name == connName:
                return conn
        raise LookupError('Invalid connName: %s' % connName)

    def resetCoverages(self):
        for wall in self.walls:
            wall.result_perc_collapsed = 0
            for cov in wall.coverages:
                cov.result_intact = True
                cov.result_num_impacts = 0

    def resetZoneInfluences(self):
        for c in self.connections:
            inflZonesByConn[c] = {}
            for infl in c.zones:
                inflZonesByConn[c][infl.zone] = infl.coeff

    def reset_connection_failure(self):
        for c in self.connections:
            c.result_failure_v = 0.0
            c.result_failure_v_i = 0

    def reset_results(self):
        self.resetZoneInfluences()
        self.resetCoverages()
        for c in self.connections:
            c.reset_results()
        for ct in self.conn_types:
            ct.reset_results()

    def getZoneInfluencesForConn(self, conn):
        return inflZonesByConn[conn]

    def getZoneByName(self, zoneName):
        for z in self.zones:
            if z.zone_name == zoneName:
                return z
        raise LookupError('Invalid zoneName: %s' % zoneName)

    def getZoneByWallDirection(self, wall_dir):
        arr = []
        for z in self.zones:
            if z.wall_dir == wall_dir:
                arr.append(z)
        return arr

    def qryZoneByWallDirection(self, wall_dir, db):
        return db.session.query(zone.Zone).filter(
            zone.Zone.house_id == self.id).filter(
            zone.Zone.wall_dir == wall_dir).all()

    def getConnTypeByName(self, ctName):
        for ct in self.conn_types:
            if ct.connection_type == ctName:
                return ct
        raise LookupError('Invalid ctName: %s' % ctName)

    def getConnTypeGroupByName(self, ctgName):
        for group in self.conn_type_groups:
            if group.group_name == ctgName:
                return group
        raise LookupError('Invalid ctgName: %s' % ctgName)

    def getCostingByName(self, costingName):
        for dmg in self.costings:
            if dmg.costing_name == costingName:
                return dmg
        raise LookupError('Invalid costingName: %s' % costingName)

    def getWallArea(self):
        area = 0
        for wall in self.walls:
            area += wall.area
        return area

    def getWallByDirection(self, wind_dir):
        for wall in self.walls:
            if wall.direction == wind_dir:
                return wall
        raise LookupError('Invalid wind_dir: %d' % wind_dir)


def loadStructurePatchesFromCSV(path, house, db):
    """
    Load structural influence patches CSV -
    format is: damaged_conn, target_conn, zone, infl, zone, infl,....
    """
    fileName = os.path.join(path, 'influencefactorpatches.csv')
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
                    damagedConn = house.getConnectionByName(data)
                elif col == 1:
                    targetConn = house.getConnectionByName(data)
                elif (col % 2) == 0:
                    zone = house.getZoneByName(data)
                elif damagedConn is not None and targetConn is not None and zone is not None:
                    ins = database.Patch(damaged_connection_id=damagedConn.id,
                                         target_connection_id=targetConn.id,
                                         zone_id=zone.id,
                                         coeff=float(data))
                    db.session.add(ins)
                    zone = None
        lineCount += 1
    db.session.commit()


def queryHouses(db):
    return db.session.query(House.house_name, House.replace_cost,
                                     House.height).all()


def queryHouseWithName(hn, db):
    """
    find (and select as current) the given house name
    Args:
        hn: house name
        db: instance of DatabaseManager

    Returns:
    """

    house = db.session.query(House).filter_by(house_name=hn).one()
    house.reset_results()

    connByTypeMap.clear()
    for ct in house.conn_types:
        conns = []
        for c in ct.connections_of_type:
            conns.append(c)
        connByTypeMap[ct] = conns

    zoneByLocationMap.clear()
    zoneByIDMap.clear()
    for z in house.zones:
        zoneByLocationMap[z.zone_name] = z
        zoneByIDMap[z.id] = z

    ctgMap.clear()
    connByTypeGroupMap.clear()
    for ctg in house.conn_type_groups:
        # groups start out enabled unless the order is negative
        # (this can be overridden later by scenario flags)
        if ctg.distribution_order >= 0:
            ctg.enabled = True
        else:
            ctg.enabled = False
        ctgMap[ctg.group_name] = ctg
        connByTypeGroupMap[ctg.group_name] = []

    connByZoneTypeMap.clear()
    for zone in house.zones:
        zoneConns = {}
        for c in zone.located_conns:
            zoneConns[c.ctype.group.group_name] = c
        connByZoneTypeMap[zone.zone_name] = zoneConns

    for c in house.connections:
        connByTypeGroupMap[c.ctype.group.group_name].append(c)
        connByIDMap[c.id] = c
        connByNameMap[c.connection_name] = c

    for wall in house.walls:
        wallByDirMap[wall.direction] = wall

    house.resetZoneInfluences()
    house.resetCoverages()
    wateringress.populate_water_costs(db, house.id)
    return house


def loadFromCSV(path, db):
    fileName = os.path.join(path, 'house_data.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = House(house_name=row[0],
                    replace_cost=float(row[1]),
                    height=float(row[2]),
                    cpe_V=float(row[3]),
                    cpe_k=float(row[4]),
                    cpe_struct_V=float(row[5]),
                    length=float(row[6]),
                    width=float(row[7]),
                    roof_columns=int(row[8]),
                    roof_rows=int(row[9]))
        db.session.add(tmp)
        db.session.commit()
        return tmp


def loadConnectionTypeGroupsFromCSV(path, house, db):
    fileName = os.path.join(path, 'conn_group.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = connection_type.ConnectionTypeGroup(
            group_name=row[0],
            distribution_order=int(row[1]),
            distribution_direction=row[2],
            trigger_collapse_at=float(row[4]),
            patch_distribution=int(row[5]),
            set_zone_to_zero=int(row[6]),
            water_ingress_order=int(row[7]))
        tmp.costing = house.getCostingByName(row[3])
        house.conn_type_groups.append(tmp)
    db.session.commit()


def loadConnectionTypesFromCSV(path, house, db):
    fileName = os.path.join(path, 'conn_type.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = connection_type.ConnectionType(row[0],
                                             float(row[6]),
                                             float(row[1]),
                                             float(row[2]),
                                             float(row[3]),
                                             float(row[4]))
        tmp.group = house.getConnTypeGroupByName(row[5])
        house.conn_types.append(tmp)
    db.session.commit()


def loadConnectionsFromCSV(path, house, db):
    fileName = os.path.join(path, 'connections.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = connection.Connection(connection_name=row[0], edge=int(row[3]))
        tmp.ctype = house.getConnTypeByName(row[1])
        tmp.zone_id = house.getZoneByName(row[2]).id
        house.connections.append(tmp)
    db.session.commit()


def loadDamageCostingsFromCSV(path, house, db):
    fileName = os.path.join(path, 'damage_costing_data.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = damage_costing.DamageCosting(
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
    fileName = os.path.join(path, 'damage_factorings.csv')
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
                'Invalid connection group name given: %s' % row[0])
        if factor_id == 0:
            raise LookupError(
                'Invalid connection group name given: %s' % row[1])
        tmp = damage_costing.DamageFactoring(parent_id=parent_id,
                                             factor_id=factor_id)
        house.factorings.append(tmp)
    db.session.commit()


def loadWaterCostingsFromCSV(path, house, db):
    fileName = os.path.join(path, 'water_ingress_costing_data.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = wateringress.WaterIngressCosting(name=row[0],
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
    fileName = os.path.join(path, 'zones.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = zone.Zone(zone_name=row[0],
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


def loadConnectionInfluencesFromCSV(path, house, db):
    fileName = os.path.join(path, 'connectionzoneinfluences.csv')
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
                    parentLeft = house.getConnectionByName(data)
                elif (col % 2) != 0:
                    childRight = house.getZoneByName(data)
                elif childRight is not None and parentLeft is not None:
                    inf = influence.Influence(coeff=float(data))
                    inf.zone = childRight
                    # print 'adding influence from %s to %s of %f' % (
                    # parentLeft, childRight, float(data))
                    parentLeft.zones.append(inf)
                    childRight = None
        lineCount += 1
    db.session.commit()


def loadWallsFromCSV(path, house, db):
    fileName = os.path.join(path, 'walls.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = Wall(direction=int(row[0]), area=float(row[1]))
        house.walls.append(tmp)
    db.session.commit()


def loadCoverageTypesFromCSV(path, house, db):
    fileName = os.path.join(path, 'cov_types.csv')
    x = pd.read_csv(fileName)
    for _, row in x.iterrows():
        tmp = CoverageType(name=row[0],
                           failure_momentum_mean=float(row[1]),
                           failure_momentum_stddev=float(row[2]))
        house.cov_types.append(tmp)
    db.session.commit()


def loadCoveragesFromCSV(path, house, db):
    fileName = os.path.join(path, 'coverages.csv')
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


def importDataFromPath(path, db):

    print 'Importing House Data from path: {}'.format(path)
    house = loadFromCSV(path, db)
    print house
    loadWallsFromCSV(path, house, db)
    print 'walls'
    loadCoverageTypesFromCSV(path, house, db)
    print 'coverage types'
    loadCoveragesFromCSV(path, house, db)
    print 'coverages'
    loadDamageCostingsFromCSV(path, house, db)
    print 'damage costings'
    loadWaterCostingsFromCSV(path, house, db)
    print 'water costings'
    loadConnectionTypeGroupsFromCSV(path, house, db)
    print 'connection type groups'
    loadConnectionTypesFromCSV(path, house, db)
    print 'connection types'
    loadDamageFactoringsFromCSV(path, house, db)
    print 'damage factorings'
    loadZoneFromCSV(path, house, db)
    print 'zones'
    loadConnectionsFromCSV(path, house, db)
    print 'connections'
    loadConnectionInfluencesFromCSV(path, house, db)
    print 'connectionzoneinfluences'
    loadStructurePatchesFromCSV(path, house, db)
    print 'structureinfluencepatches'


# unit tests
if __name__ == '__main__':
    import unittest

    db_ = database.DatabaseManager('../model.db', verbose=False)

    class MyTestCase(unittest.TestCase):
        def test_constr(self):
            h = House(house_name='Carl House',
                      replace_cost=3434.0,
                      height=5.0,
                      cpe_V=0.1212,
                      cpe_k=0.1212,
                      cpe_struct_V=0.1212,
                      length=0.1212,
                      width=0.1212,
                      roof_columns=20,
                      roof_rows=20)
            print h

        def test_house(self):
            house_name = 'Masonry Block House'
            # house_name = 'Group 4 House'
            h = queryHouseWithName(house_name, db_)
            print 'Found my house: ', h
            print 'Costings: '
            for costing in h.costings:
                print costing
            print 'Walls: '
            for wall in h.walls:
                print wall
                for cov in wall.coverages:
                    print '\tcoverage: ', cov
            print 'CoverageTypes: '
            for covt in h.cov_types:
                print covt


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
