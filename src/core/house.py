"""
    House Module - reference storage for House type information.
        - loaded from database
        - imported from '../data/houses/subfolder' (so we need python constr)
"""
import os
from sqlalchemy import Integer, String, Float, Column, ForeignKey, orm
from sqlalchemy.orm import relation
import pandas as pd

import database
import connection_type
import connection
import zone
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
        return 'Wall(dir: {:d}, area: {:.3f})'.format(self.direction, self.area)

    @orm.reconstructor
    def resetCoverages(self):
        self.result_perc_collapsed = 0
        for cov in self.coverages:
            cov.result_intact = True
            cov.result_num_impacts = 0


class CoverageType(database.Base):
    __tablename__ = 'coverage_types'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    failure_momentum_mean = Column(Float)
    failure_momentum_stddev = Column(Float)
    house_id = Column(Integer, ForeignKey('houses.id'))

    def __repr__(self):
        return 'CoverageType(name: {})'.format(self.name)


class Coverage(database.Base):
    __tablename__ = 'coverages'
    id = Column(Integer, primary_key=True)
    description = Column(String)
    area = Column(Float)
    wall_id = Column(Integer, ForeignKey('walls.id'))
    type_id = Column(Integer, ForeignKey('coverage_types.id'), nullable=False)
    type = relation(CoverageType, uselist=False)

    def __repr__(self):
        return 'Coverage(desc: {}, area: {:.3f})'.format(self.description,
                                                         self.area)


class House(database.Base):

    __tablename__ = 'houses'
    id = Column(Integer, primary_key=True)

    house_name = Column(String)
    replace_cost = Column(Float)
    height = Column(Float)
    cpe_V = Column(Float)
    cpe_k = Column(Float)
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

    # def __init__(self, house_name, replacement_cost, height, cpe_cov,
    #              cpe_k, cpe_struct_cov, length, width, roof_columns, roof_rows):
    #
    #     print '{}'.format(__file__)
    #     self.house_name = house_name
    #     self.replace_cost = replacement_cost
    #     self.height = height
    #     # Cpe (external pressure coefficient) for roof assumed to
    #     # follow Type III GEV with V (cov) and k (shape factor)
    #     self.cpe_V = cpe_cov
    #     self.cpe_k = cpe_k
    #     # Cpe for structure. Only cov is provided assuming k is the same as roof
    #     self.cpe_struct_V = cpe_struct_cov
    #     self.length = length
    #     self.width = width
    #     self.roof_columns = roof_columns
    #     self.roof_rows = roof_rows
    #
    #     # computed value
    #     self.cpe_A, self.cpe_B = zone.calc_big_a_b_values(self.cpe_k)

    def __repr__(self):
        return '({}, ${:.2f})'.format(self.house_name, self.replace_cost)

    def getConnectionByName(self, connName):
        for conn in self.connections:
            if conn.connection_name == connName:
                return conn
        raise LookupError('Invalid connName: {}'.format(connName))

    # def resetCoverages(self):
    #     for wall in self.walls:
    #         wall.result_perc_collapsed = 0
    #         for cov in wall.coverages:
    #             cov.result_intact = True
    #             cov.result_num_impacts = 0

    def resetZoneInfluences(self):
        for c in self.connections:
            inflZonesByConn[c] = dict()
            for infl in c.zones:
                inflZonesByConn[c][infl.zone] = infl.coeff

    def reset_connection_failure(self):
        for c in self.connections:
            c.result_failure_v = 0.0
            c.result_failure_v_i = 0

    @orm.reconstructor
    def reset_results(self):
        self.resetZoneInfluences()
        for w in self.walls:
            w.resetCoverages()
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
        raise LookupError('Invalid zoneName: {}'.format(zoneName))

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
        raise LookupError('Invalid ctName: {}'.format(ctName))

    def getConnTypeGroupByName(self, ctgName):
        for group in self.conn_type_groups:
            if group.group_name == ctgName:
                return group
        raise LookupError('Invalid ctgName: {}'.format(ctgName))

    def getCostingByName(self, costingName):
        for dmg in self.costings:
            if dmg.costing_name == costingName:
                return dmg
        raise LookupError('Invalid costingName: {}'.format(costingName))

    def getWallArea(self):
        area = 0
        for wall in self.walls:
            area += wall.area
        return area

    def getWallByDirection(self, wind_dir):
        for wall in self.walls:
            if wall.direction == wind_dir:
                return wall
        raise LookupError('Invalid wind_dir: {:d}'.format(wind_dir))

def queryHouses(db):
    return db.session.query(House.house_name,
                            House.replace_cost,
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
    # house.reset_results()

    connByTypeMap.clear()  # clears even the copy of the dic.
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
    # house.resetCoverages()
    wateringress.populate_water_costs(db, house.id)
    return house


# unit tests
if __name__ == '__main__':
    import unittest

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.db1 = database.DatabaseManager(os.path.join('../model.db'),
                                               verbose=False)

            cls.db2 = database.DatabaseManager(os.path.join('../test.db'),
                                               verbose=False)

        @classmethod
        def tearDown(cls):
            cls.db1.close()
            cls.db2.close()

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

        def test_house1(self):

            # db_ = database.DatabaseManager('../model.db', verbose=False)

            house_name = 'Masonry Block House'
            h = queryHouseWithName(house_name, self.db1)
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

        def test_house2(self):

            house_name = 'Group 4 House'
            h = queryHouseWithName(house_name, self.db1)
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

        def test_house3(self):

            house_name = 'Test1'
            h = queryHouseWithName(house_name, self.db2)
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
