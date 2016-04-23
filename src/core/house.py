"""
    House Module - reference storage for House type information.
        - loaded from database
        - imported from '../data/houses/subfolder' (so we need python constr)
"""
from sqlalchemy import Integer, String, Float, Column, ForeignKey
from sqlalchemy.orm import relation
import database
import connection_type
import connection
import zone
import influence
import damage_costing
import csvarray
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

    def clear_sim_results(self):
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

    def qryZoneByWallDirection(self, wall_dir):
        return database.db.session.query(zone.Zone).filter(
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


def loadStructurePatchesFromCSV(fileName, house):
    ''' 
    Load structural influence patches CSV - 
    format is: damaged_conn, target_conn, zone, infl, zone, infl,....
    '''
    db = database.db
    lineCount = 0
    for line in open(fileName, 'r'):
        if lineCount != 0:
            damagedConn = None
            targetConn = None
            zone = None
            line = line.rstrip()
            fields = line.strip().split(",")
            for col, data in enumerate(fields):
                if len(
                        data) == 0:  # excel sometimes puts extra ,,,, in when staggered cols
                    break
                if col == 0:
                    damagedConn = house.getConnectionByName(data)
                elif col == 1:
                    targetConn = house.getConnectionByName(data)
                elif (col % 2) == 0:
                    zone = house.getZoneByName(data)
                elif damagedConn is not None and targetConn is not None and zone is not None:
                    ins = db.structure_patch_table.insert().values(
                        damaged_connection_id=damagedConn.id,
                        target_connection_id=targetConn.id,
                        zone_id=zone.id,
                        coeff=float(data))
                    db.session.execute(ins)
                    zone = None
        lineCount += 1
    db.session.commit()


def queryHouses():
    return database.db.session.query(House.house_name, House.replace_cost,
                                     House.height).all()


def queryHouseWithName(hn):
    """
    find (and select as current) the given house name
    Args:
        hn: house name

    Returns:
    """

    house = database.db.session.query(House).filter_by(house_name=hn).one()
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
    wateringress.populate_water_costs(house.id)
    return house


def loadFromCSV(fileName):
    x = csvarray.readArrayFromCSV(fileName, "S50,f4,f4,f4,f4,f4,f4,f4,i4,i4")
    for row in x:
        tmp = House(house_name=row[0],
                    replace_cost=row[1],
                    height=row[2],
                    cpe_V=row[3],
                    cpe_k=row[4],
                    cpe_struct_V=row[5],
                    length=row[6],
                    width=row[7],
                    roof_columns=int(row[8]),
                    roof_rows=int(row[9]))
        database.db.session.add(tmp)
        database.db.session.commit()
        return tmp


def loadConnectionTypeGroupsFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName, "S50,i4,S50,S50,f4,i4,i4,i4")
    for row in x:
        tmp = connection_type.ConnectionTypeGroup(group_name=row[0],
                                                  distribution_order=int(
                                                      row[1]),
                                                  distribution_direction=row[2],
                                                  trigger_collapse_at=row[4],
                                                  patch_distribution=int(
                                                      row[5]),
                                                  set_zone_to_zero=int(row[6]),
                                                  water_ingress_order=int(
                                                      row[7]))
        tmp.costing = house.getCostingByName(row[3])
        house.conn_type_groups.append(tmp)
    database.db.session.commit()


def loadConnectionTypesFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName, "S50,f4,f4,f4,f4,S50,f4")
    for row in x:
        tmp = connection_type.ConnectionType(row[0], row[6], row[1], row[2],
                                             row[3], row[4])
        tmp.group = house.getConnTypeGroupByName(row[5])
        house.conn_types.append(tmp)
    database.db.session.commit()


def loadConnectionsFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName, "S50,S50,S50,i4")
    for row in x:
        tmp = connection.Connection(connection_name=row[0], edge=int(row[3]))
        tmp.ctype = house.getConnTypeByName(row[1])
        tmp.zone_id = house.getZoneByName(row[2]).id
        house.connections.append(tmp)
    database.db.session.commit()


def loadDamageCostingsFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName,
                                  "S50,f4,f4,i4,f4,f4,f4,f4,i4,f4,f4,f4")
    for row in x:
        tmp = damage_costing.DamageCosting(costing_name=row[0],
                                           area=row[1],
                                           envelope_factor_formula_type=int(
                                               row[3]),
                                           envelope_repair_rate=row[2],
                                           env_coeff_1=row[4],
                                           env_coeff_2=row[5],
                                           env_coeff_3=row[6],
                                           internal_factor_formula_type=int(
                                               row[8]),
                                           internal_repair_rate=row[7],
                                           int_coeff_1=row[9],
                                           int_coeff_2=row[10],
                                           int_coeff_3=row[11])
        house.costings.append(tmp)
    database.db.session.commit()


def loadDamageFactoringsFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName, "S50,S50")
    for row in x:
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
    database.db.session.commit()


def loadWaterCostingsFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName, "S50,f4,f4,i4,f4,f4,f4")
    for row in x:
        tmp = wateringress.WaterIngressCosting(name=row[0],
                                               wi=row[1],
                                               base_cost=row[2],
                                               formula_type=int(row[3]),
                                               coeff1=row[4],
                                               coeff2=row[5],
                                               coeff3=row[6])
        tmp.house_id = house.id
        database.db.session.add(tmp)
    database.db.session.commit()



def loadZoneFromCSV(fileName, house):
    format = "S50,f4,\
f4,f4,f4,f4,f4,f4,f4,f4,\
f4,f4,f4,f4,f4,f4,f4,f4,\
f4,f4,f4,f4,f4,f4,f4,f4,\
i4,i4,i4,i4,i4,i4,i4,i4,\
f4,i4"
    x = csvarray.readArrayFromCSV(fileName, format)
    for row in x:
        tmp = zone.Zone(zone_name=row[0],
                        zone_area=row[1],
                        coeff_N=row[6],
                        coeff_NE=row[7],
                        coeff_E=row[8],
                        coeff_SE=row[9],
                        coeff_S=row[2],
                        coeff_SW=row[3],
                        coeff_W=row[4],
                        coeff_NW=row[5],
                        struct_coeff_N=row[14],
                        struct_coeff_NE=row[15],
                        struct_coeff_E=row[16],
                        struct_coeff_SE=row[17],
                        struct_coeff_S=row[10],
                        struct_coeff_SW=row[11],
                        struct_coeff_W=row[12],
                        struct_coeff_NW=row[13],
                        eaves_coeff_N=row[22],
                        eaves_coeff_NE=row[23],
                        eaves_coeff_E=row[24],
                        eaves_coeff_SE=row[25],
                        eaves_coeff_S=row[18],
                        eaves_coeff_SW=row[19],
                        eaves_coeff_W=row[20],
                        eaves_coeff_NW=row[21],
                        leading_roof_N=int(row[30]),
                        leading_roof_NE=int(row[31]),
                        leading_roof_E=int(row[32]),
                        leading_roof_SE=int(row[33]),
                        leading_roof_S=int(row[26]),
                        leading_roof_SW=int(row[27]),
                        leading_roof_W=int(row[28]),
                        leading_roof_NW=int(row[29]),
                        cpi_alpha=row[34],
                        wall_dir=int(row[35]))
        house.zones.append(tmp)
    database.db.session.commit()



def loadConnectionInfluencesFromCSV(fileName, house):
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
    database.db.session.commit()



def loadWallsFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName, "i4,f4")
    for row in x:
        tmp = Wall(direction=int(row[0]), area=row[1])
        house.walls.append(tmp)
    database.db.session.commit()



def loadCoverageTypesFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName, "S50,f4,f4")
    for row in x:
        tmp = CoverageType(name=row[0], failure_momentum_mean=row[1],
                           failure_momentum_stddev=row[2])
        house.cov_types.append(tmp)
    database.db.session.commit()



def loadCoveragesFromCSV(fileName, house):
    x = csvarray.readArrayFromCSV(fileName, "S50,i4,f4,S50")
    for row in x:
        tmp = Coverage(description=row[0], area=row[2])
        for covt in house.cov_types:
            if covt.name == row[3]:
                tmp.type = covt
                break
        for wall in house.walls:
            if wall.direction == int(row[1]):
                tmp.wall_id = wall.id
                wall.coverages.append(tmp)
                break
    database.db.session.commit()



def importDataFromPath(path):
    print 'Importing House Data from path: %s' % (path)
    house = loadFromCSV(path + '/house_data.csv')
    print house
    loadWallsFromCSV(path + '/walls.csv', house);
    print 'walls'
    loadCoverageTypesFromCSV(path + '/cov_types.csv', house);
    print 'coverage types'
    loadCoveragesFromCSV(path + '/coverages.csv', house);
    print 'coverages'
    loadDamageCostingsFromCSV(path + '/damage_costing_data.csv', house);
    print 'damage costings'
    loadWaterCostingsFromCSV(path + '/water_ingress_costing_data.csv', house);
    print 'water costings'
    loadConnectionTypeGroupsFromCSV(path + '/conn_group.csv', house);
    print 'connection type groups'
    loadConnectionTypesFromCSV(path + '/conn_type.csv', house);
    print 'connection types'
    loadDamageFactoringsFromCSV(path + '/damage_factorings.csv', house);
    print 'damage factorings'
    loadZoneFromCSV(path + '/zones.csv', house);
    print 'zones'
    loadConnectionsFromCSV(path + '/connections.csv', house);
    print 'connections'
    loadConnectionInfluencesFromCSV(path + '/connectionzoneinfluences.csv',
                                    house);
    print 'connectionzoneinfluences'
    loadStructurePatchesFromCSV(path + '/influencefactorpatches.csv', house);
    print 'structureinfluencepatches'


# unit tests
if __name__ == '__main__':
    import unittest


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
            h = queryHouseWithName(house_name)
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
