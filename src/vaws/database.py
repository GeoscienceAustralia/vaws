"""
    database.py - manage SQLite database
"""
from sqlalchemy import create_engine, Integer, String, Float, Column, \
    ForeignKey, event, exc
from sqlalchemy.sql import select
from sqlalchemy.orm import sessionmaker, relation, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.interfaces import PoolListener
from stats import compute_logarithmic_mean_stddev

# import sys
# import os

Base = declarative_base()


class ForeignKeysListener(PoolListener):
    def connect(self, dbapi_con, con_record):
        db_cursor = dbapi_con.execute('pragma foreign_keys=ON')
        # db_cursor = dbapi_con.execute('pragma foreign_keys')
        for row in db_cursor:
            print 'row: ', row


# def configure(db_file=None, verbose=False, flag_make=False):
#
#     if not db_file:
#         path_, _ = os.path.split(os.path.abspath(__file__))
#         db_file = os.path.abspath(os.path.join(path_, '../model.db'))
#
#     if not (flag_make or os.path.exists(db_file)):
#         msg = 'Error: database file {} not found'.format(db_file)
#         sys.exit(msg)
#
#     print 'model db is loaded from or created to : {}'.format(db_file)
#
#     return DatabaseManager(db_file, verbose)


# def _add_process_guards(engine):
#     """Add multiprocessing guards.
#
#     Forces a connection to be reconnected if it is detected
#     as having been shared to a sub-process.
#
#     """
#
#     @event.listens_for(engine, "connect")
#     def connect(dbapi_connection, connection_record):
#         connection_record.info['pid'] = os.getpid()
#
#     @event.listens_for(engine, "checkout")
#     def checkout(dbapi_connection, connection_record, connection_proxy):
#         pid = os.getpid()
#         if connection_record.info['pid'] != pid:
#             # LOG.debug(_LW(
#             #     "Parent process %(orig)s forked (%(newproc)s) with an open "
#             #     "database connection, "
#             #     "which is being discarded and recreated."),
#             #     {"newproc": pid, "orig": connection_record.info['pid']})
#             connection_record.connection = connection_proxy.connection = None
#             raise exc.DisconnectionError(
#                 "Connection record belongs to pid %s, "
#                 "attempting to check out in pid %s" %
#                 (connection_record.info['pid'], pid)
#             )


class DatabaseManager(object):
    def __init__(self, file_name, verbose=False):
        self.file_name = file_name
        try:
            self.database_url = 'sqlite:///' + file_name
        except TypeError:
            self.database_url = 'sqlite://'
        self.engine = create_engine(self.database_url,
                                    echo=verbose,
                                    echo_pool=False,
                                    listeners=[ForeignKeysListener()])
        # _add_process_guards(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(self.engine)

    def close(self):
        self.session.expire_all()
        self.session.close()

    # def qryDebrisTypes(self):
    #     s = select(
    #         [DebrisType.name, DebrisType.cdav])
    #     result = self.session.execute(s)
    #     list_ = result.fetchall()
    #     result.close()
    #     return list_

    def get_costing_by_name(self, costing_name):

        try:
            return self.session.query(DamageCosting).filter_by(
                costing_name=costing_name).one()
        except:
            raise LookupError('Invalid costing_name: {}'.format(costing_name))

    def get_conn_by_name(self, conn_name):

        try:
            return self.session.query(Connection).filter_by(
                connection_name=conn_name).one()
        except:
            raise LookupError('Invalid conn_name: {}'.format(conn_name))

    def get_conn_type_by_name(self, conn_type_name):

        try:
            return self.session.query(ConnectionType).filter_by(
                connection_type=conn_type_name).one()
        except:
            raise LookupError('Invalid conn_type_name: {}'.format(conn_type_name))

    def get_conn_type_group_by_name(self, conn_type_group_name):

        try:
            return self.session.query(ConnectionTypeGroup).filter_by(
                group_name=conn_type_group_name).one()
        except:
            raise LookupError('Invalid conn_type_group_name: {}'.format(conn_type_group_name))

    def get_zone_by_name(self, zone_name):
        try:
            return self.session.query(Zone).filter_by(
                zone_name=zone_name).one()
        except:
            raise LookupError('Invalid zone_name: {}'.format(zone_name))



    '''
    def qryConnectionPatchesFromDamagedConn(self, damagedConnID):
        s = select([Patch.target_connection_id, Patch.zone_id,
                    Patch.coeff],
                   Patch.damaged_connection_id == damagedConnID)
        result = self.session.execute(s)
        list_ = result.fetchall()
        result.close()
        return list_

    def qryConnectionPatches(self):
        s = select(
            [Patch.damaged_connection_id, Patch.target_connection_id,
             Patch.zone_id, Patch.coeff]).order_by(
            Patch.damaged_connection_id, Patch.target_connection_id,
            Patch.zone_id)
        result = self.session.execute(s)
        list_ = result.fetchall()
        result.close()
        return list_

    def get_list_conn_type(self):

        s = select([Connection.connection_name,
                    ConnectionType.connection_type,
                    ConnectionTypeGroup.group_name]).where(
            Connection.connection_type_id == ConnectionType.c.id)

        list_conn, list_conn_type, list_conn_type_group = set(), set(), set()
        for item in self.session.execute(s):
            (conn, conn_type, conn_type_group) = item
            list_conn.add(conn)
            list_conn_type.add(conn_type)
            list_conn_type_group.add(conn_type_group)
        return list_conn, list_conn_type, list_conn_type_group

    def get_list_zone(self):

        s = select([Zone.zone_name])

        list_zone = set()
        for item in self.session.execute(s):
            (zone,) = item
            list_zone.add(zone)

        return list_zone

    def populate_water_costs(self):

        cached_water_costs = dict()

        damage_names = self.session.query(
            WaterIngressCosting.name).distinct().all()

        for item in damage_names:
            dmg_name = item[0]
            water_ingress = self.session.query(WaterIngressCosting).filter_by(
                name=dmg_name).order_by(WaterIngressCosting.wi).all()
            cached_water_costs[dmg_name] = water_ingress

        return cached_water_costs
    '''


# class Terrain(Base):
#     __tablename__ = 'terrain_envelopes'
#     tcat = Column(String, primary_key=True)
#     profile = Column(Integer, primary_key=True)
#     z = Column(Integer, primary_key=True)
#     m = Column(Float)


# class DebrisType(Base):
#     __tablename__ = 'debris_types'
#     name = Column(String, primary_key=True)
#     cdav = Column(Float)


class Footprint(Base):
    __tablename__ = 'footprint'
    id = Column(Integer, primary_key=True)
    x_coord = Column(Float)
    y_coord = Column(Float)
    house_id = Column(Integer, ForeignKey('houses.id'))


class Patch(Base):
    __tablename__ = 'patches'
    damaged_connection_id = Column(Integer,
                                   ForeignKey('connections.id'),
                                   primary_key=True)
    target_connection_id = Column(Integer,
                                  ForeignKey('connections.id'),
                                  primary_key=True)
    conn_id = Column(Integer,
                     ForeignKey('connections.id'),
                     primary_key=True)
    coeff = Column(Float)


class Connection(Base):
    __tablename__ = 'connections'
    id = Column(Integer, primary_key=True)

    connection_name = Column(String)
    edge = Column(Integer)

    zone_id = Column(Integer, ForeignKey('zones.id'))

    house_id = Column(Integer, ForeignKey('houses.id'))
    connection_type_id = Column(Integer, ForeignKey('connection_types.id'))

    ctype = relation('ConnectionType', uselist=False,
                     backref=backref('connections_of_type'))
    location_zone = relation('Zone', uselist=False,
                             backref=backref('located_conns'))
    influences = relation('Influence')

    def __str__(self):
        return '({} @ {})'.format(self.connection_name, self.location_zone)


class ConnectionTypeGroup(Base):
    __tablename__ = 'connection_type_groups'
    id = Column(Integer, primary_key=True)
    group_name = Column(String)
    distribution_order = Column(Integer)
    distribution_direction = Column(String)
    trigger_collapse_at = Column(Float)
    patch_distribution = Column(Integer)
    set_zone_to_zero = Column(Integer)
    water_ingress_order = Column(Integer)
    costing_id = Column(Integer, ForeignKey('damage_costings.id'))
    house_id = Column(Integer, ForeignKey('houses.id'))
    costing = relation('DamageCosting', uselist=False,
                       backref=backref('conn_types'))

    def __str__(self):
        return "({})".format(self.group_name)


class ConnectionType(Base):
    __tablename__ = 'connection_types'
    id = Column(Integer, primary_key=True)
    connection_type = Column(String)
    costing_area = Column(Float)

    # assigned with functions
    strength_mean = Column(Float)
    strength_std_dev = Column(Float)
    deadload_mean = Column(Float)
    deadload_std_dev = Column(Float)

    grouping_id = Column(Integer, ForeignKey('connection_type_groups.id'))
    house_id = Column(Integer, ForeignKey('houses.id'))
    group = relation('ConnectionTypeGroup', uselist=False,
                     backref=backref('conn_types'))

    def __init__(self, conn_type, costing_area, mu_strength, sd_strength,
                 mu_deadload, sd_deadload):
        """
        Args:
            conn_type: connection type
            costing_area: costing area
            mu_strength: arithmetic mean of strength
            sd_strength: arithmetic std of strength
            mu_deadload:
            sd_deadload:
        """
        self.connection_type = conn_type
        self.costing_area = costing_area

        self.strength_mean, self.strength_std_dev = \
            compute_logarithmic_mean_stddev(mu_strength, sd_strength)

        self.deadload_mean, self.deadload_std_dev = \
            compute_logarithmic_mean_stddev(mu_deadload, sd_deadload)

    def __str__(self):
        return "({}/{})".format(self.group.group_name, self.connection_type)


class DamageFactoring(Base):
    __tablename__ = 'damage_factorings'
    parent_id = Column(Integer, ForeignKey('connection_type_groups.id'),
                       primary_key=True)
    factor_id = Column(Integer, ForeignKey('connection_type_groups.id'),
                       primary_key=True)
    house_id = Column(Integer, ForeignKey('houses.id'))
    parent = relation('ConnectionTypeGroup',
                      primaryjoin='damage_factorings.c.parent_id'
                                  '==connection_type_groups.c.id')
    factor = relation('ConnectionTypeGroup',
                      primaryjoin='damage_factorings.c.factor_id'
                                  '==connection_type_groups.c.id')


class DamageCosting(Base):
    __tablename__ = 'damage_costings'
    id = Column(Integer, primary_key=True)
    costing_name = Column(String)
    area = Column(Float)
    envelope_factor_formula_type = Column(Integer)
    envelope_repair_rate = Column(Float)
    env_coeff_1 = Column(Float)
    env_coeff_2 = Column(Float)
    env_coeff_3 = Column(Float)
    internal_factor_formula_type = Column(Integer)
    internal_repair_rate = Column(Float)
    int_coeff_1 = Column(Float)
    int_coeff_2 = Column(Float)
    int_coeff_3 = Column(Float)
    house_id = Column(Integer, ForeignKey('houses.id'))

    def __repr__(self):
        return "<DamageCosting('%s', '%f m', '%d', '$ %f', '%f', '%f', '%f', " \
               "'%d', '$ %f', '%f', '%f', '%f')>" % (
        self.costing_name,
        self.area,
        self.envelope_factor_formula_type,
        self.envelope_repair_rate,
        self.env_coeff_1,
        self.env_coeff_2,
        self.env_coeff_3,
        self.internal_factor_formula_type,
        self.internal_repair_rate,
        self.int_coeff_1,
        self.int_coeff_2,
        self.int_coeff_3)


# class DebrisRegion(Base):
#     __tablename__ = 'debris_regions'
#     name = Column(String, primary_key=True)
#     cr = Column(Float)
#     cmm = Column(Float)
#     cmc = Column(Float)
#     cfm = Column(Float)
#     cfc = Column(Float)
#     rr = Column(Float)
#     rmm = Column(Float)
#     rmc = Column(Float)
#     rfm = Column(Float)
#     rfc = Column(Float)
#     pr = Column(Float)
#     pmm = Column(Float)
#     pmc = Column(Float)
#     pfm = Column(Float)
#     pfc = Column(Float)
#     alpha = Column(Float)
#     beta = Column(Float)


class Influence(Base):
    __tablename__ = 'influences'
    connection_id = Column(Integer, ForeignKey('connections.id'),
                           primary_key=True)
    id = Column(Integer, primary_key=True)
    coeff = Column(Float)

    def __repr__(self):
        return "('{:d}', '{:d}', '{:f}')".format(self.connection_id,
                                                 self.id,
                                                 self.coeff)


class WaterIngressCosting(Base):
    __tablename__ = 'water_costs'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    wi = Column(Float)
    base_cost = Column(Float)
    formula_type = Column(Integer)
    coeff1 = Column(Float)
    coeff2 = Column(Float)
    coeff3 = Column(Float)
    house_id = Column(Integer, ForeignKey('houses.id'))

    def __repr__(self):
        return 'WaterIngressCosting(%s; %.3f; $%.2f; %d; %.3f; %.3f; %.3f)' % (
        self.name,
        self.wi,
        self.base_cost,
        self.formula_type,
        self.coeff1,
        self.coeff2,
        self.coeff3)


class Wall(Base):
    __tablename__ = 'walls'
    id = Column(Integer, primary_key=True)

    direction = Column(Integer)
    area = Column(Float)
    house_id = Column(Integer, ForeignKey('houses.id'))

    coverages = relation('Coverage')

    def __repr__(self):
        return 'Wall(dir: {:d}, area: {:.3f})'.format(self.direction, self.area)


class CoverageType(Base):
    __tablename__ = 'coverage_types'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    failure_momentum_mean = Column(Float)
    failure_momentum_stddev = Column(Float)
    house_id = Column(Integer, ForeignKey('houses.id'))

    def __repr__(self):
        return 'CoverageType(name: {})'.format(self.name)


class Coverage(Base):
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


class Zone(Base):
    __tablename__ = 'zones'
    id = Column(Integer, primary_key=True)
    zone_name = Column(String)
    zone_area = Column(Float)

    # Cpe for roof sheeting (cladding)
    coeff_N = Column(Float)
    coeff_NE = Column(Float)
    coeff_E = Column(Float)
    coeff_SE = Column(Float)
    coeff_S = Column(Float)
    coeff_SW = Column(Float)
    coeff_W = Column(Float)
    coeff_NW = Column(Float)

    # Cpe for rafter
    struct_coeff_N = Column(Float)
    struct_coeff_NE = Column(Float)
    struct_coeff_E = Column(Float)
    struct_coeff_SE = Column(Float)
    struct_coeff_S = Column(Float)
    struct_coeff_SW = Column(Float)
    struct_coeff_W = Column(Float)
    struct_coeff_NW = Column(Float)

    # Cpe for eave
    eaves_coeff_N = Column(Float)
    eaves_coeff_NE = Column(Float)
    eaves_coeff_E = Column(Float)
    eaves_coeff_SE = Column(Float)
    eaves_coeff_S = Column(Float)
    eaves_coeff_SW = Column(Float)
    eaves_coeff_W = Column(Float)
    eaves_coeff_NW = Column(Float)

    # leading roof
    leading_roof_N = Column(Integer)
    leading_roof_NE = Column(Integer)
    leading_roof_E = Column(Integer)
    leading_roof_SE = Column(Integer)
    leading_roof_S = Column(Integer)
    leading_roof_SW = Column(Integer)
    leading_roof_W = Column(Integer)
    leading_roof_NW = Column(Integer)

    cpi_alpha = Column(Float)
    wall_dir = Column(Integer)
    house_id = Column(Integer, ForeignKey('houses.id'))


class House(Base):
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
    costings = relation(DamageCosting)
    factorings = relation(DamageFactoring)
    conn_types = relation(ConnectionType)
    conn_type_groups = relation(ConnectionTypeGroup,
                                order_by=ConnectionTypeGroup.distribution_order)
    water_groups = relation(ConnectionTypeGroup,
                            order_by=ConnectionTypeGroup.water_ingress_order)
    zones = relation(Zone)
    connections = relation(Connection)
    footprint = relation(Footprint)

    def __repr__(self):
        return '({}, ${:.2f})'.format(self.house_name, self.replace_cost)
