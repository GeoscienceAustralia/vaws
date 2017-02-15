"""
    database.py - manage SQLite database
"""
from sqlalchemy import create_engine, Integer, String, Float, Column, \
    ForeignKey, event, exc
from sqlalchemy.sql import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.interfaces import PoolListener

import sys
import os

Base = declarative_base()


class ForeignKeysListener(PoolListener):
    def connect(self, dbapi_con, con_record):
        db_cursor = dbapi_con.execute('pragma foreign_keys=ON')
        # db_cursor = dbapi_con.execute('pragma foreign_keys')
        for row in db_cursor:
            pass
            # print 'row: ', row


def configure(db_file=None, verbose=False, flag_make=False):

    if not db_file:
        path_, _ = os.path.split(os.path.abspath(__file__))
        db_file = os.path.abspath(os.path.join(path_, '../model.db'))

    if not (flag_make or os.path.exists(db_file)):
        msg = 'Error: database file {} not found'.format(db_file)
        sys.exit(msg)

    print 'model db is loaded from or created to : {}'.format(db_file)

    return DatabaseManager(db_file, verbose)


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
        # self.metadata = MetaData()
        # self.metadata.bind = self.engine

        # # DESIGN: some tables don't need ORM mappings and can be dealt with
        # # pure SQL.
        # self.terrain_table = Table('terrain_envelopes', self.metadata,
        #                            Column('tcat', String, primary_key=True),
        #                            Column('profile', Integer, primary_key=True),
        #                            Column('z', Integer, primary_key=True),
        #                            Column('m', Float))
        #
        # self.debris_types_table = Table('debris_types', self.metadata,
        #                                 Column('name', String,
        #                                        primary_key=True),
        #                                 Column('cdav', Float))
        #
        # self.structure_patch_table = Table('patches', self.metadata,
        #                                    Column('damaged_connection_id',
        #                                           Integer,
        #                                           ForeignKey('connections.id'),
        #                                           primary_key=True),
        #                                    Column('target_connection_id',
        #                                           Integer,
        #                                           ForeignKey('connections.id'),
        #                                           primary_key=True),
        #                                    Column('zone_id', Integer,
        #                                           ForeignKey('zones.id'),
        #                                           primary_key=True),
        #                                    Column('coeff', Float))

    def create_tables(self):
        # self.metadata.create_all()
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        # self.metadata.drop_all()
        Base.metadata.drop_all(self.engine)

    def close(self):
        self.session.expire_all()
        self.session.close()

    def qryDebrisTypes(self):
        s = select(
            [DebrisType.name, DebrisType.cdav])
        result = self.session.execute(s)
        list_ = result.fetchall()
        result.close()
        return list_

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

    def get_list_conn_type(self, house_name):

        tb_house = Base.metadata.tables['houses']
        tb_conn = Base.metadata.tables['connections']
        tb_conn_type = Base.metadata.tables['connection_types']
        tb_conn_type_group = Base.metadata.tables['connection_type_groups']

        s0 = select([tb_house.c.id]).where(tb_house.c.house_name == house_name)
        house_id = str(self.session.query(s0).one()[0])

        s = select([tb_conn.c.connection_name,
                    tb_conn_type.c.connection_type,
                    tb_conn_type_group.c.group_name]).where(
            (tb_conn.c.connection_type_id == tb_conn_type.c.id) &
            (tb_conn.c.house_id == house_id))

        list_conn, list_conn_type, list_conn_type_group = set(), set(), set()
        for item in self.session.execute(s):
            (conn, conn_type, conn_type_group) = item
            list_conn.add(conn)
            list_conn_type.add(conn_type)
            list_conn_type_group.add(conn_type_group)
        return list_conn, list_conn_type, list_conn_type_group

    def get_list_zone(self, house_name):
        tb_house = Base.metadata.tables['houses']
        tb_zone = Base.metadata.tables['zones']

        s0 = select([tb_house.c.id]).where(tb_house.c.house_name == house_name)
        house_id = str(self.session.query(s0).one()[0])

        s = select([tb_zone.c.zone_name]).where(tb_zone.c.house_id == house_id)

        list_zone = set()
        for item in self.session.execute(s):
            (zone,) = item
            list_zone.add(zone)

        return list_zone


class Terrain(Base):
    __tablename__ = 'terrain_envelopes'
    tcat = Column(String, primary_key=True)
    profile = Column(Integer, primary_key=True)
    z = Column(Integer, primary_key=True)
    m = Column(Float)


class DebrisType(Base):
    __tablename__ = 'debris_types'
    name = Column(String, primary_key=True)
    cdav = Column(Float)


class Patch(Base):
    __tablename__ = 'patches'
    damaged_connection_id = Column(Integer,
                                   ForeignKey('connections.id'),
                                   primary_key=True)
    target_connection_id = Column(Integer,
                                  ForeignKey('connections.id'),
                                  primary_key=True)
    zone_id = Column(Integer,
                     ForeignKey('zones.id'),
                     primary_key=True)
    coeff = Column(Float)

