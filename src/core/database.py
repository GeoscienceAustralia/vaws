"""
    database.py - manage SQLite database
"""
from sqlalchemy import create_engine, Table, Integer, String, Float, Column, \
    MetaData, ForeignKey, event, exc
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
        db_cursor = dbapi_con.execute('pragma foreign_keys')
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


def _add_process_guards(engine):
    """Add multiprocessing guards.

    Forces a connection to be reconnected if it is detected
    as having been shared to a sub-process.

    """

    @event.listens_for(engine, "connect")
    def connect(dbapi_connection, connection_record):
        connection_record.info['pid'] = os.getpid()

    @event.listens_for(engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        pid = os.getpid()
        if connection_record.info['pid'] != pid:
            # LOG.debug(_LW(
            #     "Parent process %(orig)s forked (%(newproc)s) with an open "
            #     "database connection, "
            #     "which is being discarded and recreated."),
            #     {"newproc": pid, "orig": connection_record.info['pid']})
            connection_record.connection = connection_proxy.connection = None
            raise exc.DisconnectionError(
                "Connection record belongs to pid %s, "
                "attempting to check out in pid %s" %
                (connection_record.info['pid'], pid)
            )


class DatabaseManager(object):
    def __init__(self, file_name, verbose=False):
        self.file_name = file_name
        self.database_url = 'sqlite:///' + file_name
        self.engine = create_engine(self.database_url,
                                    echo=verbose,
                                    echo_pool=False,
                                    listeners=[ForeignKeysListener()])
        # _add_process_guards(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.metadata = MetaData()
        self.metadata.bind = self.engine

        # DESIGN: some tables don't need ORM mappings and can be dealt with
        # pure SQL.
        self.terrain_table = Table('terrain_envelopes', self.metadata,
                                   Column('tcat', String, primary_key=True),
                                   Column('profile', Integer, primary_key=True),
                                   Column('z', Integer, primary_key=True),
                                   Column('m', Float))

        self.debris_types_table = Table('debris_types', self.metadata,
                                        Column('name', String,
                                               primary_key=True),
                                        Column('cdav', Float))

        self.structure_patch_table = Table('patches', self.metadata,
                                           Column('damaged_connection_id',
                                                  Integer,
                                                  ForeignKey('connections.id'),
                                                  primary_key=True),
                                           Column('target_connection_id',
                                                  Integer,
                                                  ForeignKey('connections.id'),
                                                  primary_key=True),
                                           Column('zone_id', Integer,
                                                  ForeignKey('zones.id'),
                                                  primary_key=True),
                                           Column('coeff', Float))

    def create_tables(self):
        self.metadata.create_all()
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        self.metadata.drop_all()
        Base.metadata.drop_all(self.engine)

    def close(self):
        self.session.expire_all()
        self.session.close()

    def qryDebrisTypes(self):
        s = select(
            [self.debris_types_table.c.name, self.debris_types_table.c.cdav])
        result = self.engine.execute(s)
        list_ = result.fetchall()
        result.close()
        return list_

    def qryConnectionPatchesFromDamagedConn(self, damagedConnID):
        patches = self.structure_patch_table
        s = select([patches.c.target_connection_id, patches.c.zone_id,
                    patches.c.coeff],
                   patches.c.damaged_connection_id == damagedConnID)
        result = self.engine.execute(s)
        list_ = result.fetchall()
        result.close()
        return list_

    def qryConnectionPatches(self):
        patches = self.structure_patch_table
        s = select(
            [patches.c.damaged_connection_id, patches.c.target_connection_id,
             patches.c.zone_id, patches.c.coeff]).order_by(
            patches.c.damaged_connection_id, patches.c.target_connection_id,
            patches.c.zone_id)
        result = self.engine.execute(s)
        list_ = result.fetchall()
        result.close()
        return list_
