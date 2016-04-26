"""
    database.py - manage SQLite database
"""
from sqlalchemy import create_engine, Table, Integer, String, Float, Column, \
    MetaData, ForeignKey
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


def configure(model_db=None, verbose=False, flag_make=False):

    if not model_db:
        path_, _ = os.path.split(os.path.abspath(__file__))
        model_db = os.path.abspath(os.path.join(path_, '../model.db'))

    if not (flag_make or os.path.exists(model_db)):
        msg = 'Error: database file {} not found'.format(model_db)
        sys.exit(msg)

    print 'model db is loaded from or created to : {}'.format(model_db)

    global db
    db = DatabaseManager(model_db, verbose)


class DatabaseManager(object):
    def __init__(self, file_name, verbose):
        self.file_name = file_name
        self.database_url = 'sqlite:///' + file_name
        self.engine = create_engine(self.database_url,
                                    echo=verbose,
                                    echo_pool=False,
                                    listeners=[ForeignKeysListener()])
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

        self.structure_patch_table = Table('patches', Base.metadata,
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
