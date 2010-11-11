from sqlalchemy import create_engine, Table, Integer, String, Float, Column, MetaData, ForeignKey
from sqlalchemy.orm import relation, backref
import database

## -------------------------------------------------------------
class Influence(database.Base):
    __tablename__       = 'influences'
    connection_id       = Column(Integer, ForeignKey('connections.id'), primary_key=True)
    zone_id             = Column(Integer, ForeignKey('zones.id'), primary_key=True)
    coeff               = Column(Float)
    zone                = relation("Zone")
    
    def __repr__(self):
        return "('%d', '%d', '%f', '%s')" % (self.connection_id, self.zone_id, self.coeff, 'None' if not self.zone else self.zone.zone_name)


