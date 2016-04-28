from sqlalchemy import Integer, Float, Column, ForeignKey
from sqlalchemy.orm import relation
import database


class Influence(database.Base):
    __tablename__ = 'influences'
    connection_id = Column(Integer, ForeignKey('connections.id'),
                           primary_key=True)
    zone_id = Column(Integer, ForeignKey('zones.id'), primary_key=True)
    coeff = Column(Float)
    zone = relation("Zone")

    def __repr__(self):
        return "('{:d}', '{:d}', '{:f}', '{}')".format(self.connection_id,
                                                       self.zone_id,
                                                       self.coeff,
                                                       'None' if not self.zone
                                                       else self.zone.zone_name)
