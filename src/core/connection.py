'''
    Connection Module - reference storage Connections
        - loaded from database
        - imported from '../data/houses/subfolder'
'''
from sqlalchemy import create_engine, Table, Integer, String, Float, Column, MetaData, ForeignKey
from sqlalchemy.orm import relation, backref
import database
import influence

# -------------------------------------------------------------
use_struct_pz_for = ['rafter', 'piersgroup', 'wallracking']

## -------------------------------------------------------------
class Connection(database.Base):
    __tablename__       = 'connections'
    id                  = Column(Integer, primary_key=True)
    connection_name     = Column(String)
    edge                = Column(Integer)
    zone_id             = Column(Integer, ForeignKey('zones.id'))
    house_id            = Column(Integer, ForeignKey('houses.id'))
    connection_type_id  = Column(Integer, ForeignKey('connection_types.id'))
    ctype               = relation("ConnectionType", uselist=False, backref=backref('connections_of_type'))
    location_zone       = relation("Zone", uselist=False, backref=backref('located_conns'))
    zones               = relation(influence.Influence)
    
    def __str__(self):
        return '(%s @ %s)' % (self.connection_name, self.location_zone)
    
    ## -------------------------------------------------------------
    def reset_results(self):
        self.result_strength = 0.0
        self.result_deadload = 0.0
        self.result_failure_v_raw = 0.0
        self.result_damaged = False
        self.result_damaged_report = {}
        self.result_damage_distributed = False
        
    ## -------------------------------------------------------------
    def damage(self, V, result_load, infl_by_zone_map, use_struct_pz=None):
        if self.result_damaged:
            return        
        if use_struct_pz is None:
            use_struct_pz = self.ctype.group.group_name in use_struct_pz_for        
        self.ctype.incr_damaged()     
        self.result_damaged = True                                    
        self.result_damaged_report['load'] = result_load
        infls_arr = []                            
        for z in sorted(infl_by_zone_map):
            infls_dict = {}
            infl = infl_by_zone_map[z] 
            infls_dict['infl'] = infl
            area = float(z.result_effective_area)
            infls_dict['area'] = area
            pz = z.result_pz_struct if use_struct_pz else z.result_pz
            infls_dict['pz'] = pz
            infls_dict['load'] = infl * area * pz
            infls_dict['name'] = z.zone_name
            infls_arr.append(infls_dict)                        
        self.result_damaged_report['infls'] = infls_arr                                        
        self.result_damage_distributed = False
        self.result_failure_v_i += 1
        self.result_failure_v_raw = V
        self.result_failure_v = (V + float(self.result_failure_v_i - 1)*self.result_failure_v) / float(self.result_failure_v_i)

# -------------------------------------------------------------
def assign_connection_strengths(house_conns, mean_factor, cov_factor):
    for conn in house_conns:
        conn.result_strength = conn.ctype.sample_strength(mean_factor, cov_factor)

# -------------------------------------------------------------
def assign_connection_deadloads(house_conns):
    for conn in house_conns:
        conn.result_deadload = conn.ctype.sample_deadload()
                
# -------------------------------------------------------------
def calc_connection_loads(V, ctg, house, file_damage, dmg_map, inflZonesByConn, connByTypeMap):
    if not ctg.enabled or len(ctg.conn_types) == 0:
        return
    
    # hard coded for optimization
    use_struct_pz = ctg.group_name in use_struct_pz_for
    
    # for all connection types belonging to the specified group
    for ct in ctg.conn_types:
        
        # for all connections belonging to the current connection type
        for c in connByTypeMap[ct]:
            if c.result_damaged:
                continue
    
            # sum loads from influencing zones
            result_load = 0.0
            dict = inflZonesByConn[c]
            for z in dict:
                if z.result_effective_area > 0.0:
                    if use_struct_pz:
                        result_load += (dict[z] * z.result_effective_area * z.result_pz_struct)      
                    else:
                        result_load += (dict[z] * z.result_effective_area * z.result_pz)
                
            # add dead load and if load is negative check for strength failure        
            result_load += c.result_deadload
            if result_load < 0.0:
                if -result_load >= c.result_strength:

                    # now a function so that collapse can record proper audit info.
                    c.damage(V, result_load, inflZonesByConn[c], use_struct_pz)
                                        
                    # if the damaged wind speed is less than current type min then store (for reporting of min damage speeds by type)
                    curr = dmg_map[ct.connection_type]
                    dmg_map[ct.connection_type] = V if V <= curr else curr 
                    
        # report percentage of connection type damaged at this V to file.
        file_damage.write(',')
        file_damage.write(str(ct.perc_damaged()*100.0))
        
        








