"""
    Water Ingress Module - damage and costing aspects of water ingress based
    off Martin's work.
"""
import scipy.stats
import numpy as np
from sqlalchemy import Integer, String, Float, Column, ForeignKey
import database


class WaterIngressCosting(database.Base):
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


cached_water_costs = {}


# note: house specific: needs to be rebuilt when house changes.
#
def populate_water_costs(db, hid):
    cached_water_costs.clear()
    damage_names = db.session.query(
        WaterIngressCosting.name).filter_by(house_id=hid).distinct().all()
    for damage_name in damage_names:
        dname = damage_name[0]
        wiarr = db.session.query(WaterIngressCosting).filter_by(
            house_id=hid).filter_by(name=dname).order_by(
            WaterIngressCosting.wi).all()
        cached_water_costs[dname] = wiarr


def get_watercost_for_damage_at_wi(damage_name, water_ingress):
    wiarr = cached_water_costs[damage_name]
    last_valid_row = wiarr[0]
    for wi in wiarr:
        if water_ingress >= wi.wi:
            last_valid_row = wi
    return last_valid_row


def get_curve_coeffs_for(di):
    """
    get curve coefficients given damage index
    Args:
        di: damage index
    di <= 0.1; 40.0 , 60.0
    di <= 0.2: 35.0, 55.0
    di <= 0.5: 0.0, 40.0
    di <= 1.0: -20.0, 20.0
    Returns: tuple of vl and vu

    """
    threshold = np.array([0.1, 0.2, 0.5, 2.0])
    vl = {0: 40.0, 1: 35.0, 2: 0.0, 3: -20.0}
    vu = {0: 60.0, 1: 55.0, 2: 40.0, 3: 20.0}
    idx = sum(di > threshold)
    return (vl[idx] + vu[idx]) / 2.0, (vu[idx] - vl[idx]) / 6.0


def get_wi_for_di(di, V):
    m, s = get_curve_coeffs_for(di)
    cdf_val = scipy.stats.norm.cdf(V, loc=m, scale=s)
    return cdf_val


def get_costing_for_envelope_damage_at_v(di, wind_speed, water_groups,
                                         out_file=None):
    water_ratio = get_wi_for_di(di, wind_speed)
    water_ingress_perc = water_ratio * 100.0

    damage_name = 'WI only'
    for ctg in water_groups:
        if ctg.water_ingress_order > 0 and ctg.result_percent_damaged > 0:
            damage_name = ctg.costing.costing_name
            break

    watercosting = get_watercost_for_damage_at_wi(damage_name,
                                                  water_ingress_perc)
    # water_ingress_cost = 0.0
    if watercosting.formula_type == 1:
        water_ingress_cost = watercosting.base_cost * (
            watercosting.coeff1 * di ** 2 + watercosting.coeff2 * di +
            watercosting.coeff3)
    else:
        water_ingress_cost = watercosting.base_cost * watercosting.coeff1 * \
                             di ** watercosting.coeff2

    watercosting_str = '{}'.format(watercosting)

    if out_file:
        out_file.write('\n%f,%f,%f,%s,%f,%s' % (wind_speed, di, water_ratio,
                                                damage_name, water_ingress_cost,
                                                watercosting))
    return water_ratio, damage_name, water_ingress_cost, watercosting_str


# unit tests
if __name__ == '__main__':
    import unittest

    database.configure()
    populate_water_costs(1)


    class MyTestCase(unittest.TestCase):
        def setUp(self):
            print ''

        def test_curves(self):
            import matplotlib.pyplot as plt
            for di in [0.05, 0.15, 0.35, 0.75]:
                m, s = get_curve_coeffs_for(di)
                x = []
                for X in range(100):
                    x.append(scipy.stats.norm.cdf(X, loc=m, scale=s))
                plt.plot(range(100), x, label='%.3f' % di);
            plt.legend()
            plt.show()

        def test_water_costs(self):
            wiarr = [0, 4.3, 5.5, 22.12, 50.0, 67.0, 100.0]
            expectedarr = [0, 0, 5.0, 18.0, 37.0, 67.0, 100.0]
            n = 5000
            for iter in xrange(n):
                i = 0
                for wi in wiarr:
                    watercost = get_watercost_for_damage_at_wi('Wall collapse',
                                                               wi)
                    self.assertNotEqual(watercost, None)
                    self.assertEquals(watercost.wi, expectedarr[i])
                    i += 1


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
