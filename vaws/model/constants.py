G_CONST = 9.81  # acceleration of gravity (m/sec^2)
RHO_AIR = 1.2  # air density (kg/m^3)

# rotation angle of footprint by wind direction index
ROTATION_BY_WIND_IDX = {0: 90.0, 4: 90.0,  # S, N
                        1: 45.0, 5: 45.0,  # SW, NE
                        2: 0.0, 6: 0.0,  # E, W
                        3: -45.0, 7: -45.0}  # SE, NW

# wind direction
WIND_DIR = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'RANDOM']

# debris types
DEBRIS_TYPES_KEYS = ['Rod', 'Compact', 'Sheet']

# debris attributes
DEBRIS_TYPES_ATTS = ['mass', 'frontal_area', 'cdav', 'ratio', 'flight_time']

BLDG_SPACING = [20.0, 40.0]  # building spacing for debris impact model

FLAGS_PRESSURE = ['cpe_str', 'cpe']
FLAGS_DIST_DIR = ['row', 'col', 'patch', 'none']
COVERAGE_FAILURE_KEYS = ['failure_strength_in', 'failure_strength_out',
                         'failure_momentum']

# CPI table
DOMINANT_OPENING_RATIO_THRESHOLDS = [0.5, 1.5, 2.5, 6.0]
CPI_TABLE_FOR_DOMINANT_OPENING = \
    {0: {'windward': -0.3, 'leeward': -0.3, 'side1': -0.3, 'side2': -0.3},
     1: {'windward': 0.2, 'leeward': -0.3, 'side1': -0.3, 'side2': -0.3},
     2: {'windward': 0.7, 'leeward': 1.0, 'side1': 1.0, 'side2': 1.0},
     3: {'windward': 0.85, 'leeward': 1.0, 'side1': 1.0, 'side2': 1.0},
     4: {'windward': 1.0, 'leeward': 1.0, 'side1': 1.0, 'side2': 1.0}}

# prob. dist. and shielding categories
SHIELDING_MULTIPLIERS = {'full': 0.85,
                         'partial': 0.95,
                         'no': 1.0}
SHIELDING_MULTIPLIERS_KEYS = ['full', 'partial', 'no']
SHIELDING_MULTIPLIERS_PROB = [0.63, 0.15, 0.22]

# debris flight distance
FLIGHT_DISTANCE_COEFF = {2: {'Compact': [0.011, 0.2060],
                             'Sheet': [0.3456, 0.072],
                             'Rod': [0.2376, 0.0723]},
                         5: {'Compact': [0.405, -0.036, -0.052, 0.008],
                             'Sheet': [0.456, -0.148, 0.024, -0.0014],
                             'Rod': [0.4005, -0.16, 0.036, -0.0032]}}

FLIGHT_DISTANCE_POWER = {2: [1, 2],
                         5: [2, 3, 4, 5]}


COSTING_FORMULA_TYPES = [1, 2]

VUL_DIC = {'Capital_city': {'alpha': 0.1586, 'beta': 3.8909},
           'Tropical_town': {'alpha': 0.1030, 'beta': 4.1825}}
