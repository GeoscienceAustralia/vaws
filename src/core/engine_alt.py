import math
import numpy as np
from stats import compute_logarithmic_mean_stddev

# engine.seed
rnd_state = None


def seed(value):
    global rnd_state
    rnd_state = np.random.RandomState(seed=value)


def lognormal(mean, stddev):
    mu, std = compute_logarithmic_mean_stddev(mean, stddev)
    return rnd_state.lognormal(mu, std, size=1)


def poisson(mean):

    return rnd_state.poisson(mean, size=1)


def debris_generate_item(wind_speed,
                         x_cord, y_cord,
                         flight_time_mean, flight_time_stddev,
                         debris_type_id, cdav,
                         mass_mean, mass_stddev,
                         fa_mean, fa_stddev):

    """

        Args:
            wind_speed:
            x_cord:
            y_cord:
            flight_time_mean:
            flight_time_stddev:
            debris_type_id: 0, 1, 2
            cdav:
            mass_mean:
            mass_stddev:
            fa_mean:
            fa_stddev:
            rnd_state:

        Returns:

    """

    mu_mass, std_mass = compute_logarithmic_mean_stddev(mass_mean, mass_stddev)
    mass = rnd_state.lognormal(mu_mass, std_mass)

    mu_fa, std_fa = compute_logarithmic_mean_stddev(fa_mean, fa_stddev)
    fa = rnd_state.lognormal(mu_fa, std_fa)

    mu_flt_time, std_flt_time = compute_logarithmic_mean_stddev(flight_time_mean,
                                                                      flight_time_stddev)
    flight_time = rnd_state.lognormal(mu_flt_time, std_flt_time)

    c_t = 9.81 * flight_time / wind_speed
    c_k = 1.2 * wind_speed * wind_speed / (2 * 9.81 * mass / fa)
    c_kt = c_k * c_t

    param1 = {0: 0.2060, 1: 0.072, 2: 0.0723}
    param2 = {0: 0.011, 1: 0.3456, 2: 0.2376}

    flight_distance = math.pow(wind_speed, 2) / 9.81 * (1.0 / c_k) * (
        param1[debris_type_id] * math.pow(c_kt, 2) +
        param2[debris_type_id] * c_kt)

    # Sample Impact Location
    sigma_x = flight_distance / 3.0
    sigma_y = flight_distance / 12.0
    rho_xy = 1.0

    cov_matrix = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
                  [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]

    # try:
    (x, y) = rnd_state.multivariate_normal(mean=[0.0, 0.0], cov=cov_matrix)
    # except RuntimeWarning:
    # print('cov_matrix: {}'.format(cov_matrix))
    # translate to footprint coords
    item_x = x_cord - flight_distance + x
    item_y = y_cord + y

    # calculate um/vs, ratio of horizontal vel. of debris to local wind speed
    rho_a = 1.2  # air density
    param_b = math.sqrt(rho_a * cdav * fa / mass)
    beta_mean = 1 - math.exp(-param_b * math.sqrt(flight_distance))

    try:
        dispersion = max(1.0 / beta_mean, 1.0 / (1.0 - beta_mean)) + 3.0
    except ZeroDivisionError:
        dispersion = 4.0
        beta_mean -= 0.001

    beta_a = beta_mean * dispersion
    beta_b = dispersion * (1.0 - beta_mean)

    try:
        item_momentum = mass * wind_speed * rnd_state.beta(beta_a, beta_b)
    except ValueError:
        print('A{}:B{}:mean{}, dispersion{}'.format(beta_a, beta_b, beta_mean, dispersion))

    return item_momentum, item_x, item_y




