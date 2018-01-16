.. _use_of_the_GUI:

**************
Use of the GUI
**************



Overall logic
=============

The VAWS tool takes a component-based approach to modelling building vulnerability. It is based on the premise that overall building damage is strongly related to the failure of key connections.

The tool generates a building model by randomly selecting parameter values from predetermined probability distributions using a Monte Carlo process. Values include component and connection strengths, external pressure coefficients, shielding coefficients, wind speed profile, building orientation, debris damage parameters, and component masses.

Then, for progressive gust wind speed increments, it calculates the forces in all critical connections using influence coefficients, assesses which connections have failed and translates these into a damage scenario and costs the repair. Using the repair cost and the full replacement cost, it calculates a damage index for each wind speed.

Key features
============

* Component-based approach:

  A house is modelled consisting of a large number of components, and overall damage is estimated based on damage of each of the components.

* Uncertainty captured through a Monte-Carlo process:

  Various uncertainties affecting house performance are modelled through a monte-carlo process.

* Inclusion of debris and water ingress induced damages:

  In addition to the damage to the connections by wind loads, debris and water ingress induced damages are modelled.

* Internal pressurisation:

  Internal pressure coefficients are calculated at each wind speed following the procedures of AS/NZS 1170.2 (Standards Australia, 2011) using the modelled envelope failures to determine envelope permeability.


Key uncertainties
=================

The Monte Carlo process capture a range of variability in both wind loading and component parameters. The parameter values are sampled for each model and kept the same through the wind steps.

- Wind direction

  For each house, its orientation with respect to the wind is chosen from the eight cardinal directions either randomly, or by the user.

- Gust wind profile

  Variation in the profile of wind speed with height is captured by the random sampling of a profile from a suite of user-provided profiles.

- Pressure coefficients for zone and coverage

  Pressure coefficients for different zones of the house surfaces envelope are randomly chosen from a Type III (Weibull) extreme value distribution with specified means for different zones of the house envelope, and specified coefficients of variation for different load effects.

- Construction level

  Multiple construction levels can be defined with mean and cov factors which will be used to adjust the mean and cov of distribution of connection strength.

- Strength and dead load

  Connection strengths and dead loads for generated houses are sampled from lognormal probability distributions.

Caveats and limitations
=======================

VAWS has been designed primarily as a tool for assessing vulnerability of houses to wind hazard. The simulation outcomes should be interpreted as vulnerability of a group of similar houses on average, even though an individual house is modelled. In other words, the tool is not capable of predicting performance of each individual house for a specific wind event.


