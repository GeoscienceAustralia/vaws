.. _logic:
..
  # with overline, for parts
  * with overline, for chapters
  =, for sections
  -, for subsections
  ^, for subsubsections
  ", for paragraphs

*************
Program Logic
*************

This chapter describes the logic of the program.

Overall logic
=============

The program is built around the following high level sequence:

1. Create a group of models

  - For each model

    * :ref:`sample wind direction <sample_wind_direction_section>`
    * :ref:`sample wind profile <sample_wind_profile_section>`
    * :ref:`set terrain height multiplier <set_terrain_height_section>`
    * :ref:`set shielding multiplier <set_shielding_section>`
    * :ref:`set up coverages <set_coverages_section>`
    * :ref:`set up zones <set_zones_section>`
    * :ref:`set up connections <set_connections_section>`

2. Calculate damage indices of the models over a range of wind speeds

  - For each wind speed

    - simulate damage for each model

      * :ref:`compute free stream wind pressure <compute_qz_section>`
      * :ref:`compute zone pressures <compute_zone_pressure_section>`
      * :ref:`compute coverage load and check damage <compute_coverage_load_section>`
      * :ref:`compute connection loads <compute_connection_load_section>`
      * :ref:`check damage of each connection by connection group <check_connection_damage_section>`
      * :ref:`update influence by connection group <update_influence_section>`
      * :ref:`check model collapse <check_model_collapse_section>`
      * :ref:`run debris model and update Cpi <update_cpi_section>`
      * :ref:`compute damage index <compute_damage_index_section>`

    - :ref:`compute damage index increment <compute_damage_increment_section>`

3. :ref:`Fit fragility and vulnerability curves and save outputs <save_output_section>`


Detailed logic
==============

Detailed description of the logic is explained in the following sections by module.

Main module
-----------

.. _compute_damage_increment_section:

compute damage index increment (:py:func:`.compute_damage_increment`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The increment of mean damage index for the group of models is computed. If the computed increment is less than zero, then zero value is returned.


.. _save_output_section:

fit fragility and vulnerability curves and save outputs (:py:func:`.save_results_to_files`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on the simulation results, fragility (see :ref:`fit fragility <fit_fragility_section>`) and vulnerability curves (see :ref:`fit vulnerability <fit_fragility_section>`) are fitted. The output file `results.h5` is also created, and the values of the selected attributes are saved. See :ref:`output file <output_file_section>` for the list of attributes.


House module
------------

.. _sample_wind_direction_section:

sample wind direction (:py:attr:`.House.wind_dir_index`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The wind direction is set up at the time of model creation, and kept constant during the simulation over a range of wind speeds. If `wind_direction` (:numref:`section_main_table`) is 'RANDOM', then wind direction is randomly sampled among the eight directions.

.. _sample_wind_profile_section:

sample wind profile (:py:attr:`.House.profile_index`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A set of gust envelope wind profiles is read from `wind_profiles` (:numref:`section_main_table`). Note that each profile is a normalized profile whose value is normalized to 1 at 10 metres height. An example profile is shown in :numref:`wind_profile_fig`. One profile is randomly chosen for each model and kept constant during the simulation over a range of wind speeds.

.. _set_terrain_height_section:

set terrain height multiplier (:py:attr:`.House.terrain_height_multiplier`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The terrain height multiplier (|Mz,cat|) value at the model height is calculated by the interpolation using the selected wind profile over height.


.. _set_shielding_section:

set shielding multiplier (:py:attr:`.House.shielding_multiplier`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The shielding multiplier (|Ms|) value is determined based on the location. If the value of `regional_shielding_factor` is less or equal to 0.85, which means that the model is located in Australian urban areas, then |Ms| value is sampled based on the proportion of each type of shielding listed in :numref:`shielding_table`. Otherwise, |Ms| value is set to be 1.0, which corresponds to `No shielding`. The proportion of shielding type is adopted following the recommendation in JDH Consulting, 2010 :cite:`JDH2010`.

.. tabularcolumns:: |p{3.0cm}|p{2.5cm}|p{2.5cm}|
.. _shielding_table:
.. csv-table:: Proportion of shielding type
    :header: Type, |Ms| value, Proportion

    Full shielding, 0.85, 63%
    Partial shielding, 0.95, 15%
    No shielding, 1.0, 22%

.. _combination_factor_section:

set action combination factor (:py:attr:`.House.combination_factor`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the AS/NZS 1170.2 :cite:`ASNZS-1170.2`, the action combination factor, :math:`K_{c}` is defined to reduce wind pressure when wind pressures from more than one building surfaces, for example walls and roof, contribute significantly to a peak load effect. When |Cpi| is between -0.2 and +0.2, then the effect is ignored. Otherwise 0.9 is used.

.. _set_coverages_section:

set up coverages (:py:meth:`.House.set_coverages`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The coverages make up the wall part of the envelope of the model. Two failure mechanism are implemented: 1) failure by wind load and 2) failure by windborne debris.

A set of coverage components (:py:class:`.Coverage`) is defined using the information provided in the input files of :ref:`coverages.csv <coverages.csv_section>`, :ref:`coverage_types.csv <coverage_types.csv_section>` and :ref:`coverages_cpe.csv <coverages_cpe.csv_section>`.
The |Cpe| and strength values for each coverage component are sampled when it is defined. The windward direction for each coverage component is assigned from among `windward`, `leeward`, `side1`, or `side2`, which is used in determining the windward direction of dominant opening due to coverage failure.


.. _set_zones_section:

set up zones (:py:meth:`.House.set_zones`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A set of zone components (:py:class:`.Zone`) is defined using the information provided in the input files of :ref:`zones.csv <zones.csv_section>`, :ref:`zones_cpe_mean.csv <zones_cpe_mean.csv_section>`, :ref:`zones_cpe_str_mean.csv <zones_cpe_str_mean.csv_section>`, :ref:`zones_cpe_eave_mean.csv <zones_cpe_eave_mean.csv_section>`, and :ref:`zones_edges.csv <zones_edges.csv_section>`. The |Cpe| value for each zone component is sampled when it is defined.


.. _set_connections_section:

set up connections (:py:meth:`.House.set_connections`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A set of connection components (:py:class:`.Connection`) is defined using the information provided in the input files of :ref:`conn_groups.csv <conn_groups.csv_section>`, :ref:`conn_types.csv <conn_types.csv_section>`, :ref:`connections.csv <connections.csv_section>`, :ref:`influences.csv <influences.csv_section>`, and :ref:`influence_patches.csv <influence_patches.csv_section>`. The strength and dead load values for each connection component are sampled and influence and influence patch for each connection are also defined with reference to either zone or another connection components.

A set of connection type group (:py:class:`.ConnectionTypeGroup`) is also defined, and reference is created to relate a connection component to a connection type group. A connection type group is further divided into sub-group by section in order to represent load distribution area within the same group. For instance roof sheetings on a hip roof are divided into a number of sheeting sub-groups to represent areas divided by roof ridge lines.

set footprint for debris impact (:py:attr:`.House.footprint`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the wind direction of the model is determined, the footprint for debris impact is created by rotating the model footprint with regard to the wind direction as set out in :numref:`rotation_angle_table` (:py:attr:`.House.footprint`). The boundary for debris impact assessment is also defined with the radius of boundary (:py:attr:`.Config.impact_boundary`). Note that all the debris sources are assumed to be located in the East of the model when debris impact to the model is simulated.

.. tabularcolumns:: |p{3.5cm}|p{3.5cm}|
.. _rotation_angle_table:
.. csv-table:: Rotation angle by wind direction
    :header: Wind direction, Rotation angle (deg)

    S or N, 90
    SW or NE, 45
    E or W, 0
    SE or NW, -45

set up coverages for debris impact (:py:attr:`.House.debris_coverages`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the wind direction of the model is determined, the coverages for debris impact are selected based on the wind direction.

.. _compute_qz_section:

calculate free stream wind pressure (:py:meth:`.House.compute_qz`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The free stream wind pressure, |qz| is calculated as :eq:`qz_eq`:

.. math::
    :label: qz_eq

    q_{z} = \frac{1}{2}\times\rho_{air} \times \left( V \times M_{z,cat} \times M_{s} \right)^2 \times 1.0\text{e-}3

where :math:`\rho_{air}`: air density (=1.2 |kgm^3|), :math:`V`: 3-sec gust wind speed at 10m height, |Mz,cat|: terrain-height multiplier, |Ms|: shielding multiplier. Note that :math:`1.0\text{e-}3` is multiplied to convert the unit of the wind pressure from Pa to kPa.


.. _check_model_collapse_section:

check model collapse (:py:meth:`.House.check_collapse`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model is deemed to be collapsed if the proportion of damaged components out of the total components is greater than the value of *trigger_collapse_at*, which is listed in :numref:`conn_groups_table`, for any group with non-zero value of *trigger_collapse_at*.

.. _update_cpi_section:

run debris model and update |Cpi| (:py:meth:`.House.run_debris_and_update_cpi`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the value of *debris* is *True* (see :numref:`section_options_table`), then debris impact to the model is simulated. See :ref:`Debris module <debris_module_section>` for more details.

The internal pressure coefficient, |Cpi| is determined based on :numref:`cpi_no_dominant_table` and :numref:`cpi_dominant_table` depending on the existence of dominant opening by either coverage failure or debris breach, which are revised from Tables 5.1(A) and 5.1(B) of AS/NZS 1170.2 :cite:`ASNZS-1170.2`, respectively.

.. tabularcolumns:: |p{9.0cm}|p{2.0cm}|
.. _cpi_no_dominant_table:
.. csv-table:: |Cpi| for buildings without dominant openings
    :header: Condition, |Cpi|

    All walls equally breached, -0.3
    Two or three windward walls equally breached, 0.2
    Two or three non-windward walls equally breached, -0.3


.. tabularcolumns:: |p{4.0cm}|p{3.5cm}|p{3.5cm}|p{3.5cm}|
.. _cpi_dominant_table:
.. csv-table:: |Cpi| for buildings with dominant openings
    :header: Ratio of dominant opening to total open area (:math:`r`), Dominant opening on windward wall, Dominant opening on leeward wall, Dominant opening on side wall

    :math:`r <` 0.5, -0.3, -0.3, -0.3
    0.5 :math:`\leq r <` 1.5, 0.2, -0.3, -0.3
    1.5 :math:`\leq r <` 2.5, 0.7 |Cpe|, |Cpe|, |Cpe|
    2.5 :math:`\leq r <` 6.0, 0.85 |Cpe|, |Cpe|, |Cpe|
    :math:`r \geq` 6.0, |Cpe|, |Cpe|, |Cpe|

.. _compute_damage_index_section:

compute damage index (:py:meth:`.House.compute_damage_index`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The damage index is calculated over the following steps:

1) calculate sum of damaged area by connection group (:py:meth:`.House.compute_area_by_group`)

2) Apply damage factoring (:py:meth:`.House.apply_damage_factoring`)

In order to avoid double counting of repair cost, damage cost associated with child group(s) will be factored out of damage cost of the parent group as explained in :ref:`3.4.16 <damage_factorings.csv_section>`.

3) Calculate sum of damaged area by damage scenario (:py:meth:`.House.compute_area_by_scenario`)

A damage scenario is assigned to each connection type group as explained in :ref:`3.4.2 <conn_groups.csv_section>`.

4) calculate total damage cost and damage index prior to water ingress (:math:`DI_p`) as :eq:`di_prior`:

.. math::
    :label: di_prior

    DI_p = \frac{\sum_{i=1}^{S}C_i}{R}

where :math:`S`: number of damage scenario, :math:`C_i`: damage cost for :math:`i` th damage scenario, and :math:`R`: total replacement cost.

5) Calculate cost by water ingress damage, :math:`C_{wi}` if required as explained in :ref:`damage due to water ingress <water_ingress_section>`.

6) calculate damage index as :eq:`di`:

.. math::
    :label: di

    DI = \frac{\sum_{i=1}^{S}C_i + C_{wi}}{R}


Zone module (:py:class:`.Zone`)
-------------------------------

sample Cpe (:py:attr:`.Zone.cpe`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The external pressure coefficient, :math:`C_{pe}` is used in computing zone pressures, and is sampled from Type III extreme value distribution (:py:meth:`.stats.sample_gev`) which has the cumulative distribution function and probability density as :eq:`cdf_gev` and :eq:`pdf_gev`, respectively.

.. math::
    :label: cdf_gev

    F(s; k) = \exp(-(1-ks)^{1/k})

.. math::
    :label: pdf_gev

    f(s; a, k) = \frac{1}{a}(1-ks)^{1/k-1} \exp(-(1-ks)^{1/k})

where :math:`s=(x-u)/a`, :math:`u`: location factor (:math:`\in \rm I\!R`), :math:`a`: scale factor (:math:`> 0`), and :math:`k`: shape factor (:math:`k\neq0`).

The mean and standard deviation are calculated as :eq:`mean_sd`:

.. math::
    :label: mean_sd

    \operatorname{E}(X) &= u + \frac{a}{k}\left[1-\Gamma(1+k)\right] \\
    \operatorname{SD}(X) &= \frac{a}{k}\sqrt{\Gamma(1+2k)-\Gamma^{2}(1+k)}


The :math:`u` and :math:`a` can be estimated given :math:`c_v\left(=\frac{SD}{E}\right)` and :math:`k` values as :eq:`a_u`:

.. math::
    :label: a_u

    a &= \operatorname{E} \frac{c_v}{B} \\
    u &= \operatorname{E} - a \times A

where :math:`A=(1/k)\left[1-\Gamma(1+k)\right]` and :math:`B=(1/k)\sqrt{\Gamma(1+2k)-\Gamma^{2}(1+k)}`.

.. _compute_zone_pressure_section:

calculate zone pressure (:py:meth:`.Zone.calc_zone_pressure`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two kinds of zone pressure, |pz| for zone component related to sheeting and batten and |pz,str| for zone component related to rafter, are computed as :eq:`zone_pressure_eq`:

.. math::
    :label: zone_pressure_eq

    p_{z} &= q_{z} \times \left( C_{pe} - C_{pi,\alpha} \times C_{pi} \right) \times D_{s} \times K_{c}\\
    p_{z,str} &= q_{z} \times \left( C_{pe,str} - C_{pi, \alpha} \times C_{pi} - C_{pe,eave} \right) \times D_{s} \times K_{c} \\

where |qz|: free stream wind pressure, |Cpe|: external pressure coefficient, |Cpi|: internal pressure coefficient, |Cpi,alpha|: proportion of the zone's area to which internal pressure is applied, |Cpe,str|: external pressure coefficient for zone component related to rafter, |Cpe,eave|: external pressure coefficient for zone component related to eave, :math:`D_{s}`: differential shielding, and :math:`K_{c}`: action combination factor. The value of differential shielding is determined as explained in :ref:`set differential shielding <differential_shielding_section>`. The value of action combination factor is determined as explained in :ref:`set action combination factor <combination_factor_section>`.

.. _differential_shielding_section:

set differential shielding (:py:attr:`.Zone.differential_shielding`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the value of *differential_shielding* (see :numref:`section_options_table`) is *True*, then differential shielding effect is considered in calculating zone pressure. Based on the recommendations from JDH Consulting, 2010 :cite:`JDH2010`, adjustment for shielding multiplier is made as follows:

- For outer suburban situations and country towns (*building_spacing*\=40m),
    adjust |Ms| to 1.0 except for the leading edges of upwind roofs
- For inner suburban buildings (*building_spacing* =20m) with full shielding (|Ms|\=0.85),
    adjust |Ms| to 0.7 for the leading edges of upwind roofs
- For inner suburban buildings (*building_spacing* =20m) with partial shielding (|Ms|\=0.95),
    adjust |Ms| to 0.8 for the leading edges of upwind roofs
- Otherwise, no adjustment is made.


Coverage module (:py:class:`.Coverage`)
---------------------------------------

.. _compute_coverage_load_section:

calculate coverage load and check damage (:py:meth:`.Coverage.check_damage`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The load applied for each of coverages are calculated as :eq:`coverage_load_eq`:

.. math::
    :label: coverage_load_eq

    L = q_{z} \times \left(C_{pe} - C_{pi} \right) \times A \times K_{c}

where :math:`q_{z}`: free stream wind pressure, |Cpe|: external pressure coefficient, |Cpi|: internal pressure coefficient, :math:`A`: area, and :math:`K_{c}`: action combination factor.

If the calculated load exceeds either positive or negative strength, which represents strength in either direction, then it is deemed to be damaged.


Connection module (:py:class:`.Connection` and :py:class:`.ConnectionTypeGroup`)
--------------------------------------------------------------------------------

.. _compute_connection_load_section:

calculate connection load (:py:meth:`.Connection.check_damage`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The load applied for each of connections are calculated as :eq:`connection_load_eq`:

.. math::
    :label: connection_load_eq

    L_{i} = D_{i} + \sum_{j=1}^{N_{z}} \left(I_{ji} \times A_{j} \times P_{j}\right) + \sum_{j=1}^{N_{c}} \left(I_{ji} \times L_{j}\right)


where :math:`L_{i}`: applied load for :math:`i` th connection, :math:`D_{i}`: dead load of :math:`i` th connection, :math:`N_{z}`: number of zones associated with the :math:`i` th connection, :math:`N_{c}`: number of connections associated with the :math:`i` th connection, :math:`A_{j}`: area of :math:`j` th zone, :math:`P_{j}`: wind pressure on :math:`j` th zone, :math:`I_{ji}`: influence coefficient from :math:`j` th either zone or connection to :math:`i` th connection.

If the load applied for a connection is less than the negative value of its strength, then the connection is considered damaged.

.. _check_connection_damage_section:

check connection damage by connection type group (:py:meth:`.ConnectionTypeGroup.check_damage`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Damage of each connection is checked by connection type group. If the load applied for a connection is less than the negative value of its strength, then the connection is considered damaged. Then damage grid of the connection type group (:py:attr:`.ConnectionTypeGroup.damage_grid`) is updated with the index of the damaged connection, which is later used in updating influence of intact components (:py:meth:`.ConnectionTypeGroup.update_influence`).


.. _update_influence_section:

update influence by connection group (:py:meth:`.ConnectionTypeGroup.update_influence`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The influence coefficient is used to associate one connection with another either zone or connection with regard to load distribution. For instance, if connection 1 has influences of connection 2 and 3 with coefficient 0.5 and 0.5, respectively, then the load on connection 1 is equal to the sum of 0.5 times load on connection 2 and 0.5 times load on connection 3, as shown in :eq:`connection_load_eq`.

Once a connection is damaged, then load on the damaged connection needs to be distributed to other intact connections accordingly, which means that influence set of the connections needs to be updated.

Two types of influence update are implemented:

1. update influence coefficients of the next intact connections for the distribution of load on the damaged connection, when `dist_dir` is either `col` or `row` (:py:meth:`.ConnectionTypeGroup.update_influence`)

Given the damage of connection of either sheeting and batten connection type group, the influence coefficient will be distributed evenly to the next intact connections of the same type to the distribution direction (*dist_dir* listed in :numref:`conn_groups_table`). If both the next connections, which are left and right if *dist_dir* is 'row' or above and below if 'col', of the damaged connection are intact, then the half of the load is distributed to the each of next intact connection. Otherwise, the full load of the damaged connection is distributed to the intact connection.

2. replace the existing influence set with new one, when `dist_dir` is `patch` (:py:meth:`.ConnectionTypeGroup.update_influence_by_patch`)

Unlike sheeting and batten, a connection of rafter group fails, then influence set of each connection associated with the failed connection are replaced with a new set of influence, which is termed "patch". In the current implementation, the patch is defined for a single failed connection. Thus the failure order of the connections may make difference in the resulting influences as shown in :numref:`patch_example_table`.

.. _patch_example_table:
.. csv-table::  Example of how patch works
    :header: Failed connection, Connection, "Patch (connection: influence coeff.)"

    1, 3, "1:0.0, 2:0.5, 3:0.5"
    2, 3, "1:0.5, 2:0.0, 3:1.0"
    1 and then 2, 3, "1:0.0, 2:0.0, 3:1.0"
    2 and then 1, 3, "1:0.0, 2:0.0, 3:0.5"


.. _debris_module_section:

Debris module (:py:class:`.Debris`)
-----------------------------------

The methdology of modelling damage from wind-borne debris implemented in the code is described in Holmes et al., 2010 :cite:`Holmes2010` and Wehner et al., 2010 :cite:`Wehner2010a`. The debris damage module consists of four parts: 1) debris source generation, 2) debris generation, 3) debris trajectory, and 4) debris impact.

debris source generation
^^^^^^^^^^^^^^^^^^^^^^^^

The debris sources are generated by calling :py:func:`.create_sources`, which requires a number of parameters as shown in the :numref:`debris_source_parameters_fig`.

.. _debris_source_parameters_fig:
.. figure:: _static/image/debris_source_parameters.png
    :align: center
    :width: 80 %

    Distribution of debris sources with parameters

Depending on the value of *staggered_sources*, different layout of debris sources can be generated as shown in :numref:`debris_source_staggered_true_fig` and :numref:`debris_source_staggered_false_fig`.

.. _debris_source_staggered_true_fig:
.. figure:: _static/image/source_staggered_true.png
    :align: center
    :width: 70 %

    Distribution of debris sources generated with debris_radius = 100.0 (m), debris_angle = 45.0 (deg), debris_space = 20.0 (m), and staggered_sources = *True*.

.. _debris_source_staggered_false_fig:
.. figure:: _static/image/source_staggered_false.png
    :align: center
    :width: 70 %

    Distribution of debris sources generated with debris_radius = 100.0 (m), debris_angle = 45.0 (deg), debris_space = 20.0 (m), and staggered_sources = *False*.


debris generation
^^^^^^^^^^^^^^^^^

For each wind speed, a group of debris items are generated by calling :py:func:`.generate_debris_items`. The mean number of debris items to be generated (:math:`N_{mean}`) is calculated by :eq:`number_of_debris_items_eq`.

.. math::
    :label: number_of_debris_items_eq

    N_{mean} = \operatorname{nint} \left( \Delta{DI} \times N_{items} \right)

where :math:`N_{items}`: number of debris items per source defined in :ref:`3.1.3 <debris_section>`, :math:`\Delta{DI}`: increment in damage index from previous wind step, and :math:`\operatorname{nint}`: nearest integer function.

The number of generated debris items is assumed to follow the Poisson distribution with parameter :math:`\lambda=N_{mean}`. For each debris source, the number of generated debris items is randomly sampled from the distribution, and debris type is randomly chosen as many as number of items with probability proportional to the ratio of each type defined in :numref:`debris_item_table`. The debris types are provided in the section of :ref:`3.2 debris.csv <debris.csv_section>`.

debris trajectory
^^^^^^^^^^^^^^^^^

For each generated debris item, mass (:py:attr:`.Debris.mass`), frontal area (:py:attr:`.Debris.frontal_area`), and flight time (:py:attr:`.Debris.flight_time`) are sampled from the lognormal distribution with parameter values provided in :ref:`3.1.3 <debris_section>` and :ref:`3.2 <debris.csv_section>`. The flight distance (:py:attr:`.Debris.flight_distance`) is calculated based on the methodology presented in the Appendix of Lin and Vanmarcke, 2008 :cite:`Lin2008`. Note that the original fifth polynomial functions are replaced with quadratic one with the coefficients as listed in :numref:`flight_distance_table`. The computed flight distance by debris type using the fifth and quadratic polynomials is shown in :numref:`flight_distance_fig`.

.. _flight_distance_fig:
.. figure:: _static/image/flight_distance.png
    :align: center
    :width: 80 %

    Flight distance of debris item

.. tabularcolumns:: |p{3.5cm}|p{3.5cm}|p{3.5cm}|
.. _flight_distance_table:
.. csv-table:: Coefficients of quadratic function for flight distance computation by debris type
    :header: Debris type, Linear coeff., Quadratic coeff.

    Compact, 0.011, 0.2060
    Rod, 0.2376, 0.0723
    Sheet, 0.3456, 0.072

The probability distribution of point of landing of the debris in a horizontal plane is assumed to follow a bivariate normal distribution as :eq:`bivariate_normal`.

.. math::
    :label: bivariate_normal

    f_{x,y} = \frac{1}{2\pi\sigma_x\sigma_y}\exp\left[-\frac{(x-d)^2}{2\sigma_x^2}-\frac{y^2}{2\sigma_y^2}\right]


where :math:`x` and :math:`y` are the coordinates of the landing position of the debris, :math:`\sigma_x` and :math:`\sigma_y`: standard deviation for the coordinates of the landing position, and :math:`d`: expected flight distance. The value of :math:`\sigma_x` and :math:`\sigma_y` are set to be :math:`d/3` and :math:`d/12`, respectively.

Following Lin and Vanmarcke 2008, the ratio of horizontal velocity of the windborne debris object to the wind gust velocity is modelled as a random variable with a Beta distribution as :eq:`beta_dist`.

.. math::
    :label: beta_dist

    \frac{u_m}{V_s} \sim Beta(\alpha, \beta)

where :math:`u_m`: the horizontal velocity of the debris object, :math:`V_s`: the local (gust) wind speed, :math:`\alpha` and :math:`\beta` are two parameters of the Beta distribution and estimated as :eq:`beta_dist_a_b`.

.. math::
    :label: beta_dist_a_b

    \alpha &= E \times \nu \\
    \beta &= \nu \times (1 - E)

where :math:`E`: the expected value and :math:`\nu=\alpha + \beta`.

The expected value (:math:`E`) and the parameter (:math:`\nu`) are assumed to be as :eq:`velocity_debris`.

.. math::
    :label: velocity_debris

    E &= 1-\exp\left(-b\sqrt{x}\right) \\
    \nu &= \max\left[\frac{1}{E}, \frac{1}{1-E}\right] + 3.0

where :math:`x`: the flight distance, :math:`b`: a dimensional parameter calucalted as :eq:`b`. If :math:`E` is 1, then :math:`\alpha` and :math:`\beta` are assigned with 3.996 and 0.004, respectively.

.. math::
    :label: b

    b = \sqrt{\frac{\rho_aC_{D,av}A}{m}}

where :math:`\rho_a`: the air density, :math:`C_{D,av}`: an average drag coefficient, :math:`A`: the frontal area, and :math:`m`: the mass of the object.

The momentum :math:`\xi` (:py:attr:`.Debris.momentum`) is calculated using the sampled value of the ratio, :math:`\frac{u_m}{V_s}` as :eq:`momentum`.

.. math::
    :label: momentum

    \xi = \left(\frac{u_m}{V_s}\right) \times m \times V_s

debris impact
^^^^^^^^^^^^^

Either if the landing point is within the footprint of the model or if the line linking the source to the landing point intersects with the footprint of the model and the landing point is within the boundary, then it is assumed that an impact has occurred. The criteria of debris impact is illustrated in the :numref:`debris_impact_criteria_fig` where blue line represents debris trajectory with impact while red line represents one without impact.

.. _debris_impact_criteria_fig:
.. figure:: _static/image/debris_impact.png
    :align: center
    :width: 70 %

    Graphical presentation of debris impact criteria

Based on the methodology presented in HAZUS and Lin and Vanmacke (2008), the number of impact :math:`N` is assumed to follow a Poisson distribution as :eq:`poisson_eqn`.

.. math::
    :label: poisson_eqn

    N &\sim \operatorname{Pois}(\lambda) \\
    \lambda &= N_v \cdot q \cdot F_{\xi}(\xi>\xi_d)

where :math:`N_v`: number of impacts at a single wind speed, :math:`q`: proportion of coverage area out of the total area of envelope, :math:`F_{\xi}`: the cumulative distribution of momentum, and :math:`\xi_d`: threshold of momentum or energy for damage of the material of the coverage.

The probability of damage can be calculated based on the Poisson distribution as :eq:`p_d`.

.. math::
    :label: p_d

    P_D = 1 - P(N=0) = 1-\exp\left[-\lambda\right]

:math:`q` and :math:`F_{\xi}(\xi>\xi_d)` are estimated for each coverage. 

If the material of the coverage is glass, then :math:`P_D` is computed and compared against a random value sampled from unit uniform distribution to determine whether the coverage is damaged or not. If the coverage is damaged, then damaged area is set to be equal to the coverage area.
For coverage with non-glass material, a random value of number of impact is sampled from the Poisson distribution with :math:`\lambda`, and damaged coverage area is then computed assuming that the area requiring repairs due to debris impact is 1.

Since version 2.2 of the code, a Monte Carlo based approach is implemented (:py:meth:`.Debris.check_coverages`). For each debris item, a coverage component is chosen based on the ratio of the area out of the total area of the coverages, once debris impact is assumed to occur. If the computed momentum exceeds the momentum capacity of the coverage, then damaged coverage area is computed depending on the material of the coverage as explained above.

damage_costing module (:py:class:`.Costing`)
--------------------------------------------

.. _water_ingress_section:

damage due to water ingress
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The damage cost induced by water ingress is estimated over the following three steps:

1) estimate amount of water ingress (:py:func:`.compute_water_ingress_given_damage`)

The amount of water ingress is estimated based on the relationship between water ingress and wind speed, which is listed in :numref:`section_water_ingress_table`. The estimated damage index prior to water ingress is used to choose the right curve as shown in :numref:`water_ingress_fig`.

2) determine damage scenario (:py:meth:`.House.determine_scenario_for_water_ingress_costing`)

The damage scenario for water ingress is determined based on the order of damage scenario as listed in :numref:`damage_costing_data_table`. One damage scenario is selected by the order among the damage scenarios with which damage area of connection associated is greater than zero. When the damage index is zero (or no connection damage yet), then damage scenario of 'WI only' is used.

3) calculate cost for water ingress damage (:py:meth:`.House.compute_water_ingress_cost`)

The cost for water ingress damage is estimated using the data provided in :ref:`3.4.17 <water_ingress_costing_data.csv_section>`. The example plot for the scenario of *loss of roof sheeting* is shown in :numref:`water_ingress_cost_fig`. The cost for water ingress damage is estimated using the curve for water ingress closest to the estimated amount of water ingress.


.. _water_ingress_cost_fig:
.. figure:: _static/image/wi_costing_roof_sheeting.png
    :align: center
    :width: 80 %

    Relationship between cost due to water ingress damage and damage index

Curve module
------------

.. _fit_fragility_section:

fit fragility (:py:func:`.curve.fit_fragility_curves`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The probability of exceeding a damage state :math:`ds` at a wind speed :math:`x` is calculated as :eq:`fragility_eq`:

.. math::
    :label: fragility_eq

    P\left(DS \geq ds | x \right) = \frac {\sum_{i=1}^N\left[DI_{i|x} \geq t_{ds}\right]}{N}

where :math:`N`: number of models, :math:`DI_{i|x}`: damage index of :math:`i` th model at the wind speed :math:`x`, and :math:`t_{ds}`: threshold for damage state :math:`ds`.

Then for each damage state, a curve of cumulative lognormal distribution :eq:`cdf_lognormal` is fitted to the computed probabilities of exceeding the damage state.


.. _fit_vulnerability_section:

fit vulnerability (:py:func:`.curve.fit_vulnerability_curve`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two types of curves are used to fit the results of damage indices of models: a cumulative lognormal distribution (:eq:`cdf_lognormal`, :py:func:`.curve.vulnerability_lognormal`) and Weibull distribution (:eq:`cdf_weibull_oz`, :py:func:`.curve.vulnerability_weibull`).



.. |Cpe| replace:: :math:`C_{pe}`
.. |Cpe,str| replace:: :math:`C_{pe, str}`
.. |Cpe,eave| replace:: :math:`C_{pe, eave}`
.. |Cpi| replace:: :math:`C_{pi}`
.. |Cpi,alpha| replace:: :math:`C_{pi,\alpha}`
.. |qz| replace:: :math:`q_{z}`
.. |Mz,cat| replace:: :math:`M_{z,cat}`
.. |Ms| replace:: :math:`M_{s}`
.. |pz| replace:: :math:`p_{z}`
.. |pz,str| replace:: :math:`p_{z,str}`
.. |kgm^3| replace:: :math:`\text{kg}/\text{m}^{3}`

..
  .. literalinclude:: ../../vaws/model/debris.py
     :language: python
     :pyobject: Debris.create_sources
