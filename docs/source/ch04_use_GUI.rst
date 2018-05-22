.. _use_of_the_GUI:

..
  # with overline, for parts
  * with overline, for chapters
  =, for sections
  -, for subsections
  ^, for subsubsections
  ", for paragraphs

**************
Use of the GUI
**************

This chapter provides an overview of the GUI and instructions on how to run simulations using the GUI.


Structure
=========

The main window is logically separated into distinct areas of functionality as shown as :numref:`main_window_by_function_fig`.

.. _main_window_by_function_fig:
.. figure:: _static/image/main_window_by_function.png
    :align: center
    :width: 80 %

    Program main window consisting of five areas by functionality as shown as dotted box


Top
---

At the top of the main window, tool bar with buttons is located. The file menu and action corresponding to each of the buttons is set out in the :numref:`toolbar_table`.

.. tabularcolumns:: |p{1.0cm}|p{4.0cm}|p{7.5cm}|
.. _toolbar_table:
.. csv-table:: Buttons in the toolbar
    :header: Button, Menu, "Action"

    .. figure:: _static/image/filenew.png, File -> New, Create a new scenario with default setting
    .. figure:: _static/image/fileopen.png, File -> Open Scenario, Open an existing configuration file
    .. figure:: _static/image/filesave.png, File -> Save, Save current scenario
    .. figure:: _static/image/filesaveas.png, File -> Save As, Save current scenario to a new configuration file
    .. figure:: _static/image/forward.png, Simulator -> Run, Run the scenario
    .. figure:: _static/image/filequit.png, Simulator -> Stop, Stop the simulation when in progress
    .. figure:: _static/image/home2.png, Model -> House Info, Show the current house information including wall coverages

Top left
--------

The top left panel contains simulation settings across five tabs: Scenario, Debris, Construction, Water, and Options, where parameter values for a simulation can be set. The details of each of the tab can be found in :ref:`3.1 <configuration_file>`.

There are three test button across the tabs.

The Test button in the Debris tab demonstrates debris generation function at a selected wind speed. Once the wind speed is determined, then a window showing debris traces from sources are displayed as shown as :numref:`test_debris_fig`. The vulnerability curve used in the test function is selected from the curves shown in :numref:`test_debris_vul_fig`, whereas other parameter values can be changed through the GUI.

.. _test_debris_fig:
.. figure:: _static/image/test_debris.png
    :align: center
    :width: 80 %

    Test of debris generation function: debris generated at 50 m/s in region of Capcital_city

.. _test_debris_vul_fig:
.. figure:: _static/image/test_debris_vul.png
    :align: center
    :width: 80 %

    Vulnerability curves implemented in the debris test function using the parameter values listed in :numref:`vul_parameters_table`.

.. tabularcolumns:: |p{4.0cm}|p{4.0cm}|p{4.0cm}|
.. _vul_parameters_table:
.. csv-table:: Parameter values for vulnerability curves :eq:`cdf_weibull_oz` used in the debris test
    :header: name, |alpha|, |beta|
    :widths: 30, 30, 30

    Capital_city, 0.1585, 3.8909
    Tropical_town, 0.1030, 4.1825

The Test button in the Construction tab shows distribution of connection strength of the selected connection type. Example of sampled strength of batten type is shown in :numref:`test_construction_fig`.

.. _test_construction_fig:
.. figure:: _static/image/test_construction.png
    :align: center
    :width: 80 %

    Distribution of sampled strength of the selected connection type

The Test button in the Water tab shows relationship between percentage of water ingress and wind speed for a range of damage index as shown in :numref:`test_water_ingress_fig`.

.. _test_water_ingress_fig:
.. figure:: _static/image/test_water_ingress.png
    :align: center
    :width: 80 %

    Relationship between percentage of water ingress and wind speed


Bottom left
-----------

The bottom left panel contains data browser of house and global data. This panel contains two tabs at the top: House and Global. The House tab has five tabs at the bottom: Connections, Types, Groups, Zones, and Damage, as shown in :numref:`house_tab_fig`. :numref:`house_table` sets out corresponding input file and section for each of the tabs.

.. tabularcolumns:: |p{2.0cm}|p{4.0cm}|p{7.5cm}|
.. _house_table:
.. csv-table:: House data
    :header: Tab name, Input file, Section

    Connections, connections.csv, :ref:`3.4.4 connections.csv <connections.csv_section>`
    Types, conn_types.csv, :ref:`3.4.3 conn_types.csv <conn_types.csv_section>`
    Groups, conn_groups.csv, :ref:`3.4.2 conn_groups.csv <conn_groups.csv_section>`
    Zones, zones.csv, :ref:`3.4.5 zones.csv <zones.csv_section>`
    Damage, damage_costing_data.csv, :ref:`3.4.15 damage_costing_data.csv <damage_costing_data.csv_section>`

.. _house_tab_fig:
.. figure:: _static/image/house_tab.png
    :align: center
    :width: 80 %

    House data tab showing connections information

The Global tab has two tabs at the bottom: Boundary Profile and Debris, as shown in :numref:`global_tab_fig`. In the Boundary Profile tab, gust envelope profiles of selected wind profiles is displayed. Details about the gust envelope profiles can be found in :ref:`3.3 <envelope_profiles_section>`. In the Debris tab, parameter values for debris model listed in the debris.csv (:ref:`3.1.3 <debris.csv_section>`) is displayed. Note that the contents of both tabs are to be changed dynamically upon different selection of wind profile file (Wind Profiles) and debris region (Region).

.. _global_tab_fig:
.. figure:: _static/image/global_tab.png
    :align: center
    :width: 80 %

    Global data tab showing boundary profiles information

Bottom right
------------

The bottom right panel shows input data of influence coefficients and simulation results. This panel consists of five tabs: Influences, Patches, Results, Damages, and Curves, among which Results, Damages, and Curves are empty until a simulation is completed.

Influences tab
^^^^^^^^^^^^^^
Once connection id is set by the slider at the top, then the selected connection (coloured in skyblue) and its associated either zone or connection (coloured in orange) with influence coefficient are shown as :numref:`influences_fig`.

.. _influences_fig:
.. figure:: _static/image/influences.png
    :align: center
    :width: 80 %

    Display of influence coefficient of connection id 27, which is 1.0 with Zone C3.

Patches tab
^^^^^^^^^^^
The Patches tab shows the influence coefficient of connection when associated connection is failed. Once failed connection (coloured in gray) and connection id (coloured in skyblue) are set, then associated either zone or connection (coloured in orange) with influence coefficient is shown as :numref:`patches_fig`.

.. _patches_fig:
.. figure:: _static/image/patches.png
    :align: center
    :width: 80 %

    Display of influence coefficient of connection 125 when connection 124 is failed.

Results tab
^^^^^^^^^^^

The Results tab shows the results of simulation in four sub-windows: Zones, Connections, Type Strengths, and Type Damage.

The Zones window shows sampled Cpe values for each of the zones for each realisation of the simulation models as shown as :numref:`results_zones_fig`. The first string at the *House* column refers to model index, and the string before and after slash refer to wind direction and construction quality level, respectively.

.. _results_zones_fig:
.. figure:: _static/image/results_zones.png
    :align: center
    :width: 80 %

    Display of Cpe values for each zone

Likewise, the Connections window shows the results of each connections such as sampled strength and dead load as shown as :numref:`results_connections_fig`.

.. _results_connections_fig:
.. figure:: _static/image/results_connections.png
    :align: center
    :width: 80 %

    Display of strength, dead load, and failure wind speed for each connection

The Type Strengths window show distribution of connection strength by connection type as shown as :numref:`results_type_strength_fig`.

.. _results_type_strength_fig:
.. figure:: _static/image/results_type_strength.png
    :align: center
    :width: 80 %

    Display of distribution of sampled connection strength by connection type

The Type Damage window shows distribution of speeds at which connection fails by connection type as shown as :numref:`results_type_damage_fig`.

.. _results_type_damage_fig:
.. figure:: _static/image/results_type_damage.png
    :align: center
    :width: 80 %

    Display of distribution of failure wind speed by connection type

Damages tab
^^^^^^^^^^^

The Damages tab shows heatmap by connection type group such as Sheeting, Batten, and Rafter as shown in :numref:`damages_heatmap_fig`. The heatmap averaged across models is shown by default, and heatmap for individual model can be displayed by moving the slide at the top.

.. _damages_heatmap_fig:
.. figure:: _static/image/damages_heatmap.png
    :align: center
    :width: 80 %

    Heatmap of failure wind speed averaged across models for batten group


Curves tab
^^^^^^^^^^

The Curves tab shows curves in four sub-windows: Vulnerability, Fragility, Water Ingress, and Debris. The Vulnerability window shows a scatter plot of damage indices at each wind speed along with two fitted vulnerability curves, one of which is cumulative lognormal distribution function as :eq:`cdf_lognormal` and the other one is cumulative Weibull distribution as :eq:`cdf_weibull_oz`. The estimated parameter values are displayed at the top.


.. math::
    :label: cdf_lognormal

    F_X(x; m, \sigma) = \Phi\left( \frac{\ln (x / m)} \sigma \right)

where :math:`\Phi`: the cumulative distribution function of the standard normal distribution, :math:`m`: median, and :math:`\sigma`: logarithmic standard deviation.

.. math::
    :label: cdf_weibull_oz

    F(x; \alpha, \beta) = 1- \exp\left[-\left(\frac{x}{e^\beta}\right)^\frac{1}{\alpha}\right]


An example plot is shown in :numref:`curves_vulnerability_fig`.


.. _curves_vulnerability_fig:
.. figure:: _static/image/curves_vulnerability.png
    :align: center
    :width: 80 %

    Plot in the Vulnerability window

The Fragility window shows fragility curves for discrete damage states, which are fitted to cumulative lognormal distribution function, as shown in :numref:`curves_fragility_fig`.

.. _curves_fragility_fig:
.. figure:: _static/image/curves_fragility.png
    :align: center
    :width: 80 %

    Plot in the Fragility window

The Water Ingress window shows a scatter plot of the costing associated with water ingress along wind speed, as shown in :numref:`curves_water_ingress_fig`.

.. _curves_water_ingress_fig:
.. figure:: _static/image/curves_water_ingress.png
    :align: center
    :width: 80 %

    Plot in the Water Ingress window

The	Debris window shows 1) number of generated debris items, 2) number of impacted debris items, and 3) proportion of models breached by debris along the range of wind speed, as shown in :numref:`curves_debris_fig`.

.. _curves_debris_fig:
.. figure:: _static/image/curves_debris.png
    :align: center
    :width: 80 %

    Plot in the Debris window

Bottom
------

At the bottom of the main window, configuration file name and status of current simulation are displayed as shown in :numref:`bottom_status_fig`.

.. _bottom_status_fig:
.. figure:: _static/image/bottom_status.png
    :align: center
    :width: 80 %

    Display of configuration file and status of simulation

Running simulations
===================

The simulation can be run by either 1) creating a new scenario or 2) loading a saved scenario.

Creating a new scenario
-----------------------

User can create a new scenario by clicking the *New* button, as shown in :numref:`toolbar_table`. The new scenario comes with a set of input files as an template. Once all the setting are set, then user can save the configuration file to a folder where the template input files will be saved. User need to make changes to each of the input files as required.

Loading a saved scenario
------------------------

User can load a saved scenario file (e.g., default.cfg). A collection of input files should be located in the directory with the folder structure described in :numref:`folder_structure`. User may make some changes on the settings through GUI. Once all the settings are set, then simulation can be run by clicking the *Run* button, as shown in
:numref:`toolbar_table`. Once the simulation is completed, user can either exit the program or save the current setting to a different scenario, as shown in :numref:`save_as_new_cfg_fig`.


.. _save_as_new_cfg_fig:
.. figure:: _static/image/save_as_new_cfg.png
    :align: center
    :width: 80 %

    Save as a new scenario



.. |alpha| replace:: :math:`\alpha`
.. |beta| replace:: :math:`\beta`

