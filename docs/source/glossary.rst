..
    :orphan:
    .. only:: html

        ********
        Glossary
        ********

********
Glossary
********
.. glossary::

    connection
        models a physical structural connection that will bear a simulated load and when broken cause load to be distributed and generate damage outcomes.

    connection type
        a collection of connections sharing the same strength and dead load statistical characteristics and costing area

    connection group, connection type group
        a collection of connection types sharing the same load distribution and costing scenario

    coverage
        a component making up the wall part of the envelope of the model

    |Cpe|
        external pressure coefficient

    |Cpe,str|
        external pressure coefficient for zone component related to rafter

    |Cpe,eave|
        external pressure coefficient for zone component related to eave

    |Cpi|
        internal pressure coefficient

    |Cpi,alpha|
        proportion of the zone's area to which internal pressure is applied

    CV
        coefficient of variation, the ratio of the standard deviation to the mean

    damage index
        The total cost of repairing the building fabric of a group of buildings exposed to severe natural hazard divided by the total cost of fully rebuilding the same assets in the existing locality to current local building regulations.

    differential shielding
        incremental adjustments to be applied to envelope surface pressures to account for different degrees of shielding between envelope surfaces on a single shielded structure.

    fragility, fragility function, fragility curve
        Fragility describes the probability of discrete damage states for a specific
        hazard. Fragility function or curve is referred to a damage model which describes the likelihood of a building of a particular type being damaged to a defined degree for a given level of natural hazard exposure.

    influence coefficient
        coefficient relating a connection to either zone or connection with regard to load distribution

    |Kc|
        action combination factor. This factor is devised to reduce wind pressure when wind pressures from more than one building surfaces, for example walls and roof, contribute significantly to a peak load effect.

    |Ms|
        shielding multiplier. This multiplier represents the reduction in peak 3-second gust velocity at a given height and terrain, caused by the presence of buildings and other obstructions upwind of the site of interest.

    |Mz,cat|
        terrain height multiplier

    patch
        a set of revised influence coefficients for a connection

    |qz|
        free stream wind pressure

    vulnerability, vulnerability function, vulnerability curve
        A damage model, or curve, which describes the average severity of physical economic loss to a group of buildings of a particular type in terms of a damage index with increasing natural hazard exposure.

    zone
        an area of building envelope on which wind pressure acts. Zone is linked with connection with influence coefficient so the wind pressure is transformed to wind load on a connection.


.. |Cpe| replace:: :math:`C_{pe}`
.. |Cpe,str| replace:: :math:`C_{pe,str}`
.. |Cpe,eave| replace:: :math:`C_{pe,eave}`
.. |Cpi| replace:: :math:`C_{pi}`
.. |qz| replace:: :math:`q_{z}`
.. |Kc| replace:: :math:`K_{c}`
.. |Mz,cat| replace:: :math:`M_{z,cat}`
.. |Ms| replace:: :math:`M_{s}`
.. |Cpi,alpha| replace:: :math:`C_{pi,\alpha}`
