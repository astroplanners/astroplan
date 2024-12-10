.. doctest-skip-all

.. _iers:

**************************************************
What are the IERS tables and how do I update them?
**************************************************

Contents
========

* :ref:`iers-Background`
* :ref:`iers-What_are_IERS_Bulletins_A_and_B`
* :ref:`iers-How_do_I_get_IERS_Bulletin_A_for_astroplan?`

.. _iers-Background:

Background
==========

Keeping accurate, consistent time systems is rough. We intuitively want
time to be described by a continuous cyclic clock like the clock on your
wall or the calendar, but the Earth's rate of rotation varies measurably
on human timescales. As a result, the time measured by an atomic clock and
the time that you would infer from measuring how long it has been since the
last solar noon are constantly becoming offset from one another with
stochastic changes in Earth's moment of inertia and slowly acting tidal forces.

For this reason, there is a time system that keeps track of seconds like an
atomic clock, TAI, a more commonly used one that works like an atomic clock
with a varying offset of some integer number of seconds, UTC, and one that
matches up with the Earth's rotation, UT1. The difference between UT1 and UTC is
constantly changing as the Earth's rotation changes and as leap seconds get
added to UTC to compensate.

In order to accurately predict the apparent position of a celestial object from
the Earth, for example in altitude and azimuth coordinates, one needs to know
the orientation of the Earth at any point in time, and since the Earth's
rotation does not change in a predictable way, we must rely on observations
of the Earth's orientation as a function of time to accurately calculate
UT1-UTC.

.. _iers-What_are_IERS_Bulletins_A_and_B:

What are IERS Bulletins A and B?
================================

The `International Earth Rotation and Reference Systems Service (IERS)
<http://www.iers.org/>`_ is responsible for measuring the Earth's orientation as
a function of time, and making predictions for the Earth's orientation in the
near future (accounting for scheduled leap seconds). The data products released
by the IERS used by astroplan are the IERS Bulletins A and B.

* IERS Bulletin B is a table with Earth orientation observations from the last
  few decades up through nearly the present time. `Astropy <https://astropy.org>`__ relies on IERS B to
  compute UT1-UTC, and by default will raise an error if you try to compute
  UT1-UTC for a time that is outside the bounds of the IERS Bulletin B table
  (see the `Astropy <https://astropy.org>`__ docs on the `UT1/UTC transformation offsets
  <http://astropy.readthedocs.io/en/latest/time/index.html?highlight=iers#transformation-offsets>`_
  for more details), like this::

    >>> from astropy.time import Time
    >>> Time('2040-01-01', scale='utc').ut1    # Convert from UTC to UT1
    IndexError: (some) times are outside of range covered by IERS table.

* IERS Bulletin A encompasses the observations of recent Earth orientation
  contained in Bulletin B, while also making extrapolations into the past,
  before the Bulletin B tables begin, and into the future, after the Bulletin
  B tables end. The future predictions include leap second additions scheduled
  to be added.

Let's plot the UT1-UTC from IERS Bulletins A and B to show the difference using
astropy's IERS machinery:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.units as u
    from astropy.time import Time
    from urllib.error import HTTPError

    # Download and cache the IERS Bulletins A and B  using astropy's machinery
    # (reminder: astroplan has its own function for this: `download_IERS_A`)
    from astropy.utils.iers import (IERS_A, IERS_A_URL, IERS_B, IERS_B_URL,
                                    FROM_IERS_B, IERS_Auto)
    from astropy.utils.data import download_file
    # Workaround until astropy/astropy#5194 is solved
    iers_a = IERS_Auto.open()
    iers_b = IERS_B.open(download_file(IERS_B_URL, cache=True))

    # Get a range of times to plot from 1990-2022
    time_range = Time("1990-01-01") + np.arange(0, 30, 0.2)*u.year

    # Calculate the difference between UTC and UT1 at those times,
    # allowing times "outside of the table"
    DUT1_a, success_a = time_range.get_delta_ut1_utc(return_status=True,
                                                     iers_table=iers_a)
    DUT1_b, success_b = time_range.get_delta_ut1_utc(return_status=True,
                                                     iers_table=iers_b)

    # Compare input times to the times available in the table. For details, see
    # https://docs.astropy.org/en/stable/utils/iers.html
    measurements_from_b = (success_b == FROM_IERS_B)

    # Make a plot of the time difference
    fig, ax = plt.subplots(figsize=(10,8))
    ax.axhline(0, color='gray', ls='--', lw=2)

    ax.plot(time_range.datetime64,
                 DUT1_a, '-', lw=2, label='IERS Bulletin A + extrapolation')
    ax.plot(time_range.datetime64[measurements_from_b],
                 DUT1_b[measurements_from_b], 'r--', lw=2, label='IERS Bulletin B')
    ax.set(xlabel='Year', ylabel='UT1-UTC [seconds]')
    ax.legend(loc='upper right')
    plt.show()


.. _iers-How_do_I_get_IERS_Bulletin_A_for_astroplan?:

How do I get IERS Bulletin A for astroplan?
===========================================

Without downloading IERS Bulletin A, astroplan simply approximates UT1-UTC=0
always. This will lead to lower precision position and time calculations
on the order of arcseconds or seconds, and allow you to handle times in the
far future and distant past.

To download the IERS Bulletin A table for the first time, or to refresh the
cached version that you already have, simply run::

    from astroplan import download_IERS_A
    download_IERS_A()

