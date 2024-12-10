# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Standard library
import datetime

# Third-party
import astropy.units as u
import numpy as np
import pytest
import pytz
from astropy.coordinates import (EarthLocation, Latitude, Longitude, SkyCoord,
                                 AltAz, Angle)
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from numpy.testing import assert_allclose

# Package
from astroplan.observer import Observer
from astroplan.target import FixedTarget
from astroplan.exceptions import TargetAlwaysUpWarning, TargetNeverUpWarning


def test_Observer_constructor_location():
    """
    Show that location defined by latitude/longitude/elevation is parsed
    identically to passing in an `~astropy.coordinates.EarthLocation` directly.
    """

    lat = '+19:00:00'
    lon = '-155:00:00'
    elevation = 0.0 * u.m
    location = EarthLocation.from_geodetic(lon, lat, elevation)

    environment_kwargs = dict(pressure=1*u.bar, relative_humidity=0.1,
                              temperature=10*u.deg_C)

    obs1 = Observer(name='Observatory',
                    latitude=lat,
                    longitude=lon,
                    elevation=elevation,
                    **environment_kwargs)

    obs2 = Observer(name='Observatory',
                    location=location,
                    **environment_kwargs)

    assert obs1.location == obs2.location, ('using latitude/longitude/'
                                            'elevation keywords gave a '
                                            'different answer from passing in '
                                            'an EarthLocation directly')


def test_Observer_altaz():
    """
    Check that the altitude/azimuth computed by `Observer.altaz` is similar
    to the result from PyEphem when pressure = 0 (no atmosphere) for Vega at
    2000-01-01 12:00:00 UTC.
    """
    # Define the test case
    latitude = '00:00:00'
    longitude = '00:00:00'
    elevation = 0*u.m
    pressure = 0*u.bar  # no atmosphere
    time = Time('2000-01-01 12:00:00')
    vega_coords = SkyCoord('18h36m56.33635s', '+38d47m01.2802s')

    # Calculate altitude/azimuth with astroplan
    location = EarthLocation.from_geodetic(longitude, latitude, elevation)
    astroplan_obs = Observer(name='Observatory', location=location,
                             pressure=pressure)
    astroplan_vega = FixedTarget(vega_coords)
    altaz = astroplan_obs.altaz(time, astroplan_vega)
    astroplan_altitude = altaz.alt
    astroplan_azimuth = altaz.az

    # Calculate altitude/azimuth with PyEphem, like so with print_pyephem_altaz
    pyephem_altitude = Latitude('51.198848716510874 deg')
    pyephem_azimuth = Longitude('358.4676707379987 deg')

    # Assert that altitudes/azimuths are within 30 arcsec - this is a wide
    # tolerance because the IERS tables used by astroplan may offset astroplan's
    # positions due to leap seconds.
    tolerance = (30*u.arcsec).to('deg').value
    assert_allclose(pyephem_altitude.value, astroplan_altitude.value,
                    atol=tolerance)
    assert_allclose(pyephem_azimuth.value, astroplan_azimuth.value,
                    atol=tolerance)

    # Check that alt/az without target returns AltAz frame
    from astropy.coordinates import AltAz
    assert isinstance(astroplan_obs.altaz(time), AltAz)


def test_altaz_multiple_targets():
    vega = SkyCoord(279.23473479*u.deg, 38.78368896*u.deg)
    capella = SkyCoord(79.17232794*u.deg, 45.99799147*u.deg)
    sirius = SkyCoord(101.28715533*u.deg, -16.71611586*u.deg)
    targets = [vega, capella, sirius]

    location = EarthLocation(10*u.deg, 45*u.deg, 0*u.m)

    times = Time('1995-06-21 00:00:00') + np.linspace(0, 1, 5)*u.day

    obs = Observer(location=location)
    transformed_coords = obs.altaz(times, targets, grid_times_targets=True)
    altitudes = transformed_coords.alt

    # Double check by doing one star the normal way with astropy
    vega_altaz = vega.transform_to(AltAz(location=location, obstime=times))
    vega_alt = vega_altaz.alt
    sirius_altaz = sirius.transform_to(AltAz(location=location, obstime=times))
    sirius_alt = sirius_altaz.alt
    assert all(vega_alt == altitudes[0, :])
    assert all(sirius_alt == altitudes[2, :])

    # check that a single element target list works:
    single_target_list = [vega]
    vega_list_alt = obs.altaz(times, single_target_list).alt
    assert np.all(vega_list_alt == vega_alt)

    # check that output elements are the proper lengths and types
    assert isinstance(vega_list_alt, Latitude)
    assert len(vega_list_alt) == len(times)

    # Check for single time
    single_time = times[0]
    vega_single_time = obs.altaz(single_time, single_target_list).alt
    assert vega_single_time[0] == vega_alt[0]

    # Check single target input without list
    vega_no_list = obs.altaz(times, vega).alt
    assert all(vega_no_list == vega_alt)

    # Check FixedTarget for single target
    vega_FixedTarget = FixedTarget(coord=vega, name='Vega')
    vega_FixedTarget_alt = obs.altaz(times, vega_FixedTarget).alt
    assert all(vega_FixedTarget_alt == vega_alt)

    # Check for vector FixedTarget
    vega_FixedTarget = FixedTarget(coord=vega, name='Vega')
    capella_FixedTarget = FixedTarget(coord=capella, name='Capella')
    sirius_FixedTarget = FixedTarget(coord=sirius, name='Sirius')
    ft_list = [vega_FixedTarget, capella_FixedTarget, sirius_FixedTarget]
    ft_vector_alt = obs.altaz(times[:, np.newaxis], ft_list).T.alt
    assert all(ft_vector_alt[0, :] == vega_alt)
    assert all(ft_vector_alt[2, :] == sirius_alt)


def test_rise_set_transit_nearest_vector():
    vega = SkyCoord(279.23473479*u.deg, 38.78368896*u.deg)
    mira = SkyCoord(34.83663376*u.deg, -2.97763767*u.deg)
    sirius = SkyCoord(101.28715533*u.deg, -16.71611586*u.deg)
    polaris = SkyCoord(37.95456067*u.degree, 89.26410897*u.degree)
    sc_list = [vega, mira, sirius, polaris]

    location = EarthLocation(10*u.deg, 45*u.deg, 0*u.m)
    time = Time('2022-06-21 00:00:00')

    obs = Observer(location=location)

    with pytest.warns(TargetAlwaysUpWarning):
        rise_vector = obs.target_rise_time(time, sc_list)
        polaris_rise = obs.target_rise_time(time, polaris)

    vega_rise = obs.target_rise_time(time, vega)
    mira_rise = obs.target_rise_time(time, mira)
    sirius_rise = obs.target_rise_time(time, sirius)

    assert rise_vector[0] == vega_rise
    assert rise_vector[1] == mira_rise
    assert rise_vector[2] == sirius_rise
    assert rise_vector[3].value.mask and polaris_rise.value.mask

    with pytest.warns(TargetAlwaysUpWarning):
        set_vector = obs.target_set_time(time, sc_list)
        polaris_set = obs.target_set_time(time, polaris)

    vega_set = obs.target_set_time(time, vega)
    mira_set = obs.target_set_time(time, mira)
    sirius_set = obs.target_set_time(time, sirius)

    assert set_vector[0] == vega_set
    assert set_vector[1] == mira_set
    assert set_vector[2] == sirius_set
    assert set_vector[3].value.mask and polaris_set.value.mask

    transit_vector = obs.target_meridian_transit_time(time, sc_list)
    vega_trans = obs.target_meridian_transit_time(time, vega)
    mira_trans = obs.target_meridian_transit_time(time, mira)
    sirius_trans = obs.target_meridian_transit_time(time, sirius)
    polaris_trans = obs.target_meridian_transit_time(time, polaris)

    assert transit_vector[0] == vega_trans
    assert transit_vector[1] == mira_trans
    assert transit_vector[2] == sirius_trans
    assert transit_vector[3] == polaris_trans


def print_pyephem_altaz(latitude, longitude, elevation, time, pressure,
                        target_coords):
    """
    Run PyEphem to compute the altitude/azimuth of a target at specified time
    and observatory, for comparison with astroplan calucation tested in
    `test_Observer_altaz`.
    """
    import ephem
    pyephem_obs = ephem.Observer()
    pyephem_obs.lat = latitude
    pyephem_obs.lon = longitude
    pyephem_obs.elevation = elevation
    pyephem_obs.date = time.datetime
    pyephem_obs.pressure = pressure
    pyephem_target = ephem.FixedBody()
    pyephem_target._ra = ephem.degrees(target_coords.ra.radian)
    pyephem_target._dec = ephem.degrees(target_coords.dec.radian)
    pyephem_target.compute(pyephem_obs)
    pyephem_altitude = Latitude(np.degrees(pyephem_target.alt)*u.degree)
    pyephem_azimuth = Longitude(np.degrees(pyephem_target.az)*u.degree)
    print(pyephem_altitude, pyephem_azimuth)


def test_Observer_timezone_parser():
    lat = '+19:00:00'
    lon = '-155:00:00'
    elevation = 0.0 * u.m
    location = EarthLocation.from_geodetic(lon, lat, elevation)

    obs1 = Observer(name='Observatory', location=location,
                    timezone=pytz.timezone('UTC'))
    obs2 = Observer(name='Observatory', location=location, timezone='UTC')
    obs3 = Observer(name='Observatory', location=location)

    assert obs1.timezone == obs2.timezone, ('Accept both strings to pass to '
                                            'the pytz.timezone() constructor '
                                            'and instances of pytz.timezone')

    assert obs2.timezone == obs3.timezone, ('Default timezone should be UTC')


def test_parallactic_angle():
    """
    Compute parallactic angle for targets at hour angle = {3, 19} for
    at observer at IRTF using the online SpeX calculator and PyEphem
    """
    # Set up position for IRTF
    lat = 19.826218*u.deg
    lon = -155.471999*u.deg
    elevation = 4160.0 * u.m
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2015-01-01 00:00:00')
    LST = time.sidereal_time('mean', longitude=lon)
    desired_HA_1 = 3*u.hourangle
    desired_HA_2 = 19*u.hourangle  # = -5*u.hourangle
    target1 = SkyCoord(LST - desired_HA_1, -30*u.degree)
    target2 = SkyCoord(LST - desired_HA_2, -30*u.degree)

    obs = Observer(location=location)
    q1 = obs.parallactic_angle(time, target1)
    q2 = obs.parallactic_angle(time, target2)
    q12 = obs.parallactic_angle(time, [target1, target2])

    # Get values from PyEphem for comparison from print_pyephem_parallactic_angle()
    pyephem_q1 = 46.54610060782033*u.deg
    pyephem_q2 = -65.51818282032019*u.deg

    assert_quantity_allclose(q1, pyephem_q1, atol=1*u.deg)
    assert_quantity_allclose(q2, pyephem_q2, atol=1*u.deg)

    # Get SpeX parallactic angle calculator values for comparison from
    # http://irtfweb.ifa.hawaii.edu/cgi-bin/spex/parangle.cgi to produce

    SpeX_q1 = 46.7237968*u.deg  # deg
    SpeX_q2 = -65.428924*u.deg  # deg

    assert_quantity_allclose(q1, SpeX_q1, atol=0.1*u.deg)
    assert_quantity_allclose(q2, SpeX_q2, atol=0.1*u.deg)

    assert q1 == q12[0]
    assert q2 == q12[1]


def print_pyephem_parallactic_angle():
    # lat = 19.826218*u.deg
    lon = -155.471999*u.deg
    time = Time('2015-01-01 00:00:00')
    LST = time.sidereal_time('mean', longitude=lon)
    desired_HA_1 = 3*u.hourangle
    desired_HA_2 = 19*u.hourangle  # = -5*u.hourangle

    import ephem
    obs = ephem.Observer()
    obs.lat = '19:49:34.3848'
    obs.lon = '-155:28:19.1964'
    obs.elevation = 0
    obs.date = time.datetime
    pyephem_target1 = ephem.FixedBody()
    pyephem_target1._ra = ephem.degrees((LST - desired_HA_1).to(u.rad).value)
    pyephem_target1._dec = ephem.degrees((-30*u.deg).to(u.rad).value)
    pyephem_target1.compute(obs)
    pyephem_q1 = (float(pyephem_target1.parallactic_angle())*u.rad).to(u.deg)

    pyephem_target2 = ephem.FixedBody()
    pyephem_target2._ra = ephem.degrees((LST - desired_HA_2).to(u.rad).value)
    pyephem_target2._dec = ephem.degrees((-30*u.deg).to(u.rad).value)
    pyephem_target2.compute(obs)
    pyephem_q2 = (float(pyephem_target2.parallactic_angle())*u.rad).to(u.deg)
    print(pyephem_q1, pyephem_q2)


def test_sunrise_sunset_equator():
    """
    Check that time of sunrise/set for an observer on the equator is
    consistent with PyEphem results (for no atmosphere/pressure=0)
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    obs = Observer(location=location, pressure=pressure)
    astroplan_next_sunrise = obs.sun_rise_time(time, which='next').datetime
    astroplan_next_sunset = obs.sun_set_time(time, which='next').datetime

    astroplan_prev_sunrise = obs.sun_rise_time(time, which='previous').datetime
    astroplan_prev_sunset = obs.sun_set_time(time, which='previous').datetime

    # Run print_pyephem_sunrise_sunset() to compute analogous
    # result from PyEphem:
    pyephem_next_sunrise = datetime.datetime(2000, 1, 2, 6, 3, 39, 150790)
    pyephem_next_sunset = datetime.datetime(2000, 1, 1, 18, 3, 23, 676686)
    pyephem_prev_sunrise = datetime.datetime(2000, 1, 1, 6, 3, 10, 720052)
    pyephem_prev_sunset = datetime.datetime(1999, 12, 31, 18, 2, 55, 100786)

    # Typical difference in this example between PyEphem and astroplan
    # with an atmosphere is <2 min
    threshold_minutes = 2
    assert (abs(pyephem_next_sunrise - astroplan_next_sunrise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_next_sunset - astroplan_next_sunset) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_sunrise - astroplan_prev_sunrise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_sunset - astroplan_prev_sunset) <
            datetime.timedelta(minutes=threshold_minutes))


def print_pyephem_sunrise_sunset():
    """
    To run:

    python -c 'from astroplan.tests.test_observer import print_pyephem_sunrise_sunset as f; f()'
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0
    time = Time('2000-01-01 12:00:00')

    import ephem
    obs = ephem.Observer()
    obs.lat = lat
    obs.lon = lon
    obs.elevation = elevation
    obs.date = time.datetime
    obs.pressure = pressure
    next_sunrise = obs.next_rising(ephem.Sun(), use_center=True)
    next_sunset = obs.next_setting(ephem.Sun(), use_center=True)
    prev_sunrise = obs.previous_rising(ephem.Sun(), use_center=True)
    prev_sunset = obs.previous_setting(ephem.Sun(), use_center=True)

    print(list(map(repr, [next_sunrise.datetime(), next_sunset.datetime(),
                          prev_sunrise.datetime(), prev_sunset.datetime()])))


def test_vega_rise_set_equator():
    """
    Check that time of rise/set of Vega for an observer on the equator is
    consistent with PyEphem results (for no atmosphere/pressure=0)
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    vega_ra, vega_dec = (279.23473479*u.degree, 38.78368896*u.degree)
    vega = SkyCoord(vega_ra, vega_dec)

    obs = Observer(location=location, pressure=pressure)
    astroplan_next_rise = obs.target_rise_time(time, vega, which='next').datetime
    astroplan_next_set = obs.target_set_time(time, vega, which='next').datetime

    astroplan_prev_rise = obs.target_rise_time(time, vega, which='previous').datetime
    astroplan_prev_set = obs.target_set_time(time, vega, which='previous').datetime

    astroplan_nearest_rise = obs.target_rise_time(time, vega, which='nearest').datetime
    astroplan_nearest_set = obs.target_set_time(time, vega, which='nearest').datetime

    # Run print_pyephem_vega_rise_set() to compute analogous
    # result from PyEphem:
    pyephem_next_rise = datetime.datetime(2000, 1, 2, 5, 52, 8, 257401)
    pyephem_next_set = datetime.datetime(2000, 1, 1, 17, 54, 6, 211705)
    pyephem_prev_rise = datetime.datetime(2000, 1, 1, 5, 56, 4, 165852)
    pyephem_prev_set = datetime.datetime(1999, 12, 31, 17, 58, 2, 120088)

    # Typical difference in this example between PyEphem and astroplan
    # with an atmosphere is <2 min
    threshold_minutes = 2
    assert (abs(pyephem_next_rise - astroplan_next_rise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_next_set - astroplan_next_set) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_rise - astroplan_prev_rise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_set - astroplan_prev_set) <
            datetime.timedelta(minutes=threshold_minutes))

    # Check that the 'nearest' option selects the nearest rise/set
    assert astroplan_nearest_rise == astroplan_prev_rise
    assert astroplan_nearest_set == astroplan_next_set


def print_pyephem_vega_rise_set():
    """
    To run:

    python -c 'from astroplan.tests.test_observer import print_pyephem_vega_rise_set as f; f()'
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0
    time = Time('2000-01-01 12:00:00')
    vega_ra, vega_dec = (279.23473479*u.degree, 38.78368896*u.degree)
    vega = SkyCoord(vega_ra, vega_dec)

    import ephem
    obs = ephem.Observer()
    obs.lat = lat
    obs.lon = lon
    obs.elevation = elevation
    obs.date = time.datetime
    obs.pressure = pressure
    target = ephem.FixedBody()
    target._ra = ephem.degrees(vega.ra.radian)
    target._dec = ephem.degrees(vega.dec.radian)
    target.compute(obs)
    next_rising = obs.next_rising(target).datetime()
    next_setting = obs.next_setting(target).datetime()
    prev_rising = obs.previous_rising(target).datetime()
    prev_setting = obs.previous_setting(target).datetime()
    print(list(map(repr, [next_rising, next_setting,
                          prev_rising, prev_setting])))


def test_vega_sirius_rise_set_seattle():
    """
    Check that time of rise/set of Vega for an observer in Seattle is
    consistent with PyEphem results (for no atmosphere/pressure=0)
    """
    lat = '47d36m34.92s'
    lon = '122d19m59.16s'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('1990-01-01 12:00:00')
    vega = SkyCoord(279.23473479*u.degree, 38.78368896*u.degree)
    sirius = SkyCoord(101.28715533*u.degree, -16.71611586*u.degree)

    obs = Observer(location=location, pressure=pressure)
    astroplan_vega_rise = obs.target_rise_time(time, vega,
                                               which='next').datetime
    astroplan_sirius_rise = obs.target_rise_time(time, sirius,
                                                 which='next').datetime

    astroplan_vector_rise = obs.target_rise_time(time, [vega, sirius],
                                                 which='next').datetime

    astroplan_vega_set = obs.target_set_time(time, vega, which='next').datetime
    astroplan_sirius_set = obs.target_set_time(time, sirius,
                                               which='next').datetime

    astroplan_vector_set = obs.target_set_time(time, [vega, sirius],
                                               which='next').datetime

    # Run print_pyephem_vega_sirius_rise_set() to compute analogous
    # result from PyEphem:
    pyephem_vega_rise = datetime.datetime(1990, 1, 1, 17, 36, 15, 615484)
    pyephem_sirius_rise = datetime.datetime(1990, 1, 2, 11, 4, 52, 35375)
    pyephem_vega_set = datetime.datetime(1990, 1, 1, 13, 49, 58, 788327)
    pyephem_sirius_set = datetime.datetime(1990, 1, 1, 20, 33, 42, 342885)

    # Typical difference in this example between PyEphem and astroplan
    # with an atmosphere is <2 min
    threshold_minutes = 2
    assert (abs(pyephem_vega_rise - astroplan_vega_rise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_sirius_rise - astroplan_sirius_rise) <
            datetime.timedelta(minutes=threshold_minutes))

    assert (abs(pyephem_vega_set - astroplan_vega_set) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_sirius_set - astroplan_sirius_set) <
            datetime.timedelta(minutes=threshold_minutes))

    # Now check vectorized solutions against scalar:
    assert (astroplan_vector_rise[0] == astroplan_vega_rise)
    assert (astroplan_vector_rise[1] == astroplan_sirius_rise)
    assert (astroplan_vector_set[0] == astroplan_vega_set)
    assert (astroplan_vector_set[1] == astroplan_sirius_set)


def print_pyephem_vega_sirius_rise_set():
    """
    To run:

    python -c 'from astroplan.tests.test_observer import \
               print_pyephem_vega_sirius_rise_set as f; f()'
    """

    lat = '47:36:34.92'
    lon = '122:19:59.16'
    elevation = 0.0 * u.m
    pressure = 0
    time = Time('1990-01-01 12:00:00')
    vega_coords = SkyCoord(279.23473479*u.degree, 38.78368896*u.degree)
    sirius_coords = SkyCoord(101.28715533*u.degree, -16.71611586*u.degree)

    import ephem
    obs = ephem.Observer()
    obs.lat = lat
    obs.lon = lon
    obs.elevation = elevation
    obs.date = time.datetime
    obs.pressure = pressure
    vega = ephem.FixedBody()
    vega._ra = ephem.degrees(vega_coords.ra.radian)
    vega._dec = ephem.degrees(vega_coords.dec.radian)
    vega.compute(obs)

    sirius = ephem.FixedBody()
    sirius._ra = ephem.degrees(sirius_coords.ra.radian)
    sirius._dec = ephem.degrees(sirius_coords.dec.radian)
    sirius.compute(obs)

    vega_next_rising = obs.next_rising(vega).datetime()
    vega_next_setting = obs.next_setting(vega).datetime()
    sirius_next_rising = obs.next_rising(sirius).datetime()
    sirius_next_setting = obs.next_setting(sirius).datetime()

    print(list(map(repr, [vega_next_rising, sirius_next_rising,
                          vega_next_setting, sirius_next_setting])))


def test_sunrise_sunset_equator_civil_twilight():
    """
    Check that time of sunrise/set for an observer on the equator is
    consistent with PyEphem results (for no atmosphere/pressure=0)
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    obs = Observer(location=location, pressure=pressure)
    # Manually impose horizon equivalent to civil twilight
    horizon = -6 * u.degree
    astroplan_next_sunrise = obs.sun_rise_time(time, which='next',
                                               horizon=horizon).datetime
    astroplan_next_sunset = obs.sun_set_time(time, which='next',
                                             horizon=horizon).datetime

    astroplan_prev_sunrise = obs.sun_rise_time(time, which='previous',
                                               horizon=horizon).datetime
    astroplan_prev_sunset = obs.sun_set_time(time, which='previous',
                                             horizon=horizon).datetime

    # Run print_pyephem_sunrise_sunset_equator_civil_twilight() to compute
    # analogous result from PyEphem:
    pyephem_next_rise = datetime.datetime(2000, 1, 2, 5, 37, 34, 83328)
    pyephem_next_set = datetime.datetime(2000, 1, 1, 18, 29, 29, 195908)
    pyephem_prev_rise = datetime.datetime(2000, 1, 1, 5, 37, 4, 701708)
    pyephem_prev_set = datetime.datetime(1999, 12, 31, 18, 29, 1, 530987)

    threshold_minutes = 2
    assert (abs(pyephem_next_rise - astroplan_next_sunrise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_next_set - astroplan_next_sunset) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_rise - astroplan_prev_sunrise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_set - astroplan_prev_sunset) <
            datetime.timedelta(minutes=threshold_minutes))


def print_pyephem_sunrise_sunset_equator_civil_twilight():
    """
    Calculate next sunrise and sunset with PyEphem for an observer
    on the equator.

    To run:
    python -c 'from astroplan.tests.test_observer import \
               print_pyephem_sunrise_sunset_equator_civil_twilight as f; f()'
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0
    time = Time('2000-01-01 12:00:00')

    import ephem
    obs = ephem.Observer()
    obs.lat = lat
    obs.lon = lon
    obs.elevation = elevation
    obs.date = time.datetime
    obs.pressure = pressure
    obs.horizon = '-06:00:00'
    next_sunrise = obs.next_rising(ephem.Sun(), use_center=True)
    next_sunset = obs.next_setting(ephem.Sun(), use_center=True)
    prev_sunrise = obs.previous_rising(ephem.Sun(), use_center=True)
    prev_sunset = obs.previous_setting(ephem.Sun(), use_center=True)

    def pyephem_time_to_datetime_str(t): return repr(t.datetime())
    print(list(map(pyephem_time_to_datetime_str,
                   [next_sunrise, next_sunset, prev_sunrise, prev_sunset])))


def test_twilight_convenience_funcs():
    """
    Check that the convenience functions for evening
    astronomical/nautical/civil twilight correspond to their
    PyEphem equivalents
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    obs = Observer(location=location, pressure=pressure)
    # Compute morning twilights with astroplan
    astroplan_morning_civil = obs.twilight_morning_civil(
        time, which='previous').datetime
    astroplan_morning_nautical = obs.twilight_morning_nautical(
        time, which='previous').datetime
    astroplan_morning_astro = obs.twilight_morning_astronomical(
        time, which='previous').datetime
    # Compute evening twilights with astroplan
    astroplan_evening_civil = obs.twilight_evening_civil(
        time, which='next').datetime
    astroplan_evening_nautical = obs.twilight_evening_nautical(
        time, which='next').datetime
    astroplan_evening_astro = obs.twilight_evening_astronomical(
        time, which='next').datetime

    # Compute morning and evening twilights with PyEphem from
    # the function print_pyephem_twilight_convenience_funcs()
    pyephem_morning_civil, pyephem_morning_nautical, pyephem_morning_astronomical, = (
        datetime.datetime(2000, 1, 1, 5, 37, 4, 701708),
        datetime.datetime(2000, 1, 1, 5, 10, 55, 450939),
        datetime.datetime(2000, 1, 1, 4, 44, 39, 415865))

    pyephem_evening_civil, pyephem_evening_nautical, pyephem_evening_astronomical = (
        datetime.datetime(2000, 1, 1, 18, 29, 29, 195908),
        datetime.datetime(2000, 1, 1, 18, 55, 37, 864882),
        datetime.datetime(2000, 1, 1, 19, 21, 53, 213768))

    threshold_minutes = 2
    # Compare morning twilights
    assert (abs(astroplan_morning_civil - pyephem_morning_civil) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_morning_nautical - pyephem_morning_nautical) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_morning_astro - pyephem_morning_astronomical) <
            datetime.timedelta(minutes=threshold_minutes))

    # Compare evening twilights
    assert (abs(astroplan_evening_civil - pyephem_evening_civil) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_evening_nautical - pyephem_evening_nautical) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_evening_astro - pyephem_evening_astronomical) <
            datetime.timedelta(minutes=threshold_minutes))


def print_pyephem_twilight_convenience_funcs():
    """
    To run:
    python -c 'from astroplan.tests.test_observer import \
               print_pyephem_twilight_convenience_funcs as f; f()'
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0
    time = Time('2000-01-01 12:00:00')

    import ephem
    obs = ephem.Observer()
    obs.lat = lat
    obs.lon = lon
    obs.elevation = elevation
    obs.date = time.datetime
    obs.pressure = pressure

    # Morning twilights
    obs.horizon = '-06:00:00'
    morning_civil = obs.previous_rising(ephem.Sun(), use_center=True)
    obs.horizon = '-12:00:00'
    morning_nautical = obs.previous_rising(ephem.Sun(), use_center=True)
    obs.horizon = '-18:00:00'
    morning_astronomical = obs.previous_rising(ephem.Sun(), use_center=True)

    # Evening twilights
    obs.horizon = '-06:00:00'
    evening_civil = obs.next_setting(ephem.Sun(), use_center=True)
    obs.horizon = '-12:00:00'
    evening_nautical = obs.next_setting(ephem.Sun(), use_center=True)
    obs.horizon = '-18:00:00'
    evening_astronomical = obs.next_setting(ephem.Sun(), use_center=True)

    def pyephem_time_to_datetime_str(t): return repr(t.datetime())
    print(list(map(pyephem_time_to_datetime_str,
                   [morning_civil, morning_nautical, morning_astronomical,
                    evening_civil, evening_nautical, evening_astronomical])))


def test_solar_transit():
    """
    Test that astroplan's solar transit/antitransit (which are noon and
    midnight) agree with PyEphem's
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    from astropy.coordinates import get_sun
    obs = Observer(location=location, pressure=pressure)

    # Compute next/previous noon/midnight using generic calc_transit methods
    astroplan_next_transit = obs.target_meridian_transit_time(
        time, get_sun(time), which='next').datetime
    astroplan_next_antitransit = obs.target_meridian_antitransit_time(
        time, get_sun(time), which='next').datetime
    astroplan_prev_transit = obs.target_meridian_transit_time(
        time, get_sun(time), which='previous').datetime
    astroplan_prev_antitransit = obs.target_meridian_antitransit_time(
        time, get_sun(time), which='previous').datetime

    astroplan_nearest_transit = obs.target_meridian_transit_time(
        time, get_sun(time), which='nearest').datetime
    astroplan_nearest_antitransit = obs.target_meridian_antitransit_time(
        time, get_sun(time), which='nearest').datetime

    # Computed in print_pyephem_solar_transit_noon()
    pyephem_next_transit = datetime.datetime(2000, 1, 1, 12, 3, 17, 207300)
    pyephem_next_antitransit = datetime.datetime(2000, 1, 2, 0, 3, 31, 423333)
    pyephem_prev_transit = datetime.datetime(1999, 12, 31, 12, 2, 48, 562755)
    pyephem_prev_antitransit = datetime.datetime(2000, 1, 1, 0, 3, 2, 918943)

    threshold_minutes = 5
    assert (abs(astroplan_next_transit - pyephem_next_transit) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_next_antitransit - pyephem_next_antitransit) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_prev_transit - pyephem_prev_transit) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_prev_antitransit - pyephem_prev_antitransit) <
            datetime.timedelta(minutes=threshold_minutes))

    # Check nearest
    assert astroplan_next_transit == astroplan_nearest_transit
    assert astroplan_nearest_antitransit == astroplan_prev_antitransit


def test_solar_transit_convenience_methods():
    """
    Test that astroplan's noon and midnight convenience methods agree with
    PyEphem's solar transit/antitransit time.
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    obs = Observer(location=location, pressure=pressure)

    # Compute next/previous noon/midnight using generic calc_transit methods
    astroplan_next_noon = obs.noon(time, which='next').datetime
    astroplan_next_midnight = obs.midnight(time, which='next').datetime
    astroplan_prev_noon = obs.noon(time, which='previous').datetime
    astroplan_prev_midnight = obs.midnight(time, which='previous').datetime

    # Computed in print_pyephem_solar_transit_noon()
    pyephem_next_transit = datetime.datetime(2000, 1, 1, 12, 3, 17, 207300)
    pyephem_next_antitransit = datetime.datetime(2000, 1, 2, 0, 3, 31, 423333)
    pyephem_prev_transit = datetime.datetime(1999, 12, 31, 12, 2, 48, 562755)
    pyephem_prev_antitransit = datetime.datetime(2000, 1, 1, 0, 3, 2, 918943)

    threshold_minutes = 5
    assert (abs(astroplan_next_noon - pyephem_next_transit) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_next_midnight - pyephem_next_antitransit) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_prev_noon - pyephem_prev_transit) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(astroplan_prev_midnight - pyephem_prev_antitransit) <
            datetime.timedelta(minutes=threshold_minutes))


def print_pyephem_solar_transit_noon():
    """
    Calculate next sunrise and sunset with PyEphem for an observer
    on the equator.

    To run:
    python -c 'from astroplan.tests.test_observer import print_pyephem_transit_noon as f; f()'
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0
    time = Time('2000-01-01 12:00:00')

    import ephem
    obs = ephem.Observer()
    obs.lat = lat
    obs.lon = lon
    obs.elevation = elevation
    obs.date = time.datetime
    obs.pressure = pressure
    next_transit = obs.next_transit(ephem.Sun())
    next_antitransit = obs.next_antitransit(ephem.Sun())
    prev_transit = obs.previous_transit(ephem.Sun())
    prev_antitransit = obs.previous_antitransit(ephem.Sun())

    def pyephem_time_to_datetime_str(t): return repr(t.datetime())
    print(list(map(pyephem_time_to_datetime_str,
                   [next_transit, next_antitransit,
                    prev_transit, prev_antitransit])))


def test_vega_sirius_transit_seattle():
    """
    Check that time of transit of Vega for an observer in Seattle is
    consistent with PyEphem results (for no atmosphere/pressure=0)
    """
    lat = '47d36m34.92s'
    lon = '122d19m59.16s'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('1990-01-01 12:00:00')
    vega = SkyCoord(279.23473479*u.degree, 38.78368896*u.degree)
    sirius = SkyCoord(101.28715533*u.degree, -16.71611586*u.degree)

    obs = Observer(location=location, pressure=pressure)
    astroplan_vega_transit = obs.target_meridian_transit_time(
        time, vega, which='next').datetime
    astroplan_sirius_transit = obs.target_meridian_transit_time(
        time, sirius, which='next').datetime

    astroplan_vector_transit = obs.target_meridian_transit_time(
        time, [vega, sirius], which='next').datetime

    # Run print_pyephem_vega_sirius_transit() to compute analogous
    # result from PyEphem:
    pyephem_vega_transit = datetime.datetime(1990, 1, 2, 3, 41, 9, 244067)
    pyephem_sirius_transit = datetime.datetime(1990, 1, 1, 15, 51, 15, 135167)

    # Typical difference in this example between PyEphem and astroplan
    # with an atmosphere is <2 min
    threshold_minutes = 2
    assert (abs(pyephem_vega_transit - astroplan_vega_transit) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_sirius_transit - astroplan_sirius_transit) <
            datetime.timedelta(minutes=threshold_minutes))

    # Now check vectorized solutions against scalar:
    assert (astroplan_vector_transit[0] == astroplan_vega_transit)
    assert (astroplan_vector_transit[1] == astroplan_sirius_transit)


def print_pyephem_vega_sirius_transit():
    """
    To run:

    python -c 'from astroplan.tests.test_observer import \
               print_pyephem_vega_sirius_transit as f; f()'
    """
    lat = '47:36:34.92'
    lon = '122:19:59.16'
    elevation = 0.0 * u.m
    pressure = 0
    time = Time('1990-01-01 12:00:00')
    vega_coords = SkyCoord(279.23473479*u.degree, 38.78368896*u.degree)
    sirius_coords = SkyCoord(101.28715533*u.degree, -16.71611586*u.degree)

    import ephem
    obs = ephem.Observer()
    obs.lat = lat
    obs.lon = lon
    obs.elevation = elevation
    obs.date = time.datetime
    obs.pressure = pressure
    vega = ephem.FixedBody()
    vega._ra = ephem.degrees(vega_coords.ra.radian)
    vega._dec = ephem.degrees(vega_coords.dec.radian)
    vega.compute(obs)

    sirius = ephem.FixedBody()
    sirius._ra = ephem.degrees(sirius_coords.ra.radian)
    sirius._dec = ephem.degrees(sirius_coords.dec.radian)
    sirius.compute(obs)

    vega_next_transit = obs.next_transit(vega).datetime()
    sirius_next_transit = obs.next_transit(sirius).datetime()

    print(list(map(repr, [vega_next_transit, sirius_next_transit])))


def test_target_is_up():
    """
    Test that Polaris is/isn't observable from north/south pole
    """
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    north = EarthLocation.from_geodetic('00:00:00',
                                        '90:00:00', elevation)
    south = EarthLocation.from_geodetic('00:00:00',
                                        '-90:00:00', elevation)
    time = Time('2000-01-01 12:00:00')
    polaris = SkyCoord(37.95456067*u.degree, 89.26410897*u.degree)
    polaris_B = SkyCoord(37.639725*u.degree, 89.26080556*u.degree)
    polaris_binary = [polaris, polaris_B]

    north_pole = Observer(location=north, pressure=pressure)
    south_pole = Observer(location=south, pressure=pressure)
    assert north_pole.target_is_up(time, polaris)
    assert not south_pole.target_is_up(time, polaris)

    assert all(north_pole.target_is_up(time, polaris_binary))
    assert not any(south_pole.target_is_up(time, polaris_binary))


def test_string_times():
    """
    Test that strings passed to time argument get successfully
    passed to Time constructor. Analogous test to test_vega_rise_set_equator(),
    just with a string for a time.
    """
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = '2000-01-01 12:00:00'
    vega_ra, vega_dec = (279.23473479*u.degree, 38.78368896*u.degree)
    vega = SkyCoord(vega_ra, vega_dec)

    obs = Observer(location=location, pressure=pressure)
    astroplan_next_rise = obs.target_rise_time(time, vega, which='next').datetime
    astroplan_next_set = obs.target_set_time(time, vega, which='next').datetime

    astroplan_prev_rise = obs.target_rise_time(time, vega, which='previous').datetime
    astroplan_prev_set = obs.target_set_time(time, vega, which='previous').datetime

    astroplan_nearest_rise = obs.target_rise_time(time, vega, which='nearest').datetime
    astroplan_nearest_set = obs.target_set_time(time, vega, which='nearest').datetime

    # Run print_pyephem_vega_rise_set() to compute analogous
    # result from PyEphem:
    pyephem_next_rise = datetime.datetime(2000, 1, 2, 5, 52, 8, 257401)
    pyephem_next_set = datetime.datetime(2000, 1, 1, 17, 54, 6, 211705)
    pyephem_prev_rise = datetime.datetime(2000, 1, 1, 5, 56, 4, 165852)
    pyephem_prev_set = datetime.datetime(1999, 12, 31, 17, 58, 2, 120088)

    # Typical difference in this example between PyEphem and astroplan
    # with an atmosphere is <2 min
    threshold_minutes = 2
    assert (abs(pyephem_next_rise - astroplan_next_rise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_next_set - astroplan_next_set) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_rise - astroplan_prev_rise) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_set - astroplan_prev_set) <
            datetime.timedelta(minutes=threshold_minutes))

    # Check that the 'nearest' option selects the nearest rise/set
    assert astroplan_nearest_rise == astroplan_prev_rise
    assert astroplan_nearest_set == astroplan_next_set


def test_TargetAlwaysUpWarning(recwarn):
    lat = '90:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    polaris = SkyCoord(37.95456067*u.degree, 89.26410897*u.degree)

    obs = Observer(location=location)
    no_time = obs.target_rise_time(time, polaris, which='next')

    w = recwarn.pop(TargetAlwaysUpWarning)
    assert issubclass(w.category, TargetAlwaysUpWarning)
    assert no_time.mask

    # Regression test: make sure 'nearest' also works
    no_time = obs.target_rise_time(time, polaris, which='nearest')

    # Cycle back through warnings until a TargetAlwaysUpWarning is hit
    # (other warnings can also be raised here)
    while not issubclass(w.category, TargetAlwaysUpWarning):
        w = recwarn.pop(TargetAlwaysUpWarning)

    assert issubclass(w.category, TargetAlwaysUpWarning)
    assert no_time.mask


def test_TargetNeverUpWarning(recwarn):
    lat = '-90:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    polaris = SkyCoord(37.95456067*u.degree, 89.26410897*u.degree)

    obs = Observer(location=location)
    no_time = obs.target_rise_time(time, polaris, which='next')

    w = recwarn.pop(TargetNeverUpWarning)
    assert issubclass(w.category, TargetNeverUpWarning)
    assert no_time.mask


def test_mixed_rise_and_dont_rise():
    vega = SkyCoord(279.23473479*u.deg, 38.78368896*u.deg)
    polaris = SkyCoord(37.95456067*u.deg, 89.26410897*u.deg)
    sirius = SkyCoord(101.28715533*u.deg, -16.71611586*u.deg)
    targets = [vega, polaris, sirius]

    location = EarthLocation(10*u.deg, 45*u.deg, 0*u.m)
    time = Time('1995-06-21 00:00:00')

    obs = Observer(location=location)
    with pytest.warns(TargetAlwaysUpWarning) as recwarn:
        rise_times = obs.target_rise_time(time, targets, which='next')

    assert rise_times.mask[1]

    targets_that_rise = np.array(targets)[~rise_times.mask]
    assert np.all([vega, sirius] == targets_that_rise)

    w = recwarn.pop(TargetAlwaysUpWarning)
    assert issubclass(w.category, TargetAlwaysUpWarning)


def test_timezone_convenience_methods():
    location = EarthLocation(-74.0*u.deg, 40.7*u.deg, 0*u.m)
    obs = Observer(location=location, timezone=pytz.timezone('US/Eastern'))
    t = Time(57100.3, format='mjd')
    assert (obs.astropy_time_to_datetime(t).hour == 3)

    dt = datetime.datetime(2015, 3, 19, 3, 12)
    assert (obs.datetime_to_astropy_time(dt).datetime ==
            datetime.datetime(2015, 3, 19, 7, 12))

    assert (obs.astropy_time_to_datetime(obs.datetime_to_astropy_time(dt)).replace(
            tzinfo=None) == dt)

    # Test ndarray of times:
    times = t + np.linspace(0, 24, 10)*u.hour
    times_dt_ndarray = times.datetime
    assert all((obs.datetime_to_astropy_time(times_dt_ndarray)).jd ==
               (times + 4*u.hour).jd)

    # Test list of times:
    times_dt_list = list(times.datetime)
    assert all((obs.datetime_to_astropy_time(times_dt_list)).jd ==
               (times + 4*u.hour).jd)

    dts = obs.astropy_time_to_datetime(times)
    naive_dts = list(map(lambda t: t.replace(tzinfo=None), dts))
    assert all(naive_dts == times_dt_ndarray - datetime.timedelta(hours=4))


@pytest.mark.remote_data
def test_is_night():
    lco = Observer(location=EarthLocation.of_site('lco'))  # Las Campanas
    aao = Observer(location=EarthLocation.of_site('aao'))  # Sydney, Australia
    vbo = Observer(location=EarthLocation.of_site('vbo'))  # India

    time1 = Time('2015-07-28 17:00:00')
    nights1 = [observer.is_night(time1) for observer in [lco, aao, vbo]]
    assert np.all(nights1 == [False, True, True])

    time2 = Time('2015-07-28 02:00:00')
    nights2 = [observer.is_night(time2) for observer in [lco, aao, vbo]]
    assert np.all(nights2 == [True, False, False])


def test_moon_altaz():
    time = Time('2012-06-21 03:00:00')
    location = EarthLocation.from_geodetic(-155*u.deg, 19*u.deg, 0*u.m)
    obs = Observer(location=location, pressure=0*u.bar)
    altaz = obs.moon_altaz(time)
    astroplan_altaz = [altaz.alt.radian, altaz.az.radian]
    # Get this from print_pyephem_moon_altaz():
    pyephem_altaz = [0.7092548608779907, 4.865438938140869]
    assert_allclose(astroplan_altaz, pyephem_altaz, atol=0.1)


def print_pyephem_moon_altaz():
    """
    To run:
    python -c 'from astroplan.tests.test_observer import print_pyephem_moon_altaz as f; f()'
    """
    time = Time('2012-06-21 03:00:00')
    location = EarthLocation.from_geodetic(-155*u.deg, 19*u.deg, 0*u.m)

    import ephem
    moon = ephem.Moon()
    pe_obs = ephem.Observer()
    pe_obs.lat = location.lat.to(u.degree).to_string(sep=':')
    pe_obs.lon = location.lon.to(u.degree).to_string(sep=':')
    pe_obs.elevation = location.height.to(u.m).value
    pe_obs.date = time.datetime
    pe_obs.pressure = 0
    moon.compute(pe_obs)
    print(list(map(float, [moon.alt, moon.az])))


def test_exceptions():
    lat = '00:00:00'
    lon = '00:00:00'
    elevation = 0.0 * u.m
    location = EarthLocation.from_geodetic(lon, lat, elevation)
    time = Time('2000-01-01 12:00:00')
    vega_coords = SkyCoord('18h36m56.33635s', '+38d47m01.2802s')

    obs = Observer(location=location)

    with pytest.raises(ValueError):
        obs.target_rise_time(time, vega_coords, which='oops').datetime

    with pytest.raises(ValueError):
        obs.target_set_time(time, vega_coords, which='oops').datetime

    with pytest.raises(ValueError):
        obs.target_meridian_transit_time(time, vega_coords,
                                         which='oops').datetime

    with pytest.raises(ValueError):
        obs.target_meridian_antitransit_time(time, vega_coords,
                                             which='oops').datetime

    with pytest.raises(TypeError):
        FixedTarget(['00:00:00', '00:00:00'], name='VE')

    with pytest.raises(TypeError):
        Observer(location='Greenwich')

    with pytest.raises(TypeError):
        Observer(location=EarthLocation(0, 0, 0), timezone=-6)


def vectorize_timing(n_targets):
    """
    Calculate the rise time of ``n_targets`` targets, return the
    run time in seconds.
    """
    from time import time
    vega_coord = SkyCoord(279.23473479*u.degree, 38.78368896*u.degree)
    vega = FixedTarget(name="Vega", coord=vega_coord)
    target_list = n_targets*[vega]
    t = Time("2008-02-27 22:00:00")
    obs = Observer(location=EarthLocation(10*u.deg, 20*u.deg, 0*u.m))
    start = time()
    obs.target_rise_time(t, target_list)
    end = time()
    return end-start


def test_local_sidereal_time():
    time = Time('2005-02-03 00:00:00')
    location = EarthLocation.from_geodetic(10*u.deg, 40*u.deg, 0*u.m)
    obs = Observer(location=location)
    # test sidereal time
    astroplan_lst = obs.local_sidereal_time(time)
    # Compute this with print_pyephem_lst()
    pyephem_lst = 2.5005375428099104*u.rad
    assert_quantity_allclose(astroplan_lst, pyephem_lst, atol=0.01*u.deg)


def print_pyephem_lst():
    time = Time('2005-02-03 00:00:00')
    import ephem
    pe_apo = ephem.Observer()
    pe_apo.lat = '40:00:00'
    pe_apo.lon = '10:00:00'
    pe_apo.elev = 0
    pe_apo.date = time.datetime
    pe_lst = Angle(pe_apo.sidereal_time(), unit=u.rad)
    return pe_lst


def test_hour_angle():
    # TODO: Add tests for different targets/times with tools other than PyEphem
    time = Time('2005-02-03 00:00:00')
    location = EarthLocation.from_geodetic(10*u.deg, 40*u.deg, 0*u.m)
    obs = Observer(location=location)
    vernal_eq = FixedTarget(SkyCoord(ra=0*u.deg, dec=0*u.deg))
    hour_angle = obs.target_hour_angle(time, vernal_eq)
    lst = obs.local_sidereal_time(time)
    assert_quantity_allclose(hour_angle, lst, atol=0.001*u.deg)


@pytest.mark.remote_data
def test_tonight():
    obs = Observer.at_site('Subaru')
    obs.height = 0 * u.m

    horizon = 0 * u.degree

    noon = Time('2016-02-03 22:00:00')
    midnight = Time('2016-02-04 10:00:00')

    sunset = Time('2016-02-04 04:14:00').datetime
    sunrise = Time('2016-02-04 16:54:00').datetime

    threshold_minutes = 6

    during_day = obs.tonight(time=noon, horizon=horizon)
    during_night = obs.tonight(time=midnight, horizon=horizon)

    assert (abs(sunset - during_day[0].datetime) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(sunrise - during_day[1].datetime) <
            datetime.timedelta(minutes=threshold_minutes))

    assert (abs(midnight.datetime - during_night[0].datetime) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(sunrise - during_night[1].datetime) <
            datetime.timedelta(minutes=threshold_minutes))

    astro_sunset = Time('2016-08-08 06:13:00')
    horizon = -18 * u.degree

    post_civil_sunset = Time('2016-08-08 05:00:00')
    during_twilight = obs.tonight(time=post_civil_sunset, horizon=horizon)
    during_twilight_wo_horizon = obs.tonight(time=post_civil_sunset)  # noqa

    # Get correct astro sunset if checking after civil sunset
    assert (abs(astro_sunset.datetime - during_twilight[0].datetime) <
            datetime.timedelta(minutes=threshold_minutes))


def print_pyephem_moon_rise_set():
    """
    To run:

    python -c 'from astroplan.tests.test_observer import print_pyephem_moon_rise_set as f; f()'
    """
    lat = '42:00:00'
    lon = '-70:00:00'
    elevation = 0.0 * u.m
    pressure = 0
    time = Time('2017-10-07 12:00:00')

    import ephem
    obs = ephem.Observer()
    obs.lat = lat
    obs.lon = lon
    obs.elevation = elevation
    obs.date = time.datetime
    obs.pressure = pressure
    next_moonrise = obs.next_rising(ephem.Moon(), use_center=True)
    next_moonset = obs.next_setting(ephem.Moon(), use_center=True)
    prev_moonrise = obs.previous_rising(ephem.Moon(), use_center=True)
    prev_moonset = obs.previous_setting(ephem.Moon(), use_center=True)

    print(list(map(repr, [next_moonrise.datetime(), next_moonset.datetime(),
                          prev_moonrise.datetime(), prev_moonset.datetime()])))


def test_moon_rise_set():
    pyephem_next_rise = datetime.datetime(2017, 10, 7, 23, 50, 24, 407018)
    pyephem_next_set = datetime.datetime(2017, 10, 7, 12, 30, 30, 787116)
    pyephem_prev_rise = datetime.datetime(2017, 10, 6, 23, 13, 43, 644455)
    pyephem_prev_set = datetime.datetime(2017, 10, 6, 11, 20, 9, 340009)

    time = Time('2017-10-07 12:00:00')
    lat = '42:00:00'
    lon = '-70:00:00'
    elevation = 0.0 * u.m
    # pressure = 0 * u.bar
    location = EarthLocation.from_geodetic(lon, lat, elevation)

    obs = Observer(location=location)

    astroplan_next_rise = obs.moon_rise_time(time, which='next')
    astroplan_next_set = obs.moon_set_time(time, which='next')
    astroplan_prev_rise = obs.moon_rise_time(time, which='previous')
    astroplan_prev_set = obs.moon_set_time(time, which='previous')

    threshold_minutes = 2
    assert (abs(pyephem_next_rise - astroplan_next_rise.datetime) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_next_set - astroplan_next_set.datetime) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_rise - astroplan_prev_rise.datetime) <
            datetime.timedelta(minutes=threshold_minutes))
    assert (abs(pyephem_prev_set - astroplan_prev_set.datetime) <
            datetime.timedelta(minutes=threshold_minutes))


mmto_sunsets = [
    Time('2019-01-01 00:31'),
    Time('2019-02-01 00:58'),
    Time('2019-03-01 01:21'),
    Time('2019-04-01 01:43'),
    Time('2019-05-01 02:03'),
    Time('2019-06-01 02:24'),
]


@pytest.mark.parametrize('mmto_sunset', mmto_sunsets)
def test_sun_set_vs_mmto_almanac(mmto_sunset):
    """
    Validates issue: https://github.com/astropy/astroplan/issues/409

    MMTO times to the nearest minute from the MMTO Almanac:
    http://www.mmto.org/sites/default/files/almanac_2019.pdf
    """
    loc = EarthLocation.from_geodetic(-110.8850*u.deg, 31.6883 * u.deg,
                                      2608 * u.m)
    mmt = Observer(location=loc, pressure=0*u.bar)

    # Compute equivalent time with astroplan
    astroplan_sunset = mmt.sun_set_time(mmto_sunset - 10*u.min,
                                        horizon=-0.8333*u.deg, which='next')

    assert abs(mmto_sunset - astroplan_sunset) < 1 * u.min


@pytest.mark.remote_data
def test_observer_lon_lat_el():
    """Test that astropy.EarthLocation conversion to longitude,
    latitude, and elevation works correctly.
    """
    obs = Observer.at_site('Subaru')
    for attr in ['longitude', 'latitude', 'elevation']:
        assert hasattr(obs, attr)


@pytest.mark.remote_data
def test_hash_observer():
    """Test that Observer objects are hashable."""
    obs1 = Observer.at_site('Subaru')
    obs2 = Observer.at_site('Subaru')
    assert hash(obs1) == hash(obs2)

    obs3 = Observer.at_site('Keck', timezone='US/Hawaii')
    assert hash(obs1) != hash(obs3)

    obs4 = Observer.at_site('Keck', timezone='US/Hawaii')
    assert hash(obs3) == hash(obs4)


@pytest.mark.remote_data
def test_eq_observer():
    """Test that Observer objects are comparable."""
    obs1 = Observer.at_site('Subaru')
    obs2 = Observer.at_site('Subaru')
    assert obs1 == obs2

    obs3 = Observer.at_site('Keck')
    assert obs1 != obs3

    obs4 = Observer.at_site('Subaru', timezone='US/Hawaii')
    assert obs1 != obs4

    obs5 = Observer.at_site('Subaru', timezone='US/Hawaii')
    assert obs4 == obs5
