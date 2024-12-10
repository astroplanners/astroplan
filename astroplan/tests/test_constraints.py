import datetime as dt

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Galactic, SkyCoord, get_sun, get_body
from astropy.time import Time

from astroplan.constraints import (
    AltitudeConstraint, AirmassConstraint, AtNightConstraint,
    is_observable, is_always_observable, observability_table,
    time_grid_from_range,
    GalacticLatitudeConstraint,
    SunSeparationConstraint,
    MoonSeparationConstraint, MoonIlluminationConstraint,
    TimeConstraint, LocalTimeConstraint, months_observable,
    max_best_rescale, min_best_rescale, PhaseConstraint,
    PrimaryEclipseConstraint, SecondaryEclipseConstraint,
    is_event_observable)
from astroplan.exceptions import MissingConstraintWarning
from astroplan.observer import Observer
from astroplan.periodic import EclipsingSystem
from astroplan.target import FixedTarget, get_skycoord

vega = FixedTarget(coord=SkyCoord(ra=279.23473479*u.deg, dec=38.78368896*u.deg),
                   name="Vega")
rigel = FixedTarget(coord=SkyCoord(ra=78.63446707*u.deg, dec=8.20163837*u.deg),
                    name="Rigel")
polaris = FixedTarget(coord=SkyCoord(ra=37.95456067*u.deg,
                                     dec=89.26410897*u.deg), name="Polaris")


@pytest.mark.remote_data
def test_at_night_basic():
    subaru = Observer.at_site("Subaru")
    time_ranges = [Time(['2001-02-03 04:05:06', '2001-02-04 04:05:06']),  # 1 day
                   Time(['2007-08-09 10:11:12', '2007-08-09 11:11:12'])]  # 1 hr
    targets = [vega, rigel, polaris]
    for time_range in time_ranges:
        # Calculate constraint using methods on astroplan.Observer:
        observer_is_night = subaru.is_night(time_grid_from_range(time_range))
        observer_is_night_any = any(observer_is_night)
        observer_is_night_all = all(observer_is_night)

        assert all(is_observable(AtNightConstraint(), subaru, targets,
                                 time_range=time_range) ==
                   len(targets)*[observer_is_night_any])

        assert all(is_always_observable(AtNightConstraint(), subaru, targets,
                                        time_range=time_range) ==
                   len(targets)*[observer_is_night_all])


@pytest.mark.remote_data
def test_observability_table():
    subaru = Observer.at_site("Subaru")
    # time_ranges = [Time(['2001-02-03 04:05:06', '2001-02-04 04:05:06']),  # 1 day
    #                Time(['2007-08-09 10:11:12', '2007-08-09 11:11:12'])]  # 1 hr
    targets = [vega, rigel, polaris]

    time_range = Time(['2001-02-03 04:05:06', '2001-02-04 04:05:06'])
    # note that this uses the AirmassConstraint in None min mode - that means
    # targets below the horizon will pass the airmass constraint
    constraints = [AtNightConstraint(), AirmassConstraint(3, None)]

    obstab = observability_table(constraints, subaru, targets,
                                 time_range=time_range)

    assert len(obstab) == 3
    assert np.all(obstab['target name'] == ['Vega', 'Rigel', 'Polaris'])

    assert 'times' in obstab.meta
    assert 'observer' in obstab.meta
    assert 'constraints' in obstab.meta
    assert len(obstab.meta['constraints']) == 2

    np.testing.assert_allclose(obstab['fraction of time observable'],
                               np.array([21, 22, 15])/48)

    # now compare to is_observable and is_always_observable
    is_obs = is_observable(constraints, subaru, targets, time_range=time_range)
    np.testing.assert_allclose(obstab['ever observable'], is_obs)
    all_obs = is_always_observable(constraints, subaru, targets,
                                   time_range=time_range)
    np.testing.assert_allclose(obstab['always observable'], all_obs)

    # try the scalar time_range case
    ttab = observability_table(constraints, subaru, targets,
                               time_range=(time_range[0] - 12*u.hour,
                                           time_range[0] + 12*u.hour))
    stab = observability_table(constraints, subaru, targets,
                               time_range=time_range[0])

    assert all(ttab['fraction of time observable'] == stab['fraction of time observable'])
    assert 'time observable' in stab.colnames


@pytest.mark.remote_data
def test_altitude_constraint():
    subaru = Observer.at_site("Subaru")
    time = Time('2001-02-03 15:35:00')
    time_range = Time([time, time+3*u.hour])

    constraint = AltitudeConstraint(min=40*u.deg, max=50*u.deg, boolean_constraint=False)
    results = constraint(subaru, vega, times=time_grid_from_range(time_range))
    # Check if below min and above max values are 0
    assert np.all([results != 0][0] == [False, False, True,  True,  False, False])


@pytest.mark.remote_data
def test_compare_altitude_constraint_and_observer():
    time = Time('2001-02-03 04:05:06')
    time_ranges = [Time([time, time+1*u.hour]) + offset
                   for offset in np.arange(0, 400, 100)*u.day]
    for time_range in time_ranges:
        subaru = Observer.at_site("Subaru")
        targets = [vega, rigel, polaris]

        min_alt = 40*u.deg
        max_alt = 80*u.deg
        # Check if each target meets altitude constraints using Observer
        always_from_observer = [all([min_alt < subaru.altaz(time, target).alt < max_alt
                                     for time in time_grid_from_range(time_range)])
                                for target in targets]
        # Check if each target meets altitude constraints using
        # is_always_observable and AltitudeConstraint
        always_from_constraint = is_always_observable(AltitudeConstraint(
            min_alt, max_alt), subaru, targets, time_range=time_range)
        assert all(always_from_observer == always_from_constraint)


@pytest.mark.remote_data
def test_compare_airmass_constraint_and_observer():
    time = Time('2001-02-03 04:05:06')
    time_ranges = [Time([time, time+1*u.hour]) + offset
                   for offset in np.arange(0, 400, 100)*u.day]
    for time_range in time_ranges:
        subaru = Observer.at_site("Subaru")
        targets = [vega, rigel, polaris]

        max_airmass = 2
        # Check if each target meets airmass constraint in using Observer
        always_from_observer = [all([(subaru.altaz(time, target).secz <= max_airmass) &
                                     (1 <= subaru.altaz(time, target).secz)
                                     for time in time_grid_from_range(time_range)])
                                for target in targets]
        # Check if each target meets altitude constraints using
        # is_always_observable and AirmassConstraint
        always_from_constraint = is_always_observable(
            AirmassConstraint(max_airmass), subaru, targets, time_range=time_range)
        assert all(always_from_observer == always_from_constraint)


@pytest.mark.remote_data
def test_galactic_plane_separation():
    time = Time('2003-04-05 06:07:08')
    apo = Observer.at_site("APO")
    one_deg_away = SkyCoord(0*u.deg, 1*u.deg, frame=Galactic)
    five_deg_away = SkyCoord(0*u.deg, 5*u.deg, frame=Galactic)
    twenty_deg_away = SkyCoord(0*u.deg, 20*u.deg, frame=Galactic)

    constraint = GalacticLatitudeConstraint(min=2*u.deg, max=10*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)
    assert np.all(is_constraint_met == [False, True, False])

    constraint = GalacticLatitudeConstraint(max=10*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)
    assert np.all(is_constraint_met == [True, True, False])

    constraint = GalacticLatitudeConstraint(min=2*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)
    assert np.all(is_constraint_met == [False, True, True])


@pytest.mark.remote_data
# astropy.coordinates.errors.NonRotationTransformationWarning
@pytest.mark.filterwarnings("ignore")
def test_sun_separation():
    time = Time('2003-04-05 06:07:08')
    apo = Observer.at_site("APO")
    sun = get_sun(time)
    one_deg_away = SkyCoord(ra=sun.ra, dec=sun.dec+1*u.deg)
    five_deg_away = SkyCoord(ra=sun.ra+5*u.deg, dec=sun.dec)
    twenty_deg_away = SkyCoord(ra=sun.ra+20*u.deg, dec=sun.dec)

    constraint = SunSeparationConstraint(min=2*u.deg, max=10*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)
    assert np.all(is_constraint_met == [False, True, False])

    constraint = SunSeparationConstraint(max=10*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)
    assert np.all(is_constraint_met == [True, True, False])

    constraint = SunSeparationConstraint(min=2*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)
    assert np.all(is_constraint_met == [False, True, True])


@pytest.mark.remote_data
# astropy.coordinates.errors.NonRotationTransformationWarning
@pytest.mark.filterwarnings("ignore")
def test_moon_separation():
    time = Time('2003-04-05 06:07:08')
    apo = Observer.at_site("APO")
    altaz_frame = apo.altaz(time)
    moon = get_body("moon", time, apo.location).transform_to(altaz_frame)
    one_deg_away = SkyCoord(az=moon.az, alt=moon.alt+1*u.deg, frame=altaz_frame)
    five_deg_away = SkyCoord(az=moon.az+5*u.deg, alt=moon.alt,
                             frame=altaz_frame)
    twenty_deg_away = SkyCoord(az=moon.az+20*u.deg, alt=moon.alt,
                               frame=altaz_frame)

    constraint = MoonSeparationConstraint(min=2*u.deg, max=10*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)

    assert np.all(is_constraint_met == [False, True, False])

    constraint = MoonSeparationConstraint(max=10*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)
    assert np.all(is_constraint_met == [True, True, False])

    constraint = MoonSeparationConstraint(min=2*u.deg)
    is_constraint_met = constraint(apo, [one_deg_away, five_deg_away,
                                         twenty_deg_away], times=time)
    assert np.all(is_constraint_met == [False, True, True])


@pytest.mark.remote_data
def test_moon_illumination():
    times = Time(["2015-08-28 03:30", "2015-08-28 12:00",
                  "2015-09-05 10:30", "2015-09-15 18:35"])
    lco = Observer.at_site("LCO")
    # At these times, moon illuminations are:
    # [ 0.9600664, 0.97507911, 0.49766145,  0.05427445]
    # and altitudes are:
    # [ 73.53496408, -24.55896688, 42.93207952, 66.46854598] deg

    constraint = MoonIlluminationConstraint(min=0.2, max=0.8)
    is_constraint_met = constraint(lco, None, times=times)
    assert np.all(is_constraint_met == [False, False, True, False])

    constraint = MoonIlluminationConstraint(min=0.2)
    is_constraint_met = constraint(lco, None, times=times)
    assert np.all(is_constraint_met == [True, False, True, False])

    constraint = MoonIlluminationConstraint(max=0.8)
    is_constraint_met = constraint(lco, None, times=times)
    assert np.all(is_constraint_met == [False, True, True, True])

    constraint = MoonIlluminationConstraint(max=0)
    is_constraint_met = constraint(lco, None, times=times)
    assert np.all(is_constraint_met == [False, True, False, False])

    constraint = MoonIlluminationConstraint.dark()
    is_constraint_met = constraint(lco, None, times=times)
    assert np.all(is_constraint_met == [False, True, False, True])

    constraint = MoonIlluminationConstraint.grey()
    is_constraint_met = constraint(lco, None, times=times)
    assert np.all(is_constraint_met == [False, False, True, False])

    constraint = MoonIlluminationConstraint.bright()
    is_constraint_met = constraint(lco, None, times=times)
    assert np.all(is_constraint_met == [True, False, False, False])


@pytest.mark.remote_data
def test_local_time_constraint_utc():
    time = Time('2001-02-03 04:05:06')
    subaru = Observer.at_site("Subaru")
    constraint = LocalTimeConstraint(min=dt.time(23, 50), max=dt.time(4, 8))
    is_constraint_met = constraint(subaru, None, times=time)
    assert is_constraint_met is np.bool_(True)

    constraint = LocalTimeConstraint(min=dt.time(0, 2), max=dt.time(4, 3))
    is_constraint_met = constraint(subaru, None, times=time)
    assert is_constraint_met is np.bool_(False)

    constraint = LocalTimeConstraint(min=dt.time(3, 8), max=dt.time(5, 35))
    is_constraint_met = constraint(subaru, None, times=time)
    assert is_constraint_met is np.bool_(True)


@pytest.mark.remote_data
def test_local_time_constraint_hawaii_tz():
    # Define timezone in Observer.timezone
    time = Time('2001-02-03 04:05:06')
    subaru = Observer.at_site("Subaru", timezone="US/Hawaii")
    constraint = LocalTimeConstraint(min=dt.time(23, 50), max=dt.time(4, 8))
    is_constraint_met = constraint(subaru, None, times=time)
    assert is_constraint_met is np.bool_(True)

    constraint = LocalTimeConstraint(min=dt.time(0, 2), max=dt.time(4, 3))
    is_constraint_met = constraint(subaru, None, times=time)
    assert is_constraint_met is np.bool_(False)

    constraint = LocalTimeConstraint(min=dt.time(3, 8), max=dt.time(5, 35))
    is_constraint_met = constraint(subaru, None, times=time)
    assert is_constraint_met is np.bool_(True)


@pytest.mark.remote_data
def test_docs_example():
    # Test the example in astroplan/docs/tutorials/constraints.rst
    target_table_string = """# name ra_degrees dec_degrees
    Polaris 37.95456067 89.26410897
    Vega 279.234734787 38.783688956
    Albireo 292.68033548 27.959680072
    Algol 47.042218553 40.955646675
    Rigel 78.634467067 -8.201638365
    Regulus 152.092962438 11.967208776"""

    from astroplan import Observer, FixedTarget
    from astropy.time import Time
    subaru = Observer.at_site("Subaru")
    time_range = Time(["2015-08-01 06:00", "2015-08-01 12:00"])

    # Read in the table of targets
    from astropy.io import ascii
    target_table = ascii.read(target_table_string)

    # Create astroplan.FixedTarget objects for each one in the table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    targets = [FixedTarget(coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), name=name)
               for name, ra, dec in target_table]

    from astroplan import Constraint, is_observable

    class VegaSeparationConstraint(Constraint):
        """
        Constraint the separation from Vega
        """

        def __init__(self, min=None, max=None):
            """
            min : `~astropy.units.Quantity` or `None` (optional)
                Minimum acceptable separation between Vega and target. `None`
                indicates no limit.
            max : `~astropy.units.Quantity` or `None` (optional)
                Minimum acceptable separation between Vega and target. `None`
                indicates no limit.
            """
            self.min = min if min is not None else 0*u.deg
            self.max = max if max is not None else 180*u.deg

        def compute_constraint(self, times, observer, targets):
            vega = SkyCoord(ra=279.23473479*u.deg, dec=38.78368896*u.deg)

            # Calculate separation between target and vega
            # Targets are automatically converted to SkyCoord objects
            # by __call__ before compute_constraint is called.
            vega_separation = vega.separation(targets)

            # Return an array that is True where the target is observable and
            # False where it is not
            return (self.min < vega_separation) & (vega_separation < self.max)

    constraints = [VegaSeparationConstraint(min=5*u.deg, max=30*u.deg)]
    observability = is_observable(constraints, subaru, targets,
                                  time_range=time_range)

    assert all(observability == [False, False, True, False, False, False])


@pytest.mark.remote_data
def test_regression_airmass_141():
    subaru = Observer.at_site("Subaru")
    time = Time('2001-1-1 12:00')

    coord = SkyCoord(ra=16*u.hour, dec=20*u.deg)

    assert subaru.altaz(time, coord).alt < 0*u.deg
    # ok, it's below the horizon, so it *definitely* should fail an airmass
    # constraint of being above 2.  So both of these should give False:
    consmax = AirmassConstraint(2)
    consminmax = AirmassConstraint(2, 1)

    assert not consminmax(subaru, [coord], [time]).ravel()[0]
    # prior to 141 the above works, but the below FAILS
    assert not consmax(subaru, [coord], [time]).ravel()[0]


def test_months_observable():
    obs = Observer(latitude=0*u.deg, longitude=0*u.deg, elevation=0*u.m)

    coords = [SkyCoord(ra=0*u.hourangle, dec=0*u.deg),
              SkyCoord(ra=6*u.hourangle, dec=0*u.deg),
              SkyCoord(ra=12*u.hourangle, dec=0*u.deg),
              SkyCoord(ra=18*u.hourangle, dec=0*u.deg)]
    targets = [FixedTarget(coord=coord) for coord in coords]
    constraints = [AltitudeConstraint(min=80*u.deg),
                   AtNightConstraint.twilight_astronomical()]
    time_range = Time(['2014-01-01', '2014-12-31'])
    months = months_observable(constraints, obs, targets, time_range)

    should_be = [set({7, 8, 9, 10, 11, 12}), set({1, 2, 3, 10, 11, 12}),
                 set({1, 2, 3, 4, 5, 6}), set({4, 5, 6, 7, 8, 9})]

    assert months == should_be

    with pytest.warns(MissingConstraintWarning):
        constraints = [AtNightConstraint.twilight_astronomical()]
        months_observable(constraints, obs, targets, time_range)


def test_rescale_minmax():
    a = np.array([2])
    rescaled = np.zeros(5)
    rescaled[0] = (min_best_rescale(a, 1, 6))[0]
    rescaled[1] = (max_best_rescale(a, 1, 6))[0]
    rescaled[2] = (max_best_rescale(a, 0, 1))[0]
    rescaled[3] = (min_best_rescale(a, 0, 1))[0]
    rescaled[4] = (max_best_rescale(a, 0, 1, greater_than_max=0))[0]
    assert all(np.array([0.8, 0.2, 1, 0, 0]) == rescaled)


constraint_tests = [
    AltitudeConstraint(),
    AirmassConstraint(2),
    AtNightConstraint(),
    SunSeparationConstraint(min=90*u.deg),
    MoonSeparationConstraint(min=20*u.deg),
    LocalTimeConstraint(min=dt.time(23, 50), max=dt.time(4, 8)),
    TimeConstraint(*Time(["2015-08-28 03:30", "2015-09-05 10:30"]))
]


@pytest.mark.remote_data
@pytest.mark.parametrize('constraint', constraint_tests)
# astropy.coordinates.errors.NonRotationTransformationWarning
@pytest.mark.filterwarnings("ignore")
def test_regression_shapes(constraint):
    times = Time(["2015-08-28 03:30", "2015-09-05 10:30", "2015-09-15 18:35"])
    targets = get_skycoord([FixedTarget(SkyCoord(350.7*u.deg, 18.4*u.deg)),
                           FixedTarget(SkyCoord(260.7*u.deg, 22.4*u.deg))])
    lapalma = Observer.at_site('lapalma')

    assert constraint(lapalma, targets[:, np.newaxis], times).shape == (2, 3)
    assert constraint(lapalma, targets[0], times).shape == (3,)
    assert np.array(constraint(lapalma, targets[0], times[0])).shape == ()
    assert np.array(constraint(lapalma, targets, times[0])).shape == (2,)
    with pytest.raises(ValueError):
        constraint(lapalma, targets, times)


@pytest.mark.remote_data
def test_caches_shapes():
    times = Time([2457884.43350526, 2457884.5029497, 2457884.57239415], format='jd')
    m31 = SkyCoord(10.6847929*u.deg, 41.269065*u.deg)
    ippeg = SkyCoord(350.785625*u.deg, 18.416472*u.deg)
    htcas = SkyCoord(17.5566667*u.deg, 60.0752778*u.deg)
    targets = get_skycoord([m31, ippeg, htcas])
    observer = Observer.at_site('lapalma')
    ac = AltitudeConstraint(min=30*u.deg)
    assert ac(observer, targets, times, grid_times_targets=True).shape == (3, 3)
    assert ac(observer, targets, times, grid_times_targets=False).shape == (3,)


@pytest.mark.remote_data
def test_eclipses():
    subaru = Observer.at_site("Subaru")

    epoch = Time('2016-01-01')
    period = 3 * u.day
    duration = 1 * u.hour
    eclipsing_system = EclipsingSystem(primary_eclipse_time=epoch, orbital_period=period,
                                       duration=duration, name='test system')
    pec = PrimaryEclipseConstraint(eclipsing_system)
    times = Time(['2016-01-01 00:00', '2016-01-01 03:00', '2016-01-02 12:00'])
    assert np.all(np.array([True, False, False]) == pec(subaru, None, times))

    sec = SecondaryEclipseConstraint(eclipsing_system)
    times = Time(['2016-01-01 00:00', '2016-01-01 03:00', '2016-01-02 12:00'])
    assert np.all(np.array([False, False, True]) == sec(subaru, None, times))

    pc = PhaseConstraint(eclipsing_system, min=0.2, max=0.5)
    times = Time(['2016-01-01 00:00', '2016-01-02 12:00', '2016-01-02 14:00'])
    assert np.all(np.array([False, True, False]) == pc(subaru, None, times))


@pytest.mark.remote_data
def test_event_observable():

    epoch = Time(2452826.628514, format='jd')
    period = 3.52474859 * u.day
    duration = 0.1277 * u.day

    hd209458 = EclipsingSystem(primary_eclipse_time=epoch, orbital_period=period, duration=duration,
                               name='HD 209458 b')
    observing_time = Time('2017-09-15 10:20')

    apo = Observer.at_site('APO')

    target = FixedTarget.from_name("HD 209458")
    n_transits = 100  # This is the roughly number of transits per year
    ing_egr = hd209458.next_primary_ingress_egress_time(observing_time,
                                                        n_eclipses=n_transits)
    constraints = [AltitudeConstraint(min=0*u.deg), AtNightConstraint()]
    observable = is_event_observable(constraints, apo, target,
                                     times_ingress_egress=ing_egr)

    # This answer was validated against the Czech Exoplanet Transit Database
    # transit prediction service, at:
    # http://var2.astro.cz/ETD/predict_detail.php?delka=254.1797222222222&submit=submit&sirka=32.780277777777776&STARNAME=HD209458&PLANET=b
    # There is some disagreement, as the ETD considers some transits which begin
    # before sunset or after sunrise to be observable.
    cetd_answer = [[False, False, False, True, False, True, False, True, False,
                    True, False, True, False, False, False, False, False, False,
                    False, False, False, False, True, False, True, False, False,
                    False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False,
                    False, False, False, True, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False,
                    True, False, True, False, False, False, False, False, False,
                    False, False, False, False, False, False, True, False, True,
                    False, True, False, True, False, True, False, True, False,
                    False]]

    assert np.all(observable == np.array(cetd_answer))
