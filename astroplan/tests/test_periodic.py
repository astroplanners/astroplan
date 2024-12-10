import astropy.units as u
import numpy as np
from astropy.time import Time
from numpy.testing import assert_allclose

from astroplan.periodic import PeriodicEvent, EclipsingSystem

PRECISION = 0.00001  # days


def test_phase():
    epoch = Time('2016-01-01 00:00')
    period = 3*u.day
    duration = 1*u.hour
    pe = PeriodicEvent(epoch=epoch, period=period, duration=duration,
                       name='test event')

    assert pe.phase(Time('2016-01-02 12:00')) == 0.5
    assert pe.phase(Time('2016-01-04 00:00')) == 0.0


def test_primary_secondary_eclipse():
    epoch = Time('2016-01-01 00:00')
    period = 3*u.day
    duration = 1*u.hour
    es = EclipsingSystem(primary_eclipse_time=epoch, orbital_period=period, duration=duration,
                         name='test event')

    ap_primary = es.next_primary_eclipse_time(epoch)
    ap_secondary = es.next_secondary_eclipse_time(epoch)

    soln_primary = Time(['2016-01-04 00:00'])
    soln_secondary = Time(['2016-01-02 12:00'])

    # Tolerance of 1 second
    assert_allclose([ap_primary.jd, ap_secondary.jd],
                    [soln_primary.jd, soln_secondary.jd], atol=PRECISION)


def test_out_of_eclipse():
    epoch = Time('2016-01-01 00:00')
    period = 3*u.day
    duration = 1*u.hour
    es = EclipsingSystem(primary_eclipse_time=epoch, orbital_period=period, duration=duration,
                         name='test event')

    test_times = Time(['2016-01-01 06:00', '2016-01-02 12:00',
                       '2016-01-04 00:00', '2016-01-05 00:00'])

    assert np.all(es.out_of_eclipse(test_times) ==
                  np.array([True, False, False, True]))


def test_next_eclipse():
    epoch = Time('2016-01-01 00:00')
    period = 3*u.day
    duration = 1*u.hour
    es = EclipsingSystem(primary_eclipse_time=epoch, orbital_period=period, duration=duration,
                         name='test event')

    test_time = epoch + period + 0.1*u.min
    ap_next_primary = es.next_primary_eclipse_time(test_time)
    ap_next_secondary = es.next_secondary_eclipse_time(test_time)

    soln_next_primary = Time(['2016-01-07 00:00'])
    soln_next_secondary = Time(['2016-01-05 12:00'])

    # Tolerance of 1 second
    assert_allclose([ap_next_primary.jd, ap_next_secondary.jd],
                    [soln_next_primary.jd, soln_next_secondary.jd],
                    atol=PRECISION)


def test_primary_ingress():
    epoch = Time('2016-01-01 00:00')
    period = 3*u.day
    duration = 1*u.hour
    es = EclipsingSystem(primary_eclipse_time=epoch, orbital_period=period, duration=duration,
                         name='test event')

    test_time = epoch + 0.1*u.min
    ap_next_ing_egr_0 = es.next_primary_ingress_egress_time(test_time)
    ap_next_ing_egr_1 = es.next_primary_ingress_egress_time(test_time, n_eclipses=3)

    soln_next_ing_egr_0 = Time([['2016-01-03 23:30', '2016-01-04 00:30']])
    soln_next_ing_egr_1 = Time([['2016-01-03 23:30', '2016-01-04 00:30'],
                                ['2016-01-06 23:30', '2016-01-07 00:30'],
                                ['2016-01-09 23:30', '2016-01-10 00:30']])

    # Tolerance of 1 second
    assert_allclose(ap_next_ing_egr_0.jd, soln_next_ing_egr_0.jd,
                    atol=PRECISION)
    assert_allclose(ap_next_ing_egr_1.jd, soln_next_ing_egr_1.jd,
                    atol=PRECISION)
