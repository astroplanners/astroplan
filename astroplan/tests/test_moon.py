# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from numpy.testing import assert_allclose

from astroplan.observer import Observer


def test_illumination():
    time = Time(['1990-01-01 00:00:00', '1990-03-01 06:00:00',
                 '1990-06-01 12:00:00', '1990-11-01 18:00:00'])
    location = EarthLocation.from_geodetic(-155*u.deg, 19*u.deg, 0*u.m)
    obs = Observer(location)
    # Get illumination via time
    illumination1 = obs.moon_illumination(time)

    # Run print_pyephem_illumination() for PyEphem's solution
    pyephem_illumination = [0.15475513880925418, 0.19484233284757257,
                            0.6170840254669668, 0.9780219372563843]

    assert_allclose(illumination1, pyephem_illumination, atol=0.05)


def print_pyephem_illumination():
    """
    To run, use:
    python -c "from astroplan.tests.test_moon import print_pyephem_illumination as f; f()"
    """
    time = Time(['1990-01-01 00:00:00', '1990-03-01 06:00:00',
                 '1990-06-01 12:00:00', '1990-11-01 18:00:00'])
    location = EarthLocation.from_geodetic(-155*u.deg, 19*u.deg, 0*u.m)

    import ephem
    moon = ephem.Moon()
    pe_obs = ephem.Observer()
    pe_obs.lat = location.lat.to_string(u.deg, sep=':')
    pe_obs.lon = location.lon.to_string(u.deg, sep=':')
    pe_obs.elevation = location.height.to(u.m).value
    illuminations = []
    for t in time:
        pe_obs.date = t.datetime
        moon.compute(pe_obs)
        illuminations.append(moon.moon_phase)
    print(illuminations)
