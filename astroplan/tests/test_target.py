# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest

# Third-party
import astropy.units as u
from astropy.coordinates import SkyCoord, GCRS, ICRS
from astropy.time import Time
import numpy as np
try:
    import skyfield  # noqa
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False

# Package
from ..target import FixedTarget, TLETarget, get_skycoord
from ..observer import Observer
from ..utils import time_grid_from_range
from ..exceptions import InvalidTLEDataWarning


def test_FixedTarget_from_name():
    """
    Check that resolving target names with the `SkyCoord.from_name` constructor
    to produce a `FixedTarget` accurately resolves the coordinates of Polaris.
    """

    # Resolve coordinates with SkyCoord.from_name classmethod
    polaris_from_name = FixedTarget.from_name('Polaris')
    polaris_from_name = FixedTarget.from_name('Polaris', name='Target 1')
    # Coordinates grabbed from SIMBAD
    polaris_from_SIMBAD = SkyCoord('02h31m49.09456s', '+89d15m50.7923s')

    # Make sure separation is small
    assert polaris_from_name.coord.separation(polaris_from_SIMBAD) < 1*u.arcsec


def test_FixedTarget_ra_dec():
    """
    Confirm that FixedTarget.ra and FixedTarget.dec are the same as the
    right ascension and declination stored in the FixedTarget.coord variable -
    which is a SkyCoord
    """

    vega_coords = SkyCoord('18h36m56.33635s', '+38d47m01.2802s')
    vega = FixedTarget(vega_coords, name='Vega')
    assert vega.coord == vega_coords, 'Store coordinates directly'
    assert vega.coord.ra == vega_coords.ra == vega.ra, ('Retrieve RA from '
                                                        'SkyCoord')
    assert vega.coord.dec == vega_coords.dec == vega.dec, ('Retrieve Dec from '
                                                           'SkyCoord')


@pytest.mark.skipif('not HAS_SKYFIELD')
def test_TLETarget():
    tle_string = ("ISS (ZARYA)\n"
                  "1 25544U 98067A   23215.27256123  .00041610  00000-0  73103-3 0  9990\n"
                  "2 25544  51.6403  95.2411 0000623 157.9606 345.0624 15.50085581409092")
    line1 = "1 25544U 98067A   23215.27256123  .00041610  00000-0  73103-3 0  9990"
    line2 = "2 25544  51.6403  95.2411 0000623 157.9606 345.0624 15.50085581409092"
    subaru = Observer.at_site('subaru')  # (lon, lat, el)=(-155.476111 deg, 19.825555 deg, 4139.0 m)
    time = Time("2023-08-02 10:00", scale='utc')
    times = time_grid_from_range([time, time + 3.1*u.hour],
                                 time_resolution=1 * u.hour)

    tle_target1 = TLETarget(name="ISS (ZARYA)", line1=line1, line2=line2, observer=subaru)
    tle_target2 = TLETarget.from_string(tle_string=tle_string, observer=subaru)

    assert tle_target1.name == "ISS (ZARYA)"
    assert tle_target2.name == "ISS (ZARYA)"
    assert repr(tle_target1) == repr(tle_target2)
    assert str(tle_target1) == str(tle_target2)

    # Single time (Below Horizon)
    ra_dec1 = tle_target1.coord(time)   # '08h29m26.00003243s +07d31m36.65950907s'
    ra_dec2 = tle_target2.coord(time)
    assert ra_dec1.to_string('hmsdms') == ra_dec2.to_string('hmsdms')

    # Comparison with the JPL Horizons System
    ra_dec_horizon_icrf = SkyCoord("08h29m27.029117s +07d31m28.35610s")
    # ICRF: Compensated for the down-leg light-time delay aberration
    assert ra_dec1.separation(ra_dec_horizon_icrf) < 20*u.arcsec  # 17.41″
    # Distance estimation: ~ 2 * tan(17,41/2/3600) * 11801,56 = 57 km

    ra_dec_horizon_ref_apparent = SkyCoord("08h30m54.567398s +08d05m32.72764s")
    # Refracted Apparent: In an equatorial coordinate system with all compensations
    assert ra_dec1.separation(ra_dec_horizon_ref_apparent) > 2000*u.arcsec  # 2424.44″

    ra_dec_horizon_icrf_ref_apparent = SkyCoord("08h29m37.373866s +08d10m14.78811s")
    # ICRF Refracted Apparent: In the ICRF reference frame with all compensations
    assert ra_dec1.separation(ra_dec_horizon_icrf_ref_apparent) > 2000*u.arcsec  # 2324.28″

    # Skyfield appears to use no compensations. According to this, it's not even recommended to
    # compensate for light travel time. Compensating changes the difference to ra_dec_horizon_icrf
    # to 20.05 arcsec.
    # https://rhodesmill.org/skyfield/earth-satellites.html#avoid-calling-the-observe-method

    # Single time (Above Horizon)
    time_ah = Time("2023-08-02 07:20", scale='utc')
    ra_dec_ah = tle_target1.coord(time_ah)  # '11h19m48.53631001s +44d49m45.22194611s'

    ra_dec_ah_horizon_icrf = SkyCoord("11h19m49.660349s +44d49m34.65875s")
    assert ra_dec_ah.separation(ra_dec_ah_horizon_icrf) < 20*u.arcsec  # 15.95″
    ra_dec_ah_horizon_ref_apparent = SkyCoord("11h21m34.102381s +44d43m40.06899s")
    assert ra_dec_ah.separation(ra_dec_ah_horizon_ref_apparent) > 1000*u.arcsec  # 1181.84″
    ra_dec_ah_horizon_icrf_ref_apparent = SkyCoord("11h20m16.627261s +44d51m24.25337s")
    assert ra_dec_ah.separation(ra_dec_ah_horizon_icrf_ref_apparent) > 300*u.arcsec  # 314.75″

    # Default is WGS72 for Skyfield. Coordinates with WGS84 gravity model that Horizon uses:
    # '11h19m48.28084569s +44d49m46.33649241s' - 18.75″
    # See 'Build a satellite with a specific gravity model' in Skyfield's Earth Satellites docu

    # There are many potential sources of inaccuracies, and it's not all super precise.
    # Should the accuracy be better than < 25*u.arcsec when compared to the JPL Horizons System?

    # Multiple times
    ra_dec1 = tle_target1.coord(times)
    ra_dec2 = tle_target2.coord(times)
    ra_dec_from_horizon = SkyCoord(["08h29m27.029117s +07d31m28.35610s",
                                    "06h25m46.672661s -54d32m16.77533s",
                                    "13h52m08.854291s +04d26m49.56432s",
                                    "09h20m04.872215s -00d51m21.17432s"])
    # 17.41″, 22.05″, 3.20″, 17.55″
    assert all(list(ra_dec1.separation(ra_dec_from_horizon) < 25*u.arcsec))
    assert ra_dec1.to_string('hmsdms') == ra_dec2.to_string('hmsdms')

    # TLE Check
    line1_invalid = "1 25544U 98067A .00041610  00000-0  73103-3 0  9990"
    with pytest.raises(ValueError):
        TLETarget(name="ISS (ZARYA)", line1=line1_invalid, line2=line2, observer=subaru)
    TLETarget(name="ISS (ZARYA)", line1=line1_invalid, line2=line2, observer=subaru,
              skip_tle_check=True)

    # from_string
    tle_string = ("1 25544U 98067A   23215.27256123  .00041610  00000-0  73103-3 0  9990\n"
                  "2 25544  51.6403  95.2411 0000623 157.9606 345.0624 15.50085581409092")
    tle_target3 = TLETarget.from_string(tle_string=tle_string, observer=subaru, name="ISS")
    assert tle_target3.name == "ISS"

    tle_string = "1 25544U 98067A   23215.27256123  .00041610  00000-0  73103-3 0  9990"
    with pytest.raises(ValueError):
        TLETarget.from_string(tle_string=tle_string, observer=subaru)

    # AltAz (This is slow)
    altaz_observer = subaru.altaz(time, tle_target1)
    altaz_skyfield = tle_target1.altaz(time)

    assert altaz_observer.separation(altaz_skyfield) < 20*u.arcsec

    # Time too far in the future where elements stop making physical sense
    with pytest.warns():    # ErfaWarning: ERFA function "dtf2d" yielded 1 of "dubious year (Note 6)
        time_invalid = Time("2035-08-02 10:00", scale='utc')
        times_list = list(times)
        times_list[2] = Time("2035-08-02 10:00", scale='utc')
        times_invalid = Time(times_list)
    with pytest.warns(InvalidTLEDataWarning):
        assert np.isnan(tle_target1.coord(time_invalid).ra)
    with pytest.warns(InvalidTLEDataWarning):
        assert np.isnan(tle_target1.coord(times_invalid)[2].ra)


def test_get_skycoord():
    m31 = SkyCoord(10.6847083*u.deg, 41.26875*u.deg)
    m31_with_distance = SkyCoord(10.6847083*u.deg, 41.26875*u.deg, 780*u.kpc)
    subaru = Observer.at_site('subaru')
    time = Time("2016-01-22 12:00")
    pos, vel = subaru.location.get_gcrs_posvel(time)
    gcrs_frame = GCRS(obstime=Time("2016-01-22 12:00"), obsgeoloc=pos, obsgeovel=vel)
    m31_gcrs = m31.transform_to(gcrs_frame)
    m31_gcrs_with_distance = m31_with_distance.transform_to(gcrs_frame)

    coo = get_skycoord(m31)
    assert coo.is_equivalent_frame(ICRS())
    with pytest.raises(TypeError):
        len(coo)

    coo = get_skycoord([m31])
    assert coo.is_equivalent_frame(ICRS())
    assert len(coo) == 1

    coo = get_skycoord([m31, m31_gcrs])
    assert coo.is_equivalent_frame(ICRS())
    assert len(coo) == 2

    coo = get_skycoord([m31_with_distance, m31_gcrs_with_distance])
    assert coo.is_equivalent_frame(ICRS())
    assert len(coo) == 2

    coo = get_skycoord([m31, m31_gcrs, m31_gcrs_with_distance, m31_with_distance])
    assert coo.is_equivalent_frame(ICRS())
    assert len(coo) == 4

    coo = get_skycoord([m31_gcrs, m31_gcrs_with_distance])
    assert coo.is_equivalent_frame(m31_gcrs.frame)
    assert len(coo) == 2


@pytest.mark.skipif('not HAS_SKYFIELD')
def test_get_skycoord_with_TLETarget():
    skycoord_targed = SkyCoord(10.6847083*u.deg, 41.26875*u.deg)
    fixed_target1 = FixedTarget(name="fixed1", coord=SkyCoord(279.23458, 38.78369, unit='deg'))
    line1 = "1 25544U 98067A   23215.27256123  .00041610  00000-0  73103-3 0  9990"
    line2 = "2 25544  51.6403  95.2411 0000623 157.9606 345.0624 15.50085581409092"
    subaru = Observer.at_site('subaru')
    tle_target1 = TLETarget(name="ISS (ZARYA)", line1=line1, line2=line2, observer=subaru)

    time = Time("2023-08-02 10:00", scale='utc')
    times = time_grid_from_range([time, time + 3.1*u.hour],
                                 time_resolution=1 * u.hour)

    tle_output = get_skycoord(tle_target1, time)
    assert tle_output.size == 1
    tle_output = get_skycoord(tle_target1, times)
    assert tle_output.shape == (4,)
    tle_output = get_skycoord([tle_target1, tle_target1], time)
    assert tle_output.shape == (2,)
    tle_output = get_skycoord([tle_target1, tle_target1], times)
    assert tle_output.shape == (2, 4)

    mixed_output = get_skycoord([skycoord_targed, fixed_target1, tle_target1], time)
    assert mixed_output.shape == (3,)
    mixed_output = get_skycoord([skycoord_targed, fixed_target1, tle_target1], times)
    assert mixed_output.shape == (3, 4)

    # backwards_compatible
    tle_output = get_skycoord(skycoord_targed, time, backwards_compatible=False)
    assert tle_output.size == 1

    tle_output = get_skycoord(skycoord_targed, times)
    assert tle_output.size == 1
    tle_output = get_skycoord(skycoord_targed, times, backwards_compatible=False)
    assert tle_output.shape == (4,)

    tle_output = get_skycoord([skycoord_targed, skycoord_targed], time, backwards_compatible=False)
    assert tle_output.shape == (2,)

    tle_output = get_skycoord([skycoord_targed, skycoord_targed], times)
    assert tle_output.shape == (2,)
    tle_output = get_skycoord([skycoord_targed, skycoord_targed], times, backwards_compatible=False)
    assert tle_output.shape == (2, 4)
