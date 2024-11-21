# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
from abc import ABCMeta
import warnings

# Third-party
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS, UnitSphericalRepresentation, AltAz, EarthLocation
try:
    from sgp4.io import twoline2rv
    from sgp4.earth_gravity import wgs84 as sgp4_wgs84
    from skyfield.api import load, wgs84
    from skyfield.sgp4lib import EarthSatellite
    skyfield_available = True
except ImportError:
    skyfield_available = False

# Package
from .exceptions import InvalidTLEDataWarning


__all__ = ["Target", "FixedTarget", "NonFixedTarget", "TLETarget"]

# Docstring code examples include printed SkyCoords, but the format changed
# in astropy 1.3. Thus the doctest needs astropy >=1.3 and this is the
# easiest way to make it work.

__doctest_requires__ = {'FixedTarget.*': ['astropy.modeling.Hermite1D']}


class Target(object):
    """
    Abstract base class for target objects.

    This is an abstract base class -- you can't instantiate
    examples of this class, but must work with one of its
    subclasses such as `~astroplan.target.FixedTarget` or
    `~astroplan.target.NonFixedTarget`.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name=None, ra=None, dec=None, marker=None):
        """
        Defines a single observation target.

        Parameters
        ----------
        name : str, optional

        ra : WHAT TYPE IS ra ?

        dec : WHAT TYPE IS dec ?

        marker : str, optional
            User-defined markers to differentiate between different types
            of targets (e.g., guides, high-priority, etc.).
        """
        raise NotImplementedError()

    @property
    def ra(self):
        """
        Right ascension.
        """
        if isinstance(self, FixedTarget):
            return self.coord.ra
        raise NotImplementedError()

    @property
    def dec(self):
        """
        Declination.
        """
        if isinstance(self, FixedTarget):
            return self.coord.dec
        raise NotImplementedError()


class FixedTarget(Target):
    """
    Coordinates and metadata for an object that is "fixed" with respect to the
    celestial sphere.

    Examples
    --------
    Create a `~astroplan.FixedTarget` object for Sirius:

    >>> from astroplan import FixedTarget
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> sirius_coord = SkyCoord(ra=101.28715533*u.deg, dec=16.71611586*u.deg)
    >>> sirius = FixedTarget(coord=sirius_coord, name="Sirius")

    Create an equivalent `~astroplan.FixedTarget` object for Sirius by querying
    for the coordinates of Sirius by name:

    >>> from astroplan import FixedTarget
    >>> sirius = FixedTarget.from_name("Sirius")
    """

    def __init__(self, coord, name=None, **kwargs):
        """
        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Coordinate of the target

        name : str (optional)
            Name of the target, used for plotting and representing the target
            as a string
        """
        if not (hasattr(coord, 'transform_to') and
                hasattr(coord, 'represent_as')):
            raise TypeError('`coord` must be a coordinate object.')

        self.name = name
        self.coord = coord

    @classmethod
    def from_name(cls, query_name, name=None, **kwargs):
        """
        Initialize a `FixedTarget` by querying for a name from the CDS name
        resolver, using the machinery in
        `~astropy.coordinates.SkyCoord.from_name`.

        This

        Parameters
        ----------
        query_name : str
            Name of the target used to query for coordinates.

        name : string or `None`
            Name of the target to use within astroplan. If `None`, query_name
            is used as ``name``.

        Examples
        --------
        >>> from astroplan import FixedTarget
        >>> sirius = FixedTarget.from_name("Sirius")
        >>> sirius.coord                              # doctest: +FLOAT_CMP
        <SkyCoord (ICRS): (ra, dec) in deg
            ( 101.28715533, -16.71611586)>
        """
        # Allow manual override for name keyword so that the target name can
        # be different from the query name, otherwise assume name=queryname.
        if name is None:
            name = query_name
        return cls(SkyCoord.from_name(query_name), name=name, **kwargs)

    def __repr__(self):
        """
        String representation of `~astroplan.FixedTarget`.

        Examples
        --------
        Show string representation of a `~astroplan.FixedTarget` for Vega:

        >>> from astroplan import FixedTarget
        >>> from astropy.coordinates import SkyCoord
        >>> vega_coord = SkyCoord(ra='279.23473479d', dec='38.78368896d')
        >>> vega = FixedTarget(coord=vega_coord, name="Vega")
        >>> print(vega)                             # doctest: +FLOAT_CMP
        <FixedTarget "Vega" at SkyCoord (ICRS): (ra, dec) in deg ( 279.23473479, 38.78368894)>
        """
        class_name = self.__class__.__name__
        fmt_coord = repr(self.coord).replace('\n   ', '')[1:-1]
        return '<{} "{}" at {}>'.format(class_name, self.name, fmt_coord)

    @classmethod
    def _from_name_mock(cls, query_name, name=None):
        """
        Mock method to replace `FixedTarget.from_name` in tests without
        internet connection.
        """
        # The lowercase method will be run on names, so enter keys in lowercase:
        stars = {
            "rigel": {"ra": 78.63446707*u.deg, "dec": -8.20163837*u.deg},
            "sirius": {"ra": 101.28715533*u.deg, "dec": -16.71611586*u.deg},
            "vega": {"ra": 279.23473479*u.deg, "dec": 38.78368896*u.deg},
            "aldebaran": {"ra": 68.98016279*u.deg, "dec": 16.50930235*u.deg},
            "polaris": {"ra": 37.95456067*u.deg, "dec": 89.26410897*u.deg},
            "deneb": {"ra": 310.35797975*u.deg, "dec": 45.28033881*u.deg},
            "m13": {"ra": 250.423475*u.deg, "dec": 36.4613194*u.deg},
            "altair": {"ra": 297.6958273*u.deg, "dec": 8.8683212*u.deg},
            "hd 209458": {"ra": 330.79*u.deg, "dec": 18.88*u.deg}
        }

        if query_name.lower() in stars:
            return cls(coord=SkyCoord(**stars[query_name.lower()]),
                       name=query_name)
        else:
            raise ValueError("Target named {} not in mocked FixedTarget "
                             "method".format(query_name))


class NonFixedTarget(Target):
    """
    Placeholder for future function.
    """


class TLETarget(Target):
    """
    A target defined by TLE (Two-Line Element set) for satellites.
    """
    def __init__(self, line1, line2, name=None, observer=None, skip_tle_check=False):
        """
        Parameters
        ----------
        line1 : str
            The first line of the TLE set

        line2 : str
            The second line of the TLE set

        name : str, optional
            Name of the target, used for plotting and representing the target
            as a string

        observer : `~astropy.coordinates.EarthLocation`, `~astroplan.Observer`, optional
            The location of observer.
            If `None`, the observer is assumed to be at sea level at the equator.

        skip_tle_check : bool, optional
            Whether to skip TLE validation
        """
        if not skyfield_available:
            raise ImportError("Please install the skyfield package to use the TLETarget class.")

        if not skip_tle_check:
            twoline2rv(line1, line2, sgp4_wgs84)    # Raises ValueError if TLE is invalid

        self.name = name
        self.satellite = EarthSatellite(line1, line2, name, load.timescale())

        if observer is None:
            self.observer = wgs84.latlon(0, 0, 0)
        else:
            observer = getattr(observer, 'location', observer)
            longitude, latitude, height = observer.to_geodetic()
            self.observer = wgs84.latlon(latitude.to(u.deg).value,
                                         longitude.to(u.deg).value,
                                         height.to(u.m).value)

    @classmethod
    def from_string(cls, tle_string, name=None, *args, **kwargs):
        """
        Creates a TLETarget instance from a complete TLE string.

        Parameters
        ----------
        tle_string : str
            String to be parsed, expected to contain 2 or 3 newline-separated lines.

        name : str, optional
            Name of the target. If not provided and the tle_string contains 3 lines,
            the first line will be used as the name.

        args, kwargs : tuple, dict, optional
            Additional arguments and keyword arguments to be passed to the TLETarget constructor.
        """
        lines = tle_string.strip().splitlines()

        if len(lines) not in (2, 3):
            raise ValueError(f"Expected TLE string to contain 2 or 3 lines, got {len(lines)}")

        if len(lines) == 3:
            line1, line2, name = lines[1], lines[2], name or lines[0]
        else:   # len(lines) == 2
            line1, line2 = lines
        return cls(line1, line2, name, *args, **kwargs)

    @property
    def ra(self):
        raise NotImplementedError("Compute satellite RA at a specific time with self.coord(time)")

    @property
    def dec(self):
        raise NotImplementedError("Compute satellite RA at a specific time with self.coord(time)")

    def _compute_topocentric(self, time=None):
        """
        Compute the topocentric coordinates (relative to observer) at a particular time.

        Parameters
        ----------
        time : `~astropy.time.Time`, optional
            The time(s) to use in the calculation.

        Returns
        -------
        topocentric : `skyfield.positionlib.ICRF`
            The topocentric object representing the relative coordinates of the target.
        """
        if time is None:
            time = Time.now()
        ts = load.timescale()
        t = ts.from_astropy(time)

        topocentric = (self.satellite - self.observer).at(t)

        # Check for invalid TLE data. A non-None usually message means the computation went beyond
        # the physically sensible point. Details:
        # https://rhodesmill.org/skyfield/earth-satellites.html#detecting-propagation-errors
        message = topocentric.message
        if (
            (message is not None and not isinstance(message, list)) or
            (isinstance(message, list) and not all(x is None for x in message))
        ):
            warnings.warn(f"Invalid TLE Data: {message}", InvalidTLEDataWarning)

        return topocentric

    def coord(self, time=None):
        """
        Get the coordinates of the target at a particular time.

        Parameters
        ----------
        time : `~astropy.time.Time`, optional
            The time(s) to use in the calculation.

        Returns
        -------
        coord : `~astropy.coordinates.SkyCoord`
            A single SkyCoord object, which may be non-scalar, representing the target's
            RA/Dec coordinates at the specified time(s). Might return np.nan and output a
            warning for times where the elements stop making physical sense.
        """
        topocentric = self._compute_topocentric(time)
        ra, dec, distance = topocentric.radec()
        # No distance, in SkyCoord, distance is from frame origin, but here, it's from observer.
        return SkyCoord(ra.hours*u.hourangle, dec.degrees*u.deg, obstime=time, frame='icrs')

    def altaz(self, time=None):
        """
        Get the altitude and azimuth of the target at a particular time.

        Parameters
        ----------
        time : `~astropy.time.Time`, optional
            The time(s) to use in the calculation.

        Returns
        -------
        altaz_coord : `~astropy.coordinates.SkyCoord`
            SkyCoord object representing the target's altitude and azimuth at the specified time(s)
        """
        topocentric = self._compute_topocentric(time)
        alt, az, distance = topocentric.altaz()

        earth_location = EarthLocation(lat=self.observer.latitude.degrees*u.deg,
                                       lon=self.observer.longitude.degrees*u.deg,
                                       height=self.observer.elevation.m*u.m)

        altaz = AltAz(alt=alt.degrees*u.deg, az=az.degrees*u.deg,
                      obstime=time, location=earth_location)
        return SkyCoord(altaz)

    def __repr__(self):
        return f'<{self.__class__.__name__} "{self.name}">'

    def __str__(self):
        return self.name


def repeat_skycoord(coord, times):
    """
    Repeats the coordinates of a SkyCoord object 'times.size' number of times.

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
        The original SkyCoord object whose coordinates need to be repeated.

    times : `~astropy.time.Time`
        The size of times determines the number of times the coordinates should be repeated.

    Returns
    --------
    SkyCoord : `~astropy.coordinates.SkyCoord`
        A new SkyCoord object with the coordinates of the original object
        repeated 'times.size' number of times. If the SkyCoord object is scalar
        or 'times' is None or a scalar, this function returns the
        original SkyCoord object.
    """
    if coord.size != 1 or times is None or times.size == 1:
        return coord
    return SkyCoord(
        ra=coord.ra.repeat(times.size),
        dec=coord.dec.repeat(times.size),
        distance=None if coord.distance.unit is u.one else coord.distance.repeat(times.size),
        frame=coord.frame,
        obstime=times
    )


def get_skycoord(targets, time=None, backwards_compatible=True):
    """
    Return an `~astropy.coordinates.SkyCoord` object.

    When performing calculations it is usually most efficient to have
    a single `~astropy.coordinates.SkyCoord` object, rather than a
    list of `Target` or `~astropy.coordinates.SkyCoord` objects.

    This is a convenience routine to do that.

    Parameters
    -----------
    targets : list, `~astropy.coordinates.SkyCoord`, `Target`
        either a single target or a list of targets

    time : `~astropy.time.Time`, optional
        The time(s) to use in the calculation.

    backwards_compatible : bool, optional
        Controls output format when FixedTarget or SkyCoord targets are used with a time argument.
        If False, it will return (targets x times), where all coordinates per target are the same.
        If True, it will return one coordinate per target (default is True).

    Returns
    --------
    coord : `~astropy.coordinates.SkyCoord`
        a single SkyCoord object, which may be non-scalar
    """

    # Note on backwards_compatible:
    # Method always returns (targets x times) with TLETarget in targets, as RA/Dec changes with time
    # Do we want to be 100% backwards compatible, or do we prefer consistent output,
    # for FixedTarget or SkyCoord targets combined with multiple times?
    # backwards_compatible = True will continue to return one coordinate per target
    # backwards_compatible = False, returns (targets x times) with identical coordinates per target

    # Early exit for single target
    if not isinstance(targets, list):
        if isinstance(targets, TLETarget):
            return targets.coord(time)
        else:
            if backwards_compatible:
                return getattr(targets, 'coord', targets)
            else:
                return repeat_skycoord(getattr(targets, 'coord', targets), time)

    # Identify if any of the targets is not FixedTarget or SkyCoord
    has_non_fixed_target = any(
        not isinstance(target, (FixedTarget, SkyCoord))
        for target in targets
    )

    # Get the SkyCoord object itself
    coords = [
        target.coord(time) if isinstance(target, TLETarget)
        else getattr(target, 'coord', target)
        for target in targets
    ]

    # Fill time dimension for SkyCoords that only have a single coordinate
    if (
        (backwards_compatible and has_non_fixed_target or not backwards_compatible) and
        time is not None
    ):
        coords = [repeat_skycoord(coord, time) for coord in coords]

    # are all SkyCoordinate's in equivalent frames? If not, convert to ICRS
    convert_to_icrs = not all(
        [coord.frame.is_equivalent_frame(coords[0].frame) for coord in coords[1:]])

    # we also need to be careful about handling mixtures of
    # UnitSphericalRepresentations and others
    targets_is_unitsphericalrep = [x.data.__class__ is
                                   UnitSphericalRepresentation for x in coords]

    longitudes = []
    latitudes = []
    distances = []
    get_distances = not all(targets_is_unitsphericalrep)
    if convert_to_icrs:
        # mixture of frames
        for coordinate in coords:
            icrs_coordinate = coordinate.icrs
            longitudes.append(icrs_coordinate.ra)
            latitudes.append(icrs_coordinate.dec)
            if get_distances:
                distances.append(icrs_coordinate.distance)
        frame = ICRS()
    else:
        # all the same frame, get the longitude and latitude names
        try:
            # from astropy v2.0, keys are classes
            lon_name, lat_name = [
                mapping.framename for mapping in
                coords[0].frame_specific_representation_info[UnitSphericalRepresentation]]
        except BaseException:            # whereas prior to that they were strings.
            lon_name, lat_name = [mapping.framename for mapping in
                                  coords[0].frame_specific_representation_info['spherical']]

        frame = coords[0].frame
        for coordinate in coords:
            longitudes.append(getattr(coordinate, lon_name))
            latitudes.append(getattr(coordinate, lat_name))
            if get_distances:
                distances.append(coordinate.distance)

    # now let's deal with the fact that we may have a mixture of coords with distances and
    # coords with UnitSphericalRepresentations
    if all(targets_is_unitsphericalrep):
        return SkyCoord(longitudes, latitudes, frame=frame)
    elif not any(targets_is_unitsphericalrep):
        return SkyCoord(longitudes, latitudes, distances, frame=frame)
    else:
        """
        We have a mixture of coords with distances and without.
        Since we don't know in advance the origin of the frame where further transformation
        will take place, it's not safe to drop the distances from those coords with them set.

        Instead, let's assign large distances to those objects with none.
        """
        distances = [distance if distance != 1 else 100*u.kpc for distance in distances]
        return SkyCoord(longitudes, latitudes, distances, frame=frame)


class SpecialObjectFlag(object):
    """
    Flag this object as a special non-fixed target, which has a ``get_*`` method
    within astropy (like the Sun or Moon)
    """
    pass


class SunFlag(SpecialObjectFlag):
    """
    Flag for a computation with the Sun
    """
    approx_sidereal_drift = 5 * u.min


class MoonFlag(SpecialObjectFlag):
    """
    Flag for a computation with the Moon
    """
    approx_sidereal_drift = 60 * u.min
