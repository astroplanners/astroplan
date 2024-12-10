# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Standard library
import sys
import datetime
import warnings

# Third-party

import astropy.units as u
import numpy as np
import pytz
from astropy.coordinates import (EarthLocation, SkyCoord, AltAz, get_sun,
                                 get_body, Angle, Longitude)
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning

# Package
from .exceptions import TargetNeverUpWarning, TargetAlwaysUpWarning
from .moon import moon_illumination, moon_phase_angle
from .target import get_skycoord, SunFlag, MoonFlag


__all__ = ["Observer"]

MAGIC_TIME = Time(-999, format='jd')


# Handle deprecated MAGIC_TIME variable
def deprecation_wrap_module(mod, deprecated):
    """Return a wrapped object that warns about deprecated accesses"""
    deprecated = set(deprecated)

    class DeprecateWrapper:
        def __getattr__(self, attr):
            if attr in deprecated:
                warnmsg = ("`MAGIC_TIME` will be deprecated in future versions "
                           "of astroplan. Use masked Time objects instead.")
                warnings.warn(warnmsg, AstropyDeprecationWarning)
            return getattr(mod, attr)

    return DeprecateWrapper()


sys.modules[__name__] = deprecation_wrap_module(sys.modules[__name__],
                                                deprecated=['MAGIC_TIME'])


def _process_nans_in_jds(jds):
    """
    Some functions calculate times for events that won't happen, yielding nans. This wrapper
    manages vectors of (potentially) invalid JDs that must be passed to the astropy.time.Time
    constructor. Returns a masked Time object.
    """
    if np.isscalar(jds):
        jds = np.array([jds])
    not_finite = ~np.isfinite(jds)
    masked_jds = np.ma.masked_array(jds.to(u.day).value if hasattr(jds, 'unit') else jds.copy())
    masked_jds[not_finite] = np.ma.masked
    return masked_jds


def _generate_24hr_grid(t0, start, end, n_grid_points, for_deriv=False):
    """
    Generate a nearly linearly spaced grid of time durations.

    The midpoints of these grid points will span times from ``t0``+``start``
    to ``t0``+``end``, including the end points, which is useful when taking
    numerical derivatives.

    Parameters
    ----------
    t0 : `~astropy.time.Time`
        Time queried for, grid will be built from or up to this time.

    start : float
        Number of days before/after ``t0`` to start the grid.

    end : float
        Number of days before/after ``t0`` to end the grid.

    n_grid_points : int (optional)
        Number of grid points to generate

    for_deriv : bool
        Generate time series for taking numerical derivative (modify
        bounds)?

    Returns
    -------
    `~astropy.time.Time`
    """

    if for_deriv:
        time_grid = np.concatenate([[start - 1 / (n_grid_points - 1)],
                                    np.linspace(start, end,
                                                n_grid_points)[1:-1],
                                    [end + 1 / (n_grid_points - 1)]]) * u.day
    else:
        time_grid = np.linspace(start, end, n_grid_points) * u.day

    # broadcast so grid is first index, and remaining shape of t0
    # falls in later indices. e.g. if t0 is shape (10), time_grid
    # will be shape (N, 10). If t0 is shape (5, 2), time_grid is (N, 5, 2)
    while time_grid.ndim <= t0.ndim:
        time_grid = time_grid[:, np.newaxis]
    # we want to avoid 1D grids since we always want to broadcast against targets
    if time_grid.ndim == 1:
        time_grid = time_grid[:, np.newaxis]
    return t0 + time_grid


class Observer:

    """
    A container class for information about an observer's location and
    environment.

    Examples
    --------
    We can create an observer at Subaru Observatory in Hawaii two ways. First,
    locations for some observatories are stored in astroplan, and these can be
    accessed by name, like so:

    >>> from astroplan import Observer
    >>> subaru = Observer.at_site("Subaru", timezone="US/Hawaii")

    To find out which observatories can be accessed by name, check out
    `~astropy.coordinates.EarthLocation.get_site_names`.

    Next, you can initialize an observer by specifying the location with
    `~astropy.coordinates.EarthLocation`:

    >>> from astropy.coordinates import EarthLocation
    >>> import astropy.units as u
    >>> location = EarthLocation.from_geodetic(-155.4761*u.deg, 19.825*u.deg,
    ...                                        4139*u.m)
    >>> subaru = Observer(location=location, name="Subaru", timezone="US/Hawaii")

    You can also create an observer without an
    `~astropy.coordinates.EarthLocation`:

    >>> from astroplan import Observer
    >>> import astropy.units as u
    >>> subaru = Observer(longitude=-155.4761*u.deg, latitude=19.825*u.deg,
    ...                   elevation=0*u.m, name="Subaru", timezone="US/Hawaii")

    """
    @u.quantity_input(elevation=u.m)
    def __init__(self, location=None, timezone='UTC', name=None, latitude=None,
                 longitude=None, elevation=0*u.m, pressure=None,
                 relative_humidity=None, temperature=None, description=None):
        """
        Parameters
        ----------
        location : `~astropy.coordinates.EarthLocation`
            The location (latitude, longitude, elevation) of the observatory.

        timezone : str or `datetime.tzinfo` (optional)
            The local timezone to assume. If a string, it will be passed
            through ``pytz.timezone()`` to produce the timezone object.

        name : str
            A short name for the telescope, observatory or location.

        latitude : float, str, `~astropy.units.Quantity` (optional)
            The latitude of the observing location. Should be valid input for
            initializing a `~astropy.coordinates.Latitude` object.

        longitude : float, str, `~astropy.units.Quantity` (optional)
            The longitude of the observing location. Should be valid input for
            initializing a `~astropy.coordinates.Longitude` object.

        elevation : `~astropy.units.Quantity` (optional), default = 0 meters
            The elevation of the observing location, with respect to sea
            level. Defaults to sea level.

        pressure : `~astropy.units.Quantity` (optional)
            The ambient pressure. Defaults to zero (i.e. no atmosphere).

        relative_humidity : float (optional)
            The ambient relative humidity.

        temperature : `~astropy.units.Quantity` (optional)
            The ambient temperature.

        description : str (optional)
            A short description of the telescope, observatory or observing
            location.
        """

        self.name = name
        self.pressure = pressure
        self.temperature = temperature
        self.relative_humidity = relative_humidity

        # If lat/long given instead of EarthLocation, convert them
        # to EarthLocation
        if location is None and (latitude is not None and
                                 longitude is not None):
            self.location = EarthLocation.from_geodetic(longitude, latitude,
                                                        elevation)

        elif isinstance(location, EarthLocation):
            self.location = location

        else:
            raise TypeError('Observatory location must be specified with '
                            'either (1) an instance of '
                            'astropy.coordinates.EarthLocation or (2) '
                            'latitude and longitude in degrees as '
                            'accepted by astropy.coordinates.Latitude and '
                            'astropy.coordinates.Latitude.')

        # Accept various timezone inputs, default to UTC
        if isinstance(timezone, datetime.tzinfo):
            self.timezone = timezone
        elif isinstance(timezone, str):
            self.timezone = pytz.timezone(timezone)
        else:
            raise TypeError('timezone keyword should be a string, or an '
                            'instance of datetime.tzinfo')

    @property
    def longitude(self):
        """The longitude of the observing location, derived from the location."""
        return self.location.lon

    @property
    def latitude(self):
        """The latitude of the observing location, derived from the location."""
        return self.location.lat

    @property
    def elevation(self):
        """The elevation of the observing location with respect to sea level."""
        return self.location.height

    def __repr__(self):
        """
        String representation of the `~astroplan.Observer` object.

        Examples
        --------

        >>> from astroplan import Observer
        >>> keck = Observer.at_site("Keck", timezone="US/Hawaii")
        >>> print(keck)                                    # doctest: +FLOAT_CMP
        <Observer: name='Keck',
            location (lon, lat, el)=(-155.478333333 deg, 19.8283333333 deg, 4160.0 m),
            timezone=<DstTzInfo 'US/Hawaii' LMT-1 day, 13:29:00 STD>>
        """
        class_name = self.__class__.__name__
        attr_names = ['name', 'location', 'timezone', 'pressure', 'temperature',
                      'relative_humidity']
        attr_values = [getattr(self, attr) for attr in attr_names]
        attributes_strings = []
        for name, value in zip(attr_names, attr_values):
            if value is not None:
                # Format location for easy readability
                if name == 'location':
                    formatted_loc = ["{} {}".format(i.value, i.unit)
                                     for i in value.to_geodetic()]
                    attributes_strings.append(
                        "{} (lon, lat, el)=({})".format(
                            name, ", ".join(formatted_loc)))
                else:
                    if name != 'name':
                        value = repr(value)
                    else:
                        value = "'{}'".format(value)
                    attributes_strings.append("{}={}".format(name, value))
        return "<{}: {}>".format(class_name, ",\n    ".join(attributes_strings))

    def _key(self):
        """
        Generate a tuple of the attributes that determine uniqueness of
        `~astroplan.Observer` objects.

        Returns
        -------
        key : tuple

        Examples
        --------

        >>> from astroplan import Observer
        >>> keck = Observer.at_site("Keck", timezone="US/Hawaii")
        >>> keck._key()
        ('Keck', None, None, None, <Longitude -155.47833333 deg>,
            <Latitude 19.82833333 deg>, <Quantity 4160. m>,
            <DstTzInfo 'US/Hawaii' LMT-1 day, 13:29:00 STD>)
        """

        return (self.name,
                self.pressure,
                self.temperature,
                self.relative_humidity,
                self.longitude,
                self.latitude,
                self.elevation,
                self.timezone,)

    def __hash__(self):
        """
        Hash the `~astroplan.Observer` object.

        Examples
        --------

        >>> from astroplan import Observer
        >>> keck = Observer.at_site("Keck", timezone="US/Hawaii")
        >>> hash(keck)
        -3872382927731250571
        """

        return hash(self._key())

    def __eq__(self, other):
        """
        Equality check for `~astroplan.Observer` objects.

        Examples
        --------

        >>> from astroplan import Observer
        >>> keck = Observer.at_site("Keck", timezone="US/Hawaii")
        >>> keck2 = Observer.at_site("Keck", timezone="US/Hawaii")
        >>> keck == keck2
        True
        """

        if isinstance(other, Observer):
            return self._key() == other._key()
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        Inequality check for `~astroplan.Observer` objects.

        Examples
        --------

        >>> from astroplan import Observer
        >>> keck = Observer.at_site("Keck", timezone="US/Hawaii")
        >>> kpno = Observer.at_site("KPNO", timezone="US/Arizona")
        >>> keck != kpno
        True
        """

        return not self.__eq__(other)

    @classmethod
    def at_site(cls, site_name, **kwargs):
        """
        Initialize an `~astroplan.observer.Observer` object with a site name.

        Extra keyword arguments are passed to the `~astroplan.Observer`
        constructor (see `~astroplan.Observer` for available keyword
        arguments).

        Parameters
        ----------
        site_name : str
            Observatory name, must be resolvable with
            `~astropy.coordinates.EarthLocation.get_site_names`.

        Returns
        -------
        `~astroplan.observer.Observer`
            Observer object.

        Examples
        --------
        Initialize an observer at Kitt Peak National Observatory:

        >>> from astroplan import Observer
        >>> import astropy.units as u
        >>> kpno_generic = Observer.at_site('kpno')
        >>> kpno_today = Observer.at_site('kpno', pressure=1*u.bar, temperature=0*u.deg_C)
        """
        name = kwargs.pop('name', site_name)
        if 'location' in kwargs:
            raise ValueError("Location kwarg should not be used if "
                             "initializing an Observer with Observer.at_site()")
        return cls(location=EarthLocation.of_site(site_name), name=name, **kwargs)

    def astropy_time_to_datetime(self, astropy_time):
        """
        Convert the `~astropy.time.Time` object ``astropy_time`` to a
        localized `~datetime.datetime` object.

        Timezones localized with `pytz`_.

        .. _pytz: https://pypi.python.org/pypi/pytz/

        Parameters
        ----------
        astropy_time : `~astropy.time.Time`
            Scalar or list-like.

        Returns
        -------
        `~datetime.datetime`
            Localized datetime, where the timezone of the datetime is
            set by the ``timezone`` keyword argument of the
            `~astroplan.Observer` constructor.

        Examples
        --------
        Convert an astropy time to a localized `~datetime.datetime`:

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> subaru = Observer.at_site("Subaru", timezone="US/Hawaii")
        >>> astropy_time = Time('1999-12-31 06:00:00')
        >>> print(subaru.astropy_time_to_datetime(astropy_time))
        1999-12-30 20:00:00-10:00
        """

        if not astropy_time.isscalar:
            return [self.astropy_time_to_datetime(t) for t in astropy_time]

        # Convert astropy.time.Time to a UTC localized datetime (aware)
        utc_datetime = pytz.utc.localize(astropy_time.utc.datetime)

        # Convert UTC to local timezone
        return self.timezone.normalize(utc_datetime)

    def datetime_to_astropy_time(self, date_time):
        """
        Convert the `~datetime.datetime` object ``date_time`` to a
        `~astropy.time.Time` object.

        Timezones localized with `pytz`_. If the ``date_time`` is naive, the
        implied timezone is the ``timezone`` structure of ``self``.

        Parameters
        ----------
        date_time : `~datetime.datetime` or list-like

        Returns
        -------
        `~astropy.time.Time`
            Astropy time object (no timezone information preserved).

        Examples
        --------
        Convert a localized `~datetime.datetime` to a `~astropy.time.Time`
        object. Non-localized datetimes are assumed to be UTC.
        <Time object: scale='utc' format='datetime' value=1999-12-31 06:00:00>

        >>> from astroplan import Observer
        >>> import datetime
        >>> import pytz
        >>> subaru = Observer.at_site("Subaru", timezone="US/Hawaii")
        >>> hi_date_time = datetime.datetime(2005, 6, 21, 20, 0, 0, 0)
        >>> subaru.datetime_to_astropy_time(hi_date_time)
        <Time object: scale='utc' format='datetime' value=2005-06-22 06:00:00>
        >>> utc_date_time = datetime.datetime(2005, 6, 22, 6, 0, 0, 0,
        ...                                   tzinfo=pytz.timezone("UTC"))
        >>> subaru.datetime_to_astropy_time(utc_date_time)
        <Time object: scale='utc' format='datetime' value=2005-06-22 06:00:00>
        """

        if hasattr(date_time, '__iter__'):
            return Time([self.datetime_to_astropy_time(t) for t in date_time])

        # For timezone-naive datetimes, assign local timezone
        if date_time.tzinfo is None:
            date_time = self.timezone.localize(date_time)

        return Time(date_time, location=self.location)

    def _is_broadcastable(self, shp1, shp2):
        """Test if two shape tuples are broadcastable"""
        if shp1 == shp2:
            return True
        for a, b in zip(shp1[::-1], shp2[::-1]):
            if a == 1 or b == 1 or a == b:
                pass
            else:
                return False
        return True

    def _preprocess_inputs(self, time, target=None, grid_times_targets=False):
        """
        Preprocess time and target inputs

        This routine takes the inputs for time and target and attempts to
        return a single `~astropy.time.Time` and `~astropy.coordinates.SkyCoord`
        for each argument, which may be non-scalar if necessary.

        time : `~astropy.time.Time` or other (see below)
            The time(s) to use in the calculation. It can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astroplan.FixedTarget`, `~astropy.coordinates.SkyCoord`, or list
            The target(s) to use in the calculation.

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed onto
            the end, so that calculations with M targets and N times will
            return an (M, N) shaped result. Otherwise, we rely on broadcasting
            the shapes together using standard numpy rules. Useful for grid
            searches for rise/set times etc.
        """
        # make sure we have a non-scalar time
        if not isinstance(time, Time):
            time = Time(time)

        if target is None:
            return time, None

        # convert any kind of target argument to non-scalar SkyCoord
        target = get_skycoord(target)
        if grid_times_targets:
            if target.isscalar:
                # ensure we have a (1, 1) shape coord
                target = SkyCoord(np.tile(target, 1))[:, np.newaxis]
            else:
                while target.ndim <= time.ndim:
                    target = target[:, np.newaxis]

        elif not self._is_broadcastable(target.shape, time.shape):
            raise ValueError('Time and Target arguments cannot be broadcast '
                             'against each other with shapes {} and {}'
                             .format(time.shape, target.shape))
        return time, target

    def altaz(self, time, target=None, obswl=None, grid_times_targets=False):
        """
        Get an `~astropy.coordinates.AltAz` frame or coordinate.

        If ``target`` is None, generates an altitude/azimuth frame. Otherwise,
        calculates the transformation to that frame for the requested ``target``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            The time at which the observation is taking place. Will be used as
            the ``obstime`` attribute in the resulting frame or coordinate. This
            will be passed in as the first argument to the `~astropy.time.Time`
            initializer, so it can be anything that `~astropy.time.Time` will
            accept (including a `~astropy.time.Time` object)

        target : `~astroplan.FixedTarget`, `~astropy.coordinates.SkyCoord`, or list (optional)
            Celestial object(s) of interest. If ``target`` is `None`, returns
            the `~astropy.coordinates.AltAz` frame without coordinates.

        obswl : `~astropy.units.Quantity` (optional)
            Wavelength of the observation used in the calculation.

        grid_times_targets: bool (optional)
            If True, the target object will have extra dimensions packed
            onto the end, so that calculations with M targets and N times
            will return an (M, N) shaped result. Otherwise, we rely on
            broadcasting the shapes together using standard numpy
            rules. Useful for grid searches for rise/set times etc.

        Returns
        -------
        `~astropy.coordinates.AltAz`
            If ``target`` is `None`, returns `~astropy.coordinates.AltAz`
            frame. If ``target`` is not `None`, returns the ``target``
            transformed to the `~astropy.coordinates.AltAz` frame.

        Examples
        --------
        Create an instance of the `~astropy.coordinates.AltAz` frame for an
        observer at Apache Point Observatory at a particular time:

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> from astropy.coordinates import SkyCoord
        >>> apo = Observer.at_site("APO")
        >>> time = Time('2001-02-03 04:05:06')
        >>> target = SkyCoord(0*u.deg, 0*u.deg)
        >>> altaz_frame = apo.altaz(time)

        Now transform the target's coordinates to the alt/az frame:

        >>> target_altaz = target.transform_to(altaz_frame) # doctest: +SKIP

        Alternatively, construct an alt/az frame and transform the target to
        that frame all in one step:

        >>> target_altaz = apo.altaz(time, target) # doctest: +SKIP
        """
        if target is not None:
            if target is MoonFlag:
                target = get_body("moon", time, location=self.location)
            elif target is SunFlag:
                target = get_sun(time)

            time, target = self._preprocess_inputs(time, target, grid_times_targets)

        altaz_frame = AltAz(location=self.location, obstime=time,
                            pressure=self.pressure, obswl=obswl,
                            temperature=self.temperature,
                            relative_humidity=self.relative_humidity)
        if target is None:
            # Return just the frame
            return altaz_frame
        else:
            return target.transform_to(altaz_frame)

    def parallactic_angle(self, time, target, grid_times_targets=False,
                          kind='mean', model=None):
        """
        Calculate the parallactic angle.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Observation time.

        target : `~astroplan.FixedTarget` or `~astropy.coordinates.SkyCoord` or list
            Target celestial object(s).

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed onto
            the end, so that calculations with M targets and N times will
            return an (M, N) shaped result. Otherwise, we rely on broadcasting
            the shapes together using standard numpy rules.

        kind : str
            Argument to the `~astropy.time.Time.sidereal_time` function, which may
            be either ``'mean'`` or ``'apparent'``, i.e., accounting for precession only,
            or also for nutation.

        model : str or None; optional
            Precession (and nutation) model to use in the call to
            `~astropy.time.Time.sidereal_time`, see docs for that method for details.

        Returns
        -------
        `~astropy.coordinates.Angle`
            Parallactic angle.

        Notes
        -----
        The parallactic angle is the angle between the great circle that
        intersects a celestial object and the zenith, and the object's hour
        circle [1]_.

        .. [1] https://en.wikipedia.org/wiki/Parallactic_angle

        """
        time, coordinate = self._preprocess_inputs(time, target, grid_times_targets)

        # Eqn (14.1) of Meeus' Astronomical Algorithms
        LST = time.sidereal_time(kind, longitude=self.location.lon, model=model)
        H = (LST - coordinate.ra).radian
        q = np.arctan2(np.sin(H),
                       (np.tan(self.location.lat.radian) *
                        np.cos(coordinate.dec.radian) -
                        np.sin(coordinate.dec.radian)*np.cos(H)))*u.rad

        return Angle(q)

    # Sun-related methods.
    @u.quantity_input(horizon=u.deg)
    def _horiz_cross(self, t, alt, rise_set, horizon=0*u.degree):
        """
        Find time ``t`` when values in array ``a`` go from
        negative to positive or positive to negative (exclude endpoints)

        ``return_limits`` will return nearest times to zero-crossing.

        Parameters
        ----------
        t : `~astropy.time.Time`
            Grid of N times, any shape. Search grid along first axis, e.g (N, ...)
        alt : `~astropy.units.Quantity`
            Grid of altitudes
            Depending on broadcasting we either have ndim >=3 and
            M targets along first axis, e.g (M, N, ...), or
            ndim = 2 and targets/times in last axis
        rise_set : {"rising",  "setting"}
            Calculate either rising or setting across the horizon
        horizon : float
            Number of degrees above/below actual horizon to use
            for calculating rise/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        Returns
        -------
        Returns the lower and upper limits on the time and altitudes
        of the horizon crossing. The altitude limits have shape (M, ...) and the
        time limits have shape (...). These arrays aresuitable for interpolation
        to find the horizon crossing time.
        """
        # handle different cases by enforcing standard shapes on
        # the altitude grid
        finesse_time_indexes = False
        if alt.ndim == 1:
            raise ValueError('Must supply more at least a 2D grid of altitudes')
        elif alt.ndim == 2:
            # TODO: this test for ndim=2 doesn't work. if times is e.g (2,5)
            # then alt will have ndim=3, but shape (100, 2, 5) so grid
            # is in first index...
            ntargets = alt.shape[1]
            ngrid = alt.shape[0]
            unit = alt.unit
            alt = np.broadcast_to(alt, (ntargets, ngrid, ntargets)).T
            alt = alt*unit
            extra_dimension_added = True
            if t.shape[1] == 1:
                finesse_time_indexes = True
        else:
            extra_dimension_added = False
        output_shape = (alt.shape[0],) + alt.shape[2:]

        if rise_set == 'rising':
            # Find index where altitude goes from below to above horizon
            condition = ((alt[:, :-1, ...] < horizon) *
                         (alt[:, 1:, ...] > horizon))
        elif rise_set == 'setting':
            # Find index where altitude goes from above to below horizon
            condition = ((alt[:, :-1, ...] > horizon) *
                         (alt[:, 1:, ...] < horizon))

        noncrossing_indices = np.sum(condition, axis=1, dtype=np.intp) < 1
        alt_lims1 = u.Quantity(np.zeros(output_shape), unit=u.deg)
        alt_lims2 = u.Quantity(np.zeros(output_shape), unit=u.deg)
        jd_lims1 = np.zeros(output_shape)
        jd_lims2 = np.zeros(output_shape)
        if np.any(noncrossing_indices):
            for target_index in set(np.where(noncrossing_indices)[0]):
                warnmsg = ('Target with index {} does not cross horizon={} '
                           'within 24 hours'.format(target_index, horizon))
                if (alt[target_index, ...] > horizon).all():
                    warnings.warn(warnmsg, TargetAlwaysUpWarning)
                else:
                    warnings.warn(warnmsg, TargetNeverUpWarning)

            alt_lims1[np.nonzero(noncrossing_indices)] = np.nan
            alt_lims2[np.nonzero(noncrossing_indices)] = np.nan
            jd_lims1[np.nonzero(noncrossing_indices)] = np.nan
            jd_lims2[np.nonzero(noncrossing_indices)] = np.nan

        before_indices = np.array(np.nonzero(condition))
        # we want to add an vector like (0, 1, ...) to get after indices
        after_indices = before_indices.copy()
        after_indices[1, :] += 1

        al1 = alt[tuple(before_indices)]
        al2 = alt[tuple(after_indices)]
        # slice the time in the same way, but delete the object index
        before_time_index_tuple = np.delete(before_indices, 0, 0)
        after_time_index_tuple = np.delete(after_indices, 0, 0)
        if finesse_time_indexes:
            before_time_index_tuple[1:] = 0
            after_time_index_tuple[1:] = 0
        tl1 = t[tuple(before_time_index_tuple)]
        tl2 = t[tuple(after_time_index_tuple)]

        alt_lims1[tuple(np.delete(before_indices, 1, 0))] = al1
        alt_lims2[tuple(np.delete(before_indices, 1, 0))] = al2
        jd_lims1[tuple(np.delete(before_indices, 1, 0))] = tl1.utc.jd
        jd_lims2[tuple(np.delete(before_indices, 1, 0))] = tl2.utc.jd

        if extra_dimension_added:
            return (alt_lims1.diagonal(), alt_lims2.diagonal(),
                    jd_lims1.diagonal(), jd_lims2.diagonal())
        else:
            return alt_lims1, alt_lims2, jd_lims1, jd_lims2

    @u.quantity_input(horizon=u.deg)
    def _two_point_interp(self, jd_before, jd_after,
                          alt_before, alt_after, horizon=0*u.deg):
        """
        Do linear interpolation between two ``altitudes`` at
        two ``times`` to determine the time where the altitude
        goes through zero.

        Parameters
        ----------
        jd_before : `float`
            JD(UTC) before crossing event

        jd_after : `float`
            JD(UTC) after crossing event

        alt_before : `~astropy.units.Quantity`
            altitude before crossing event

        alt_after : `~astropy.units.Quantity`
            altitude after crossing event

        horizon : `~astropy.units.Quantity`
            Solve for the time when the altitude is equal to
            reference_alt.

        Returns
        -------
        t : `~astropy.time.Time`
            Time when target crosses the horizon

        """
        # Approximate the horizon-crossing time:
        slope = (alt_after-alt_before) / ((jd_after - jd_before) * u.d)
        crossing_jd = (jd_after * u.d - ((alt_after - horizon) / slope))

        # Determine whether or not there are NaNs in the crossing_jd array which
        # represent computations where no horizon crossing was found:
        times = Time(_process_nans_in_jds(crossing_jd), format='jd')

        return np.squeeze(times)

    def _altitude_trig(self, LST, target, grid_times_targets=False):
        """
        Calculate the altitude of ``target`` at local sidereal times ``LST``.

        This method provides a factor of ~3 speed up over calling `altaz`, and
        inherently does *not* take the atmosphere into account.

        Parameters
        ----------
        LST : `~astropy.time.Time`
            Local sidereal times (array)

        target : {`~astropy.coordinates.SkyCoord`, `FixedTarget`} or similar
            Target celestial object's coordinates.

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed onto
            the end, so that calculations with M targets and N times will
            return an (M, N) shaped result. Otherwise, we rely on broadcasting
            the shapes together using standard numpy rules.

        Returns
        -------
        alt : `~astropy.unit.Quantity`
            Array of altitudes
        """
        LST, target = self._preprocess_inputs(LST, target, grid_times_targets)
        alt = np.arcsin(np.sin(self.location.lat.radian) *
                        np.sin(target.dec) +
                        np.cos(self.location.lat.radian) *
                        np.cos(target.dec) *
                        np.cos(LST.radian - target.ra.radian))
        return alt

    def _calc_riseset(self, time, target, prev_next, rise_set, horizon,
                      n_grid_points=150, grid_times_targets=False):
        """
        Time at next rise/set of ``target``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument to
            the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astropy.coordinates.SkyCoord`
            Position of target or multiple positions of that target
            at multiple times (if target moves, like the Sun)

        prev_next : str - either 'previous' or 'next'
            Test next rise/set or previous rise/set

        rise_set : str - either 'rising' or 'setting'
            Compute prev/next rise or prev/next set

        horizon : `~astropy.units.Quantity`
            Degrees above/below actual horizon to use
            for calculating rise/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        n_grid_points : int (optional)
            Number of altitudes to compute when searching for
            rise or set.

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed onto
            the end, so that calculations with M targets and N times will
            return an (M, N) shaped result. Otherwise, we rely on broadcasting
            the shapes together using standard numpy rules.

        Returns
        -------
        ret1 : `~astropy.time.Time`
            Time of rise/set
        """
        if not isinstance(time, Time):
            time = Time(time)

        if prev_next == 'next':
            start = 0
            end = (1 + (target.approx_sidereal_drift.to(u.day).value
                        if hasattr(target, 'approx_sidereal_drift') else 0))
        else:
            start = (-1 - (target.approx_sidereal_drift.to(u.day).value
                           if hasattr(target, 'approx_sidereal_drift') else 0))
            end = 0

        times = _generate_24hr_grid(time, start, end, n_grid_points)

        altaz = self.altaz(times, target,
                           grid_times_targets=grid_times_targets)

        altitudes = altaz.alt

        al1, al2, jd1, jd2 = self._horiz_cross(times, altitudes, rise_set,
                                               horizon)
        return self._two_point_interp(jd1, jd2, al1, al2,
                                      horizon=horizon)

    def _calc_transit(self, time, target, prev_next, antitransit=False,
                      n_grid_points=150, grid_times_targets=False):
        """
        Time at next transit of the meridian of `target`.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument to
            the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astropy.coordinates.SkyCoord`
            Position of target or multiple positions of that target
            at multiple times (if target moves, like the Sun)

        prev_next : str - either 'previous' or 'next'
            Test next rise/set or previous rise/set

        antitransit : bool
            Toggle compute antitransit (below horizon, equivalent to midnight
            for the Sun)

        n_grid_points : int (optional)
            Number of altitudes to compute when searching for
            rise or set.

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed onto
            the end, so that calculations with M targets and N times will
            return an (M, N) shaped result. Otherwise, we rely on broadcasting
            the shapes together using standard numpy rules.

        Returns
        -------
        ret1 : `~astropy.time.Time`
            Time of transit/antitransit
        """
        # TODO FIX BROADCASTING HERE
        if not isinstance(time, Time):
            time = Time(time)

        if prev_next == 'next':
            times = _generate_24hr_grid(time, 0, 1, n_grid_points,
                                        for_deriv=True)
        else:
            times = _generate_24hr_grid(time, -1, 0, n_grid_points,
                                        for_deriv=True)

        # The derivative of the altitude with respect to time is increasing
        # from negative to positive values at the anti-transit of the meridian
        if antitransit:
            rise_set = 'rising'
        else:
            rise_set = 'setting'

        altaz = self.altaz(times, target,
                           grid_times_targets=grid_times_targets)

        altitudes = altaz.alt
        if altitudes.ndim > 2:
            # shape is (M, N, ...) where M is targets and N is grid
            d_altitudes = altitudes.diff(axis=1)
        else:
            # shape is (N, M) where M is targets and N is grid
            d_altitudes = altitudes.diff(axis=0)

        dt = Time((times.jd[1:] + times.jd[:-1])/2, format='jd')

        horizon = 0*u.degree  # Find when derivative passes through zero
        al1, al2, jd1, jd2 = self._horiz_cross(dt, d_altitudes,
                                               rise_set, horizon)
        return self._two_point_interp(jd1, jd2, al1, al2,
                                      horizon=horizon)

    def _determine_which_event(self, function, args_dict):
        """
        Run through the next/previous/nearest permutations of the solutions
        to `function(time, ...)`, and return the previous/next/nearest one
        specified by the args stored in args_dict.
        """
        time = args_dict.pop('time', None)
        target = args_dict.pop('target', None)
        which = args_dict.pop('which', None)
        horizon = args_dict.pop('horizon', None)
        rise_set = args_dict.pop('rise_set', None)
        antitransit = args_dict.pop('antitransit', None)
        grid_times_targets = args_dict.pop('grid_times_targets', False)
        n_grid_points = args_dict.pop('n_grid_points', 150)

        # Assemble arguments for function, depending on the function.
        if function == self._calc_riseset:
            def event_function(w):
                return function(time, target, w, rise_set, horizon,
                                grid_times_targets=grid_times_targets,
                                n_grid_points=n_grid_points)
        elif function == self._calc_transit:
            def event_function(w):
                return function(time, target, w, antitransit=antitransit,
                                grid_times_targets=grid_times_targets,
                                n_grid_points=n_grid_points)
        else:
            raise ValueError('Function {} not supported in '
                             '_determine_which_event.'.format(function))

        if not isinstance(time, Time):
            time = Time(time)

        if which == 'next' or which == 'nearest':
            next_event = event_function('next')
            if which == 'next':
                return next_event

        if which == 'previous' or which == 'nearest':
            previous_event = event_function('previous')
            if which == 'previous':
                return previous_event

        if which == 'nearest':
            try:
                mask = abs(time - previous_event) < abs(time - next_event)
                ma = np.where(mask, previous_event.utc.jd, next_event.utc.jd)
            except TypeError:
                # encountered if time is scalar & nan
                ma = next_event.utc.jd if not next_event.isscalar else [next_event.utc.jd]
            return Time(_process_nans_in_jds(ma), format='jd', scale='utc')

        raise ValueError('"which" kwarg must be "next", "previous" or '
                         '"nearest".')

    @u.quantity_input(horizon=u.deg)
    def target_rise_time(self, time, target, which='nearest',
                         horizon=0*u.degree, grid_times_targets=False, n_grid_points=150):
        """
        Calculate rise time.

        Compute time of the next/previous/nearest rise of the ``target``
        object, where "rise" is defined as the time when the ``target``
        transitions from altitudes below the ``horizon`` to above the
        ``horizon``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`, or list
            Target celestial object(s)

        which : {'next', 'previous', 'nearest'}
            Choose which sunrise relative to the present ``time`` would you
            like to calculate

        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use
            for calculating rise/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed
            onto the end, so that calculations with M targets and N times
            will return an (M, N) shaped result. Otherwise, we rely on
            broadcasting the shapes together using standard numpy rules.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields rise time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Rise time of target

        Examples
        --------
        Calculate the rise time of Rigel at Keck Observatory:

        >>> from astroplan import Observer, FixedTarget
        >>> from astropy.time import Time
        >>> time = Time("2001-02-03 04:05:06")
        >>> target = FixedTarget.from_name("Rigel")
        >>> keck = Observer.at_site("Keck")
        >>> rigel_rise_time = keck.target_rise_time(time, target, which="next") # doctest: +SKIP
        >>> print("ISO: {0.iso}, JD: {0.jd}".format(rigel_rise_time)) # doctest: +SKIP
        ISO: 2001-02-04 00:51:23.330, JD: 2451944.53569
        """
        return self._determine_which_event(self._calc_riseset,
                                           dict(time=time, target=target,
                                                which=which, rise_set='rising',
                                                horizon=horizon,
                                                n_grid_points=n_grid_points,
                                                grid_times_targets=grid_times_targets))

    @u.quantity_input(horizon=u.deg)
    def target_set_time(self, time, target, which='nearest', horizon=0*u.degree,
                        grid_times_targets=False, n_grid_points=150):
        """
        Calculate set time.

        Compute time of the next/previous/nearest set of ``target``, where
        "set" is defined as when the ``target`` transitions from altitudes
        above ``horizon`` to below ``horizon``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`, or list
            Target celestial object(s)

        which : {'next', 'previous', 'nearest'}
            Choose which sunset relative to the present ``time`` would you
            like to calculate

        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use
            for calculating rise/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed
            onto the end, so that calculations with M targets and N times
            will return an (M, N) shaped result. Otherwise, we rely on
            broadcasting the shapes together using standard numpy rules.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields set time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Set time of target.

        Examples
        --------
        Calculate the set time of Rigel at Keck Observatory:

        >>> from astroplan import Observer, FixedTarget
        >>> from astropy.time import Time
        >>> time = Time("2001-02-03 04:05:06")
        >>> target = FixedTarget.from_name("Rigel")
        >>> keck = Observer.at_site("Keck")
        >>> rigel_set_time = keck.target_set_time(time, target, which="next") # doctest: +SKIP
        >>> print("ISO: {0.iso}, JD: {0.jd}".format(rigel_set_time)) # doctest: +SKIP
        ISO: 2001-02-03 12:29:34.768, JD: 2451944.02054
        """
        return self._determine_which_event(self._calc_riseset,
                                           dict(time=time, target=target,
                                                which=which,
                                                rise_set='setting',
                                                horizon=horizon,
                                                n_grid_points=n_grid_points,
                                                grid_times_targets=grid_times_targets))

    def target_meridian_transit_time(self, time, target, which='nearest',
                                     grid_times_targets=False, n_grid_points=150):
        """
        Calculate time at the transit of the meridian.

        Compute time of the next/previous/nearest transit of the ``target``
        object.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`, or list
            Target celestial object(s)

        which : {'next', 'previous', 'nearest'}
            Choose which sunrise relative to the present ``time`` would you
            like to calculate

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed
            onto the end, so that calculations with M targets and N times
            will return an (M, N) shaped result. Otherwise, we rely on
            broadcasting the shapes together using standard numpy rules.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields rise time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Transit time of target

        Examples
        --------
        Calculate the meridian transit time of Rigel at Keck Observatory:

        >>> from astroplan import Observer, FixedTarget
        >>> from astropy.time import Time
        >>> time = Time("2001-02-03 04:05:06")
        >>> target = FixedTarget.from_name("Rigel")
        >>> keck = Observer.at_site("Keck")
        >>> rigel_transit_time = keck.target_meridian_transit_time(time, target,
        ...                                                        which="next") # doctest: +SKIP
        >>> print("ISO: {0.iso}, JD: {0.jd}".format(rigel_transit_time)) # doctest: +SKIP
        ISO: 2001-02-03 06:42:26.863, JD: 2451943.77948
        """
        return self._determine_which_event(self._calc_transit,
                                           dict(time=time, target=target,
                                                which=which,
                                                n_grid_points=n_grid_points,
                                                rise_set='setting',
                                                grid_times_targets=grid_times_targets))

    def target_meridian_antitransit_time(self, time, target, which='nearest',
                                         grid_times_targets=False, n_grid_points=150):
        """
        Calculate time at the antitransit of the meridian.

        Compute time of the next/previous/nearest antitransit of the ``target``
        object.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`, or list
            Target celestial object(s)

        which : {'next', 'previous', 'nearest'}
            Choose which sunrise relative to the present ``time`` would you
            like to calculate

        grid_times_targets : bool
            If True, the target object will have extra dimensions packed onto the end,
            so that calculations with M targets and N times will return an (M, N)
            shaped result. Otherwise, we rely on broadcasting the shapes together
            using standard numpy rules.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields rise time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Antitransit time of target

        Examples
        --------
        Calculate the meridian anti-transit time of Rigel at Keck Observatory:

        >>> from astroplan import Observer, FixedTarget
        >>> from astropy.time import Time
        >>> time = Time("2001-02-03 04:05:06")
        >>> target = FixedTarget.from_name("Rigel")
        >>> keck = Observer.at_site("Keck")
        >>> rigel_antitransit_time = keck.target_meridian_antitransit_time(
        ...     time, target, which="next") # doctest: +SKIP
        >>> print("ISO: {0.iso}, JD: {0.jd}".format(rigel_antitransit_time)) # doctest: +SKIP
        ISO: 2001-02-03 18:40:29.761, JD: 2451944.27812

        """
        return self._determine_which_event(self._calc_transit,
                                           dict(time=time, target=target,
                                                which=which, antitransit=True,
                                                rise_set='setting',
                                                n_grid_points=n_grid_points,
                                                grid_times_targets=grid_times_targets))

    @u.quantity_input(horizon=u.deg)
    def sun_rise_time(self, time, which='nearest', horizon=0*u.degree, n_grid_points=150):
        """
        Time of sunrise.

        Compute time of the next/previous/nearest sunrise, where
        sunrise is defined as when the Sun transitions from altitudes
        below ``horizon`` to above ``horizon``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which sunrise relative to the present ``time`` would you
            like to calculate.

        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use
            for calculating rise/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields rise time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time of sunrise

        Examples
        --------
        Calculate the time of the previous sunrise at Apache Point Observatory:

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> apo = Observer.at_site("APO")
        >>> time = Time('2001-02-03 04:05:06')
        >>> sun_rise = apo.sun_rise_time(time, which="previous") # doctest: +SKIP
        >>> print("ISO: {0.iso}, JD: {0.jd}".format(sun_rise)) # doctest: +SKIP
        ISO: 2001-02-02 14:02:50.554, JD: 2451943.08531
        """
        return self.target_rise_time(time, get_sun(time), which, horizon,
                                     n_grid_points=n_grid_points)

    @u.quantity_input(horizon=u.deg)
    def sun_set_time(self, time, which='nearest', horizon=0*u.degree, n_grid_points=150):
        """
        Time of sunset.

        Compute time of the next/previous/nearest sunset, where
        sunset is defined as when the Sun transitions from altitudes
        below ``horizon`` to above ``horizon``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which sunset relative to the present ``time`` would you
            like to calculate

        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use
            for calculating rise/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields set time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time of sunset

        Examples
        --------
        Calculate the time of the next sunset at Apache Point Observatory:

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> apo = Observer.at_site("APO")
        >>> time = Time('2001-02-03 04:05:06')
        >>> sun_set = apo.sun_set_time(time, which="next") # doctest: +SKIP
        >>> print("ISO: {0.iso}, JD: {0.jd}".format(sun_set)) # doctest: +SKIP
        ISO: 2001-02-04 00:35:42.102, JD: 2451944.52479
        """
        return self.target_set_time(time, get_sun(time), which, horizon,
                                    n_grid_points=n_grid_points)

    def noon(self, time, which='nearest', n_grid_points=150):
        """
        Time at solar noon.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which noon relative to the present ``time`` would you
            like to calculate

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields noon time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time at solar noon
        """
        return self.target_meridian_transit_time(time, get_sun(time), which,
                                                 n_grid_points=n_grid_points)

    def midnight(self, time, which='nearest', n_grid_points=150):
        """
        Time at solar midnight.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which noon relative to the present ``time`` would you
            like to calculate

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields midnight time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time at solar midnight
        """
        return self.target_meridian_antitransit_time(time, get_sun(time), which,
                                                     n_grid_points=n_grid_points)

    # Twilight convenience functions

    def twilight_evening_astronomical(self, time, which='nearest', n_grid_points=150):
        """
        Time at evening astronomical (-18 degree) twilight.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which twilight relative to the present ``time`` would you
            like to calculate. Default is nearest.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields twilight time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time of twilight
        """
        return self.sun_set_time(time, which, horizon=-18*u.degree,
                                 n_grid_points=n_grid_points)

    def twilight_evening_nautical(self, time, which='nearest', n_grid_points=150):
        """
        Time at evening nautical (-12 degree) twilight.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which twilight relative to the present ``time`` would you
            like to calculate. Default is nearest.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields twilight time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time of twilight
        """
        return self.sun_set_time(time, which, horizon=-12*u.degree,
                                 n_grid_points=n_grid_points)

    def twilight_evening_civil(self, time, which='nearest', n_grid_points=150):
        """
        Time at evening civil (-6 degree) twilight.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which twilight relative to the present ``time`` would you
            like to calculate. Default is nearest.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields twilight time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time of twilight
        """
        return self.sun_set_time(time, which, horizon=-6*u.degree,
                                 n_grid_points=n_grid_points)

    def twilight_morning_astronomical(self, time, which='nearest', n_grid_points=150):
        """
        Time at morning astronomical (-18 degree) twilight.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which twilight relative to the present ``time`` would you
            like to calculate

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields twilight time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time of twilight
        """
        return self.sun_rise_time(time, which, horizon=-18*u.degree,
                                  n_grid_points=n_grid_points)

    def twilight_morning_nautical(self, time, which='nearest', n_grid_points=150):
        """
        Time at morning nautical (-12 degree) twilight.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which twilight relative to the present ``time`` would you
            like to calculate. Default is nearest.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields twilight time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time of twilight
        """
        return self.sun_rise_time(time, which, horizon=-12*u.degree,
                                  n_grid_points=n_grid_points)

    def twilight_morning_civil(self, time, which='nearest', n_grid_points=150):
        """
        Time at morning civil (-6 degree) twilight.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which twilight relative to the present ``time`` would you
            like to calculate. Default is nearest.

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields twilight time precisions better than one minute.

        Returns
        -------
        `~astropy.time.Time`
            Time of sunset
        """
        return self.sun_rise_time(time, which, horizon=-6*u.degree,
                                  n_grid_points=n_grid_points)

    # Moon-related methods.

    def moon_rise_time(self, time, which='nearest', horizon=0*u.deg, n_grid_points=150):
        """
        Returns the local moon rise time.

        Compute time of the next/previous/nearest moon rise, where
        moon rise is defined as the time when the moon transitions from
        altitudes below ``horizon`` to above ``horizon``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which moon rise relative to the present ``time`` would you
            like to calculate.

        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use
            for calculating rise/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields rise time precisions better than one minute.
        """
        return self.target_rise_time(time, MoonFlag, which, horizon,
                                     n_grid_points=n_grid_points)

    def moon_set_time(self, time, which='nearest', horizon=0*u.deg, n_grid_points=150):
        """
        Returns the local moon set time.

        Compute time of the next/previous/nearest moon set, where
        moon set is defined as the time when the moon transitions from
        altitudes below ``horizon`` to above ``horizon``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        which : {'next', 'previous', 'nearest'}
            Choose which moon set relative to the present ``time`` would you
            like to calculate.

        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use
            for calculating set/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        n_grid_points : int (optional)
            The number of grid points on which to search for the horizon
            crossings of the target over a 24 hour period, default is 150 which
            yields set time precisions better than one minute.
        """
        return self.target_set_time(time, MoonFlag, which, horizon,
                                    n_grid_points=n_grid_points)

    def moon_illumination(self, time):
        """
        Calculate the illuminated fraction of the moon.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            This will be passed in as the first argument to
            the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object).

        Returns
        -------
        float
            Fraction of lunar surface illuminated

        Examples
        --------
        How much of the lunar surface is illuminated at 2015-08-29 18:35 UTC,
        which we happen to know is the time of a full moon?

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> apo = Observer.at_site("APO")
        >>> time = Time("2015-08-29 18:35")
        >>> apo.moon_illumination(time) # doctest: +SKIP
        array([ 0.99972487])
        """
        if not isinstance(time, Time):
            time = Time(time)

        return moon_illumination(time)

    def moon_phase(self, time=None):
        """
        Calculate lunar orbital phase.

        For example, phase=pi is "new", phase=0 is "full".

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            This will be passed in as the first argument to
            the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object).

        Returns
        -------
        moon_phase_angle : float
            Orbital phase angle of the moon where pi corresponds to new moon,
            zero corresponds to full moon.

        Examples
        --------
        Calculate the phase of the moon at 2015-08-29 18:35 UTC. Near zero
        radians corresponds to a nearly full moon.

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> apo = Observer.at_site('APO')
        >>> time = Time('2015-08-29 18:35')
        >>> apo.moon_phase(time) # doctest: +SKIP
        <Quantity [ 0.03317537] rad>
        """
        if time is not None and not isinstance(time, Time):
            time = Time(time)

        return moon_phase_angle(time)

    def moon_altaz(self, time, ephemeris=None):
        """
        Returns the position of the moon in alt/az.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            This will be passed in as the first argument to
            the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object).

        ephemeris : str, optional
            Ephemeris to use.  If not given, use the one set with
            ``astropy.coordinates.solar_system_ephemeris.set`` (which is
            set to 'builtin' by default).


        Returns
        -------
        altaz : `~astropy.coordinates.SkyCoord`
            Position of the moon transformed to altitude and azimuth

        Examples
        --------
        Calculate the altitude and azimuth of the moon at Apache Point
        Observatory:

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> apo = Observer.at_site("APO")
        >>> time = Time("2015-08-29 18:35")
        >>> altaz_moon = apo.moon_altaz(time) # doctest: +SKIP
        >>> print("alt: {0.alt}, az: {0.az}".format(altaz_moon)) # doctest: +SKIP
        alt: -63.72706397691421 deg, az: 345.3640380598265 deg
        """
        if not isinstance(time, Time):
            time = Time(time)

        moon = get_body("moon", time, location=self.location, ephemeris=ephemeris)
        return self.altaz(time, moon)

    def sun_altaz(self, time):
        """
        Returns the position of the Sun in alt/az.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            This will be passed in as the first argument to
            the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object).

        ephemeris : str, optional
            Ephemeris to use.  If not given, use the one set with
            ``astropy.coordinates.solar_system_ephemeris.set`` (which is
            set to 'builtin' by default).


        Returns
        -------
        altaz : `~astropy.coordinates.SkyCoord`
            Position of the sun transformed to altitude and azimuth
        """
        if not isinstance(time, Time):
            time = Time(time)

        sun = get_sun(time)
        return self.altaz(time, sun)

    @u.quantity_input(horizon=u.deg)
    def target_is_up(self, time, target, horizon=0*u.degree,
                     return_altaz=False, grid_times_targets=False):
        """
        Is ``target`` above ``horizon`` at this ``time``?

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`, or list
            Target celestial object(s)

        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use
            for calculating rise/set times (i.e.,
            -6 deg horizon = civil twilight, etc.)

        return_altaz : bool (optional)
            Also return the '~astropy.coordinates.AltAz' coordinate.

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed
            onto the end, so that calculations with M targets and N times
            will return an (M, N) shaped result. Otherwise, we rely on
            broadcasting the shapes together using standard numpy rules.

        Returns
        -------
        observable : boolean or np.ndarray(bool)
            True if ``target`` is above ``horizon`` at ``time``, else False.

        Examples
        --------
        Are Aldebaran and Vega above the horizon at Apache Point Observatory
        at 2015-08-29 18:35 UTC?

        >>> from astroplan import Observer, FixedTarget
        >>> from astropy.time import Time
        >>> apo = Observer.at_site("APO")
        >>> time = Time("2015-08-29 18:35")
        >>> aldebaran = FixedTarget.from_name("Aldebaran")
        >>> vega = FixedTarget.from_name("Vega")
        >>> apo.target_is_up(time, aldebaran) # doctest: +SKIP
        True
        >>> apo.target_is_up(time, [aldebaran, vega]) # doctest: +SKIP
        array([ True, False], dtype=bool)
        """
        if not isinstance(time, Time):
            time = Time(time)

        altaz = self.altaz(time, target, grid_times_targets=grid_times_targets)
        observable = altaz.alt > horizon

        if altaz.isscalar:
            observable = bool(observable)

        if not return_altaz:
            return observable
        else:
            return observable, altaz

    @u.quantity_input(horizon=u.deg)
    def is_night(self, time, horizon=0*u.deg, obswl=None):
        """
        Is the Sun below ``horizon`` at ``time``?

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use
            for calculating day/night (i.e.,
            -6 deg horizon = civil twilight, etc.)

        obswl : `~astropy.units.Quantity` (optional)
            Wavelength of the observation used in the calculation

        Returns
        -------
        sun_below_horizon : bool or np.ndarray(bool)
            `True` if sun is below ``horizon`` at ``time``, else `False`.

        Examples
        --------
        Is it "nighttime" (i.e. is the Sun below ``horizon``) at Apache Point
        Observatory at 2015-08-29 18:35 UTC?

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> apo = Observer.at_site("APO")
        >>> time = Time("2015-08-29 18:35")
        >>> apo.is_night(time) # doctest: +SKIP
        False
        """
        if not isinstance(time, Time):
            time = Time(time)

        solar_altitude = self.altaz(time, target=get_sun(time), obswl=obswl).alt

        if solar_altitude.isscalar:
            return bool(solar_altitude < horizon)
        else:
            return solar_altitude < horizon

    def local_sidereal_time(self, time, kind='apparent', model=None):
        """
        Convert ``time`` to local sidereal time for observer.

        This is a thin wrapper around the `~astropy.time.Time.sidereal_time`
        method.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        kind : {'mean', 'apparent'} (optional)
            Passed to the ``kind`` argument of
            `~astropy.time.Time.sidereal_time`

        model : str or `None`; optional
            The precession/nutation model to assume - see
            `~astropy.time.Time.sidereal_time` for more details.

        Returns
        -------
        `~astropy.coordinates.Longitude`
            Local sidereal time.
        """
        if not isinstance(time, Time):
            time = Time(time)

        return time.sidereal_time(kind, longitude=self.location.lon,
                                  model=model)

    def target_hour_angle(self, time, target, grid_times_targets=False):
        """
        Calculate the local hour angle of ``target`` at ``time``.

        Parameters
        ----------
        time : `~astropy.time.Time` or other (see below)
            Time of observation. This will be passed in as the first argument
            to the `~astropy.time.Time` initializer, so it can be anything that
            `~astropy.time.Time` will accept (including a `~astropy.time.Time`
            object)

        target : `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`, or list
            Target celestial object(s)

        grid_times_targets: bool
            If True, the target object will have extra dimensions packed
            onto the end, so that calculations with M targets and N times
            will return an (M, N) shaped result. Otherwise, we rely on
            broadcasting the shapes together using standard numpy rules.

        Returns
        -------
        hour_angle : `~astropy.coordinates.Angle`
            The hour angle(s) of the target(s) at ``time``
        """
        time, target = self._preprocess_inputs(time, target, grid_times_targets)
        return Longitude(self.local_sidereal_time(time) - target.ra)

    @u.quantity_input(horizon=u.degree)
    def tonight(self, time=None, horizon=0 * u.degree, obswl=None):
        """
        Return a time range corresponding to the nearest night

        This will return a range of `~astropy.time.Time` corresponding to the
        beginning and ending of the night. If in the middle of a given night,
        return times from `~astropy.time.Time.now` until the nearest
        `~astroplan.Observer.sun_rise_time`

        Parameters
        ----------
        time : `~astropy.time.Time` (optional), default = `~astropy.time.Time.now`
            The start time for tonight, which is allowed to be arbitrary.
            See description above for behavior
        horizon : `~astropy.units.Quantity` (optional), default = zero degrees
            Degrees above/below actual horizon to use for calculating rise/set
            times (e.g., -6 deg horizon = civil twilight, etc.)
        obswl : `~astropy.units.Quantity` (optional)
            Wavelength of the observation used in the calculation

        Returns
        -------
        times : `~astropy.time.Time`
            A tuple of times corresponding to the start and end of current night
        """
        current_time = Time.now() if time is None else time
        night_mask = self.is_night(current_time, horizon=horizon, obswl=obswl)
        sun_set_time = self.sun_set_time(current_time, which='next',
                                         horizon=horizon)

        start_time = np.where(night_mask, current_time, sun_set_time)
        # np.where gives us a list of start Times - convert to Time object
        if not isinstance(start_time, Time):
            start_time = Time(start_time)
        end_time = self.sun_rise_time(start_time, which='next', horizon=horizon)

        return start_time, end_time
