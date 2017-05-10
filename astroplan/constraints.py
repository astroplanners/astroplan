# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Specify and constraints to determine which targets are observable for
an observer.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
from abc import ABCMeta, abstractmethod
import datetime
import warnings

# Third-party
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import (get_sun, get_moon, Angle, SkyCoord,
                                 AltAz)
from astropy import table
import numpy as np

# Package
from .moon import moon_illumination
from .utils import time_grid_from_range
from .target import FixedTarget, get_skycoord

__all__ = ["AltitudeConstraint", "AirmassConstraint", "AtNightConstraint",
           "is_observable", "is_always_observable", "time_grid_from_range",
           "SunSeparationConstraint", "MoonSeparationConstraint",
           "MoonIlluminationConstraint", "LocalTimeConstraint", "Constraint",
           "TimeConstraint", "observability_table", "months_observable",
           "max_best_rescale", "min_best_rescale"]


def _get_altaz(times, observer, targets, force_zero_pressure=False):
    """
    Calculate alt/az for ``target`` at times linearly spaced between
    the two times in ``time_range`` with grid spacing ``time_resolution``
    for ``observer``.

    Cache the result on the ``observer`` object.

    Parameters
    ----------
    times : `~astropy.time.Time`
        Array of times on which to test the constraint.
    targets : {list, `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`}
        Target or list of targets.
    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``.
    force_zero_pressure : bool
        Forcefully use 0 pressure.

    Returns
    -------
    altaz_dict : dict
        Dictionary containing two key-value pairs. (1) 'times' contains the
        times for the alt/az computations, (2) 'altaz' contains the
        corresponding alt/az coordinates at those times.
    """
    if not hasattr(observer, '_altaz_cache'):
        observer._altaz_cache = {}

    # convert times, targets to tuple for hashing
    aakey = (tuple(times.jd), tuple(targets))

    if aakey not in observer._altaz_cache:
        try:
            if force_zero_pressure:
                observer_old_pressure = observer.pressure
                observer.pressure = 0

            altaz = observer.altaz(times, targets)
            observer._altaz_cache[aakey] = dict(times=times,
                                                altaz=altaz)
        finally:
            if force_zero_pressure:
                observer.pressure = observer_old_pressure

    return observer._altaz_cache[aakey]


def _get_moon_data(times, observer, force_zero_pressure=False):
    """
    Calculate moon altitude az and illumination for an array of times for
    ``observer``.

    Cache the result on the ``observer`` object.

    Parameters
    ----------
    times : `~astropy.time.Time`
        Array of times on which to test the constraint.
    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``.
    force_zero_pressure : bool
        Forcefully use 0 pressure.

    Returns
    -------
    moon_dict : dict
        Dictionary containing three key-value pairs. (1) 'times' contains the
        times for the computations, (2) 'altaz' contains the
        corresponding alt/az coordinates at those times and (3) contains
        the moon illumination for those times.
    """
    if not hasattr(observer, '_moon_cache'):
        observer._moon_cache = {}

    # convert times to tuple for hashing
    aakey = (tuple(times.jd))

    if aakey not in observer._moon_cache:
        try:
            if force_zero_pressure:
                observer_old_pressure = observer.pressure
                observer.pressure = 0

            altaz = observer.moon_altaz(times)
            illumination = np.array(moon_illumination(times))
            observer._moon_cache[aakey] = dict(times=times,
                                               illum=illumination,
                                               altaz=altaz)
        finally:
            if force_zero_pressure:
                observer.pressure = observer_old_pressure

    return observer._moon_cache[aakey]


def _get_meridian_transit_times(times, observer, targets):
    """
    Calculate next meridian transit for an array of times for ``targets`` and
    ``observer``.

    Cache the result on the ``observer`` object.

    Parameters
    ----------
    times : `~astropy.time.Time`
        Array of times on which to test the constraint
    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``
    targets : {list, `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`}
        Target or list of targets

    Returns
    -------
    time_dict : dict
        Dictionary containing a key-value pair. 'times' contains the
        meridian_transit times.
    """
    if not hasattr(observer, '_meridian_transit_cache'):
        observer._meridian_transit_cache = {}

    # convert times to tuple for hashing
    aakey = (tuple(times.jd), tuple(targets))

    if aakey not in observer._meridian_transit_cache:
        meridian_transit_times = Time([observer.target_meridian_transit_time(
                                       time, target, which='next')
                                       for target in targets
                                       for time in times])
        observer._meridian_transit_cache[aakey] = dict(times=meridian_transit_times)

    return observer._meridian_transit_cache[aakey]


def limit(storage_name):
    """
    A property factory for constraint limits.

    This will ensure that any attempt to set the limit goes through
    a call to recast_limits.

    Examples
    ---------
    >>> class MyConstraint(Constraint):
    ...     # A demo constraint that does nothing
    ...     min = limit('min')
    ...     max = limit('max')  # set min and max properties to use limit property factory
    ...
    ...     def __init__(self, min=None, max=None):
    ...             self.min = min
    ...             self.max = max  # setting self.max goes through _recast_limits
    ...
    ...     def compute_constraint(self, times, observer, targets):
    ...             return np.atleast_2d(True)
    ...
    ...     @classmethod
    ...     def vectorize(cls, constraint_list):
    ...         min_vals = _get_limit_values(constraint_list, 'min')
    ...         max_vals = _get_limit_values(constraint_list, 'max')
    ...         return cls(min_vals, max_vals)
    ...
    >>>
    >>> cons = MyConstraint(max=2)
    >>> cons.max
    array([[2]])
    >>> cons.max = 3
    >>> cons.max.shape
    (1, 1)
    >>> cons.max = [3, 2]
    >>> cons.max
    array([[3],
           [2]])
    >>> cons.max.shape
    (2, 1)
    """
    def limit_getter(instance):
        return instance.__dict__[storage_name]

    def limit_setter(instance, value):
        instance.__dict__[storage_name] = instance._recast_limits(value)

    return property(limit_getter, limit_setter)


def _get_limit_values(constraint_list, limit_name):
    """
    A utility function to help extract a list of constraint limits from a list of constraints.
    This routine will raise an error if any of the constraints in the list are themselves
    vectorized.
    Parameters
    ----------
    constraint_list : list
        A list of `~astroplan.Constraint` objects.
    limit_name : str
        The name of the limit you wish to extract
    Returns
    -------
    limit_vals : list
        A list of the limit values.
    """
    vals = [getattr(c, limit_name) for c in constraint_list]
    vals_are_none = [val is None for val in vals]
    if np.any(vals_are_none):
        if np.all(vals_are_none):
            return None
        else:
            raise ValueError("Cannot vectorize constraints with a mixture of scalar and NoneType limits")
    else:
        # compare the shapes to (1,1) - use (1,1) as default for none
        shape_check = [getattr(val, 'shape', (1, 1)) == (1, 1) for val in vals]
        if not np.all(shape_check):
            msg = "Not all {} limits have scalar values".format(limit_name)
            raise ValueError(msg)
        return [val[0, 0] if val is not None else None for val in vals]


@abstractmethod
class Constraint(object):
    """
    Abstract class for objects defining observational constraints.
    """
    __metaclass__ = ABCMeta

    def __call__(self, observer, targets, times=None,
                 time_range=None, time_grid_resolution=0.5*u.hour):
        """
        Compute the constraint for this class

        Parameters
        ----------
        observer : `~astroplan.Observer`
            the observation location from which to apply the constraints
        targets : sequence of `~astroplan.Target`
            The targets on which to apply the constraints.
        times : `~astropy.time.Time`
            The times to compute the constraint.
            WHAT HAPPENS WHEN BOTH TIMES AND TIME_RANGE ARE SET?
        time_range : `~astropy.time.Time` (length = 2)
            Lower and upper bounds on time sequence.
        time_grid_resolution : `~astropy.units.quantity`
            Time-grid spacing

        Returns
        -------
        constraint_result : 2D array of float or bool
            The constraints, with targets along the first index and times along
            the second.
        """

        if times is None and time_range is not None:
            times = time_grid_from_range(time_range,
                                         time_resolution=time_grid_resolution)
        elif not isinstance(times, Time):
            times = Time(times)

        if times.isscalar:
            times = Time([times])

        if hasattr(targets, '__len__'):
            targets = [FixedTarget(coord=target) if isinstance(target, SkyCoord)
                       else target for target in targets]
        else:
            if isinstance(targets, SkyCoord):
                targets = FixedTarget(coord=targets)

        return self.compute_constraint(times, observer, targets)

    @abstractmethod
    def compute_constraint(self, times, observer, targets):
        """
        Actually do the real work of computing the constraint.  Subclasses
        override this.

        Parameters
        ----------
        times : `~astropy.time.Time`
            The times to compute the constraint
        observer : `~astroplan.Observer`
            the observaton location from which to apply the constraints
        targets : sequence of `~astroplan.Target`
            The targets on which to apply the constraints.

        Returns
        -------
        constraint_result : 2D array of float or bool
            The constraints, with targets along the first index and times along
            the second.
        """
        # Should be implemented on each subclass of Constraint
        raise NotImplementedError

    # should be replaced with abstractclassmethod when only Python 3 is supported
    @abstractmethod
    def vectorize(self, constraint_list):
        """
        Given a list of constraints, return a vector constraint of this type.

        Parameters
        ----------
        constraint_list : list
            A list of `~astroplan.Constraint` objects.

        Returns
        -------
        constraint : `~astroplan.Constraint`
            A vectorised version of this constraint.
        """
        # A classmethod that should be implemented on each subclass of Constraint
        raise NotImplementedError

    def _recast_limits(self, limit):
        """
        Ensure the limits can be broadcast against the supplied targets

        If we want to broadcast the limits of a constraint against a list
        of targets, then the limits should be an array of shape (N, 1),
        where N is the number of targets in the list.

        Returns
        -------
        recast_limit : `~numpy.ndarray`
            The limit recast to the correct shape
        """
        # do nothing if limit is None
        if limit is None:
            return None
        # change lists of Quantities to a non-scalar Quantity
        # This is very slow. Is it needed?
        if isinstance(limit, list) and isinstance(limit[0], u.Quantity):
            limit = u.Quantity(limit)
        # change lists of Times to a non-scalar Time
        if isinstance(limit, list) and isinstance(limit[0], Time):
            limit = Time(limit)
        return np.atleast_2d(limit).T

    def _check_limit_shape(self, values, limit):
        """
        Check to make sure that limit shape is broadcastable against values

        For a well-behaved constraint, values should have a leading dimension of N,
        where N is the number of targets. This should be broadcastable against the
        (1,N) shape of the limits. This routine checks that, and raises a ValueError
        if not true.
        """
        # do nothing if limit is None
        if limit is None:
            return
        # get shapes, or () if no shape attribute (i.e scalar)
        limit_shape = getattr(limit, 'shape', ())
        value_shape = getattr(values, 'shape', ())
        if limit_shape == () or limit_shape == (1, 1):
            # scalar limits always OK
            return
        # we have non-scalar limits
        if value_shape == () or value_shape[0] != limit_shape[0]:
            raise ValueError("Cannot broadcast number of targets and constraint limits")


class AltitudeConstraint(Constraint):
    """
    Constrain the altitude of the target.

    .. note::
        This can misbehave if you try to constrain negative altitudes, as
        the `~astropy.coordinates.AltAz` frame tends to mishandle negative


    Parameters
    ----------
    min : `~astropy.units.Quantity` or `None`
        Minimum altitude of the target (inclusive). `None` indicates no limit.
    max : `~astropy.units.Quantity` or `None`
        Maximum altitude of the target (inclusive). `None` indicates no limit.
    boolean_constraint : bool
        If True, the constraint is treated as a boolean (True for within the
        limits and False for outside).  If False, the constraint returns a
        float on [0, 1], where 0 is the min altitude and 1 is the max.
    """

    min = limit('min')  # ensure any attempt to set min goes through recast_limits
    max = limit('max')

    def __init__(self, min=None, max=None, boolean_constraint=True):
        if min is None:
            self.min = -90*u.deg
        else:
            self.min = min
        if max is None:
            self.max = 90*u.deg
        else:
            self.max = max

        self.boolean_constraint = boolean_constraint

    @classmethod
    def vectorize(cls, constraint_list):
        # in the spirit of duck typing we don't run checks
        min_vals = _get_limit_values(constraint_list, 'min')
        max_vals = _get_limit_values(constraint_list, 'max')
        boolean_constraint = np.all([c.boolean_constraint for c in constraint_list])
        return cls(min_vals, max_vals, boolean_constraint)

    def compute_constraint(self, times, observer, targets):
        cached_altaz = _get_altaz(times, observer, targets)
        alt = cached_altaz['altaz'].alt
        # ensure broadcastability
        self._check_limit_shape(alt, self.min)
        self._check_limit_shape(alt, self.max)
        if self.boolean_constraint:
            lowermask = self.min <= alt
            uppermask = alt <= self.max
            return lowermask & uppermask
        else:
            return max_best_rescale(alt, self.min, self.max)


class AirmassConstraint(Constraint):
    """
    Constrain the airmass of a target.

    In the current implementation the airmass is approximated by the secant of
    the zenith angle.

    .. note::
        The ``max`` and ``min`` arguments appear in the order (max, min)
        in this initializer to support the common case for users who care
        about the upper limit on the airmass (``max``) and not the lower
        limit.

    Parameters
    ----------
    max : float or `None`
        Maximum airmass of the target. `None` indicates no limit.
    min : float or `None`
        Minimum airmass of the target. `None` indicates no limit.
    boolean_contstraint : bool

    Examples
    --------
    To create a constraint that requires the airmass be "better than 2",
    i.e. at a higher altitude than airmass=2::

        AirmassConstraint(2)
    """

    min = limit('min')  # ensure any attempt to set min goes through recast_limits
    max = limit('max')

    def __init__(self, max=None, min=1, boolean_constraint=True):
        self.min = min
        self.max = max
        self.boolean_constraint = boolean_constraint

    @classmethod
    def vectorize(cls, constraint_list):
        # in the spirit of duck typing we don't run checks
        min_vals = _get_limit_values(constraint_list, 'min')
        max_vals = _get_limit_values(constraint_list, 'max')
        boolean_constraint = np.any([c.boolean_constraint for c in constraint_list])
        return cls(max_vals, min_vals, boolean_constraint)

    def compute_constraint(self, times, observer, targets):
        cached_altaz = _get_altaz(times, observer, targets)
        secz = cached_altaz['altaz'].secz
        # ensure broadcastability
        self._check_limit_shape(secz, self.min)
        self._check_limit_shape(secz, self.max)
        if self.boolean_constraint:
            if self.min is None and self.max is not None:
                mask = secz <= self.max
            elif self.max is None and self.min is not None:
                mask = self.min <= secz
            elif self.min is not None and self.max is not None:
                mask = (self.min <= secz) & (secz <= self.max)
            else:
                raise ValueError("No max and/or min specified in "
                                 "AirmassConstraint.")
            return mask
        else:
            if self.max is None:
                raise ValueError("Cannot have a float AirmassConstraint if max "
                                 "is None")
            else:
                mx = self.max

            mi = self._recast_limits(1) if self.min is None else self.min
            # values below 1 should be disregarded
            return min_best_rescale(secz, mi, mx, less_than_min=0)


class AtNightConstraint(Constraint):
    """
    Constrain the Sun to be below ``horizon``.
    """

    max_solar_altitude = limit('max_solar_altitude')

    @u.quantity_input(horizon=u.deg)
    def __init__(self, max_solar_altitude=0*u.deg, force_pressure_zero=True):
        """
        Parameters
        ----------
        max_solar_altitude : `~astropy.units.Quantity`
            The altitude of the sun below which it is considered to be "night"
            (inclusive).
        force_pressure_zero : bool (optional)
            Force the pressure to zero for solar altitude calculations. This
            avoids errors in the altitude of the Sun that can occur when the
            Sun is below the horizon and the corrections for atmospheric
            refraction return nonsense values.
        """
        self.max_solar_altitude = max_solar_altitude
        self.force_pressure_zero = force_pressure_zero

    @classmethod
    def vectorize(cls, constraint_list):
        max_vals = _get_limit_values(constraint_list, 'max_solar_altitude')
        pressure_zero = np.any([c.force_pressure_zero for c in constraint_list])
        return cls(max_vals, pressure_zero)

    @classmethod
    def twilight_civil(cls, **kwargs):
        """
        Consider nighttime as time between civil twilights (-6 degrees).
        """
        return cls(max_solar_altitude=-6*u.deg, **kwargs)

    @classmethod
    def twilight_nautical(cls, **kwargs):
        """
        Consider nighttime as time between nautical twilights (-12 degrees).
        """
        return cls(max_solar_altitude=-12*u.deg, **kwargs)

    @classmethod
    def twilight_astronomical(cls, **kwargs):
        """
        Consider nighttime as time between astronomical twilights (-18 degrees).
        """
        return cls(max_solar_altitude=-18*u.deg, **kwargs)

    def _get_solar_altitudes(self, times, observer, targets):
        if not hasattr(observer, '_altaz_cache'):
            observer._altaz_cache = {}

        aakey = (tuple(times.jd), 'sun')

        if aakey not in observer._altaz_cache:
            try:
                if self.force_pressure_zero:
                    observer_old_pressure = observer.pressure
                    observer.pressure = 0

                # find solar altitude at these times
                altaz = observer.altaz(times, get_sun(times), grid=False)
                altitude = altaz.alt
                # cache the altitude
                observer._altaz_cache[aakey] = dict(times=times,
                                                    altitude=altitude)
            finally:
                if self.force_pressure_zero:
                    observer.pressure = observer_old_pressure
        else:
            altitude = observer._altaz_cache[aakey]['altitude']

        # Broadcast the solar altitudes for the number of targets.
        # Needs to be done after storing/fetching cache so we get the
        # correct shape if targets changes, but times does not.
        altitude = np.atleast_2d(altitude)
        altitude = altitude + np.zeros((len(targets), 1))
        return altitude

    def compute_constraint(self, times, observer, targets):
        solar_altitude = self._get_solar_altitudes(times, observer, targets)
        self._check_limit_shape(solar_altitude, self.max_solar_altitude)
        mask = solar_altitude <= self.max_solar_altitude
        return mask


class SunSeparationConstraint(Constraint):
    """
    Constrain the distance between the Sun and some targets.
    """

    min = limit('min')
    max = limit('max')

    def __init__(self, min=None, max=None):
        """
        Parameters
        ----------
        min : `~astropy.units.Quantity` or `None` (optional)
            Minimum acceptable separation between Sun and target (inclusive).
            `None` indicates no limit.
        max : `~astropy.units.Quantity` or `None` (optional)
            Minimum acceptable separation between Sun and target (inclusive).
            `None` indicates no limit.
        """
        if min is None:
            self.min = 0*u.deg
        else:
            self.min = min
        if max is None:
            self.max = 180*u.deg
        else:
            self.max = max

    @classmethod
    def vectorize(cls, constraint_list):
        return cls(
            _get_limit_values(constraint_list, 'min'),
            _get_limit_values(constraint_list, 'max')
        )

    def compute_constraint(self, times, observer, targets):
        sunaltaz = observer.altaz(times, get_sun(times), grid=False)
        target_coos = [target.coord if hasattr(target, 'coord') else target
                       for target in targets]
        target_altazs = [observer.altaz(times, coo) for coo in target_coos]
        solar_separation = Angle([sunaltaz.separation(taa) for taa in target_altazs])

        # check broadcastability
        self._check_limit_shape(solar_separation, self.min)
        self._check_limit_shape(solar_separation, self.max)

        mask = ((self.min <= solar_separation) &
                (solar_separation <= self.max))
        return mask


class MoonSeparationConstraint(Constraint):
    """
    Constrain the distance between the Earth's moon and some targets.
    """

    min = limit('min')
    max = limit('max')

    def __init__(self, min=None, max=None, ephemeris=None):
        """
        Parameters
        ----------
        min : `~astropy.units.Quantity` or `None` (optional)
            Minimum acceptable separation between moon and target (inclusive).
            `None` indicates no limit.
        max : `~astropy.units.Quantity` or `None` (optional)
            Maximum acceptable separation between moon and target (inclusive).
            `None` indicates no limit.
        ephemeris : str, optional
            Ephemeris to use.  If not given, use the one set with
            ``astropy.coordinates.solar_system_ephemeris.set`` (which is
            set to 'builtin' by default).
        """
        if min is None:
            self.min = 0*u.deg
        else:
            self.min = min
        if max is None:
            self.max = 180*u.deg
        else:
            self.max = max
        self.ephemeris = ephemeris

    @classmethod
    def vectorize(cls, constraint_list):
        return cls(
            _get_limit_values(constraint_list, 'min'),
            _get_limit_values(constraint_list, 'max')
        )

    def compute_constraint(self, times, observer, targets):
        # TODO: when astropy/astropy#5069 is resolved, replace this workaround which
        # handles scalar and non-scalar time inputs differently
        if times.isscalar:
            moon = get_moon(times, observer.location, ephemeris=self.ephemeris)
        else:
            # must get moon coordinates in an earth centred frame
            altaz_frame = AltAz(obstime=times[0], location=observer.location)
            moon_coords = [get_moon(t, observer.location, ephemeris=self.ephemeris).transform_to(altaz_frame)
                           for t in times]
            obstime = [coord.obstime for coord in moon_coords]
            alts = u.Quantity([coord.alt for coord in moon_coords])
            azs = u.Quantity([coord.az for coord in moon_coords])
            dists = u.Quantity([coord.distance for coord in moon_coords])
            moon = SkyCoord(AltAz(azs, alts, dists, obstime=obstime, location=observer.location))

        targets = get_skycoord(targets)

        # has to be this way around, so the target coords are transformed to an
        # Earth-centred frame before calculating angular separation.
        moon_separation = targets[:, np.newaxis].separation(moon)
        # check broadcastability
        self._check_limit_shape(moon_separation, self.min)
        self._check_limit_shape(moon_separation, self.max)
        mask = ((self.min <= moon_separation) &
                (moon_separation <= self.max))
        return mask


class MoonIlluminationConstraint(Constraint):
    """
    Constrain the fractional illumination of the Earth's moon.

    Constraint is also satisfied if the Moon has set.
    """

    min = limit('min')
    max = limit('max')

    def __init__(self, min=None, max=None, ephemeris=None):
        """
        Parameters
        ----------
        min : float or `None` (optional)
            Minimum acceptable fractional illumination (inclusive). `None`
            indicates no limit.
        max : float or `None` (optional)
            Maximum acceptable fractional illumination (inclusive). `None`
            indicates no limit.
        ephemeris : str, optional
            Ephemeris to use.  If not given, use the one set with
            `~astropy.coordinates.solar_system_ephemeris` (which is
            set to 'builtin' by default).
        """
        self.min = min if min is not None else -0.5
        self.max = max if max is not None else 1.5
        self.ephemeris = ephemeris

    @classmethod
    def dark(cls, min=None, max=0.25, **kwargs):
        """
        initialize a `~astroplan.constraints.MoonIlluminationConstraint`
        with defaults of no minimum and a maximum of 0.25

        Parameters
        ----------
        min : float or `None` (optional)
            Minimum acceptable fractional illumination (inclusive). `None`
            indicates no limit.
        max : float or `None` (optional)
            Maximum acceptable fractional illumination (inclusive). `None`
            indicates no limit.
        """
        return cls(min, max, **kwargs)

    @classmethod
    def grey(cls, min=0.25, max=0.65, **kwargs):
        """
        initialize a `~astroplan.constraints.MoonIlluminationConstraint`
        with defaults of a minimum of 0.25 and a maximum of 0.65

        Parameters
        ----------
        min : float or `None` (optional)
            Minimum acceptable fractional illumination (inclusive). `None`
            indicates no limit.
        max : float or `None` (optional)
            Maximum acceptable fractional illumination (inclusive). `None`
            indicates no limit.
        """
        return cls(min, max, **kwargs)

    @classmethod
    def bright(cls, min=0.65, max=None, **kwargs):
        """
        initialize a `~astroplan.constraints.MoonIlluminationConstraint`
        with defaults of a minimum of 0.65 and no maximum

        Parameters
        ----------
        min : float or `None` (optional)
            Minimum acceptable fractional illumination (inclusive). `None`
            indicates no limit.
        max : float or `None` (optional)
            Maximum acceptable fractional illumination (inclusive). `None`
            indicates no limit.
        """
        return cls(min, max, **kwargs)

    def vectorize(cls, constraint_list):
        return cls(
            _get_limit_values(constraint_list, 'min'),
            _get_limit_values(constraint_list, 'max')
        )

    def compute_constraint(self, times, observer, targets):
        # first is the moon up?
        cached_moon = _get_moon_data(times, observer)
        moon_alt = cached_moon['altaz'].alt
        moon_down_mask = moon_alt < 0
        moon_up_mask = ~moon_down_mask

        illumination = cached_moon['illum']

        """
        illumination and mask arrays are (ntimes,) whilst
        limits are (ntargets, 1). These will broadcast to make
        an (ntargets, ntimes) array
        """
        if self.min < 0 and self.max < 1.5:
            mask = (self.max >= illumination) | moon_down_mask
        elif self.max > 1.0 and self.min > -0.5:
            mask = (self.min <= illumination) & moon_up_mask
        elif self.min > -0.5 and self.max < 1.5:
            mask = ((self.min <= illumination) &
                    (illumination <= self.max)) & moon_up_mask
        else:
            raise ValueError("No max and/or min specified in "
                             "MoonSeparationConstraint.")

        return mask


class LocalTimeConstraint(Constraint):
    """
    Constrain the observable hours.
    """

    min = limit('min')
    max = limit('max')

    def __init__(self, min=None, max=None):
        """
        Parameters
        ----------
        min : `~datetime.time`
            Earliest local time (inclusive). `None` indicates no limit.

        max : `~datetime.time`
            Latest local time (inclusive). `None` indicates no limit.

        Examples
        --------
        Constrain the observations to targets that are observable between
        23:50 and 04:08 local time:

        >>> from astroplan import Observer
        >>> from astroplan.constraints import LocalTimeConstraint
        >>> import datetime as dt
        >>> subaru = Observer.at_site("Subaru", timezone="US/Hawaii")
        >>> # bound times between 23:50 and 04:08 local Hawaiian time
        >>> constraint = LocalTimeConstraint(min=dt.time(23,50), max=dt.time(4,8))
        """
        if min is None and max is None:
            raise ValueError("You must at least supply either a minimum or a maximum time.")

        if min is not None:
            valid_type = False
            try:
                valid_type = all(isinstance(item, datetime.time) for item in min)
            except:
                valid_type = isinstance(min, datetime.time)
            if not valid_type:
                raise TypeError("Time limits must be specified as datetime.time objects.")
            self.min = min
        else:
            self.min = datetime.time(0, 0, 0)

        if max is not None:
            valid_type = False
            try:
                valid_type = all(isinstance(item, datetime.time) for item in max)
            except:
                valid_type = isinstance(max, datetime.time)
            if not valid_type:
                raise TypeError("Time limits must be specified as datetime.time objects.")
            self.max = max
        else:
            self.max = datetime.time(23, 59, 59)

    @classmethod
    def vectorize(cls, constraint_list):
        return cls(
            _get_limit_values(constraint_list, 'min'),
            _get_limit_values(constraint_list, 'max')
        )

    def compute_constraint(self, times, observer, targets):

        timezone = None
        gettz = np.frompyfunc(lambda x: getattr(x, 'tzinfo'), 1, 1)

        # get timezone from time objects, or from observer
        timezone = gettz(self.min)

        if timezone is None:
            timezone = gettz(self.max)

        if timezone is None:
            timezone = self._recast_limits(observer.timezone)

        # make numpy ufunc to get time from astropy time object
        gettime = np.frompyfunc(lambda x: x.time(), 1, 1)

        # If time limits occur on same day:
        same_day_mask = np.tile(self.min < self.max, len(times))
        same_day_mask_values = np.logical_and(self.min <= gettime(times.datetime),
                                              gettime(times.datetime) <= self.max)
        mask = np.logical_or(gettime(times.datetime) >= self.min,
                             gettime(times.datetime) <= self.max)
        mask[same_day_mask] = same_day_mask_values[same_day_mask]

        if targets is not None:
            if mask.shape != (len(targets), len(times)):
                mask = np.tile(mask, len(targets))
                mask = mask.reshape(len(targets), len(times))
        return mask


class TimeConstraint(Constraint):
    """Constrain the observing time to be within certain time limits.

    An example use case for this class would be to associate an acceptable
    time range with a specific observing block. This can be useful if not
    all observing blocks are valid over the time limits used in calls
    to `is_observable` or `is_always_observable`.
    """
    def __init__(self, min=None, max=None):
        """
        Parameters
        ----------
        min : `~astropy.time.Time`
            Earliest time (inclusive). `None` indicates no limit.

        max : `~astropy.time.Time`
            Latest time (inclusive). `None` indicates no limit.

        Examples
        --------
        Constrain the observations to targets that are observable between
        2016-03-28 and 2016-03-30:

        >>> from astroplan import Observer
        >>> from astropy.time import Time
        >>> subaru = Observer.at_site("Subaru")
        >>> t1 = Time("2016-03-28T12:00:00")
        >>> t2 = Time("2016-03-30T12:00:00")
        >>> constraint = TimeConstraint(t1,t2)
        """
        self.min = min
        self.max = max

        if self.min is None and self.max is None:
            raise ValueError("You must at least supply either a minimum or a "
                             "maximum time.")

        if self.min is not None:
            valid_input = False
            try:
                valid_input = all(isinstance(item, Time) for item in self.min)
                self.min = Time(self.min)  # change lists of Times to a non-scalar Time
            except:
                valid_input = isinstance(self.min, Time)
            if not valid_input:
                raise TypeError("Time limits must be specified as astropy.time.Time objects.")

        if self.max is not None:
            valid_input = False
            try:
                valid_input = all(isinstance(item, Time) for item in self.max)
                self.max = Time(self.max)  # change lists of Times to a non-scalar Time
            except:
                valid_input = isinstance(self.max, Time)
            if not valid_input:
                raise TypeError("Time limits must be specified as astropy.time.Time objects.")

    @classmethod
    def vectorize(cls, constraint_list):
        min_vals = [c.min for c in constraint_list]
        max_vals = [c.max for c in constraint_list]
        if np.any(min_vals is None):
            if np.all(min_vals is None):
                min_vals = None
            else:
                raise ValueError("Cannot vectorize constraints with mixture of scalar and NoneType limits")
        if np.any(max_vals is None):
            if np.all(max_vals is None):
                max_vals = None
            else:
                raise ValueError("Cannot vectorize constraints with mixture of scalar and NoneType limits")

        if np.any([not val.isscalar for val in min_vals]):
            raise ValueError("Not all min limits are scalar")
        if np.any([not val.isscalar for val in max_vals]):
            raise ValueError("Not all max limits are scalar")
        return cls(min_vals, max_vals)

    def compute_constraint(self, times, observer, targets):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            min_time = Time("1950-01-01T00:00:00") if self.min is None else self.min
            max_time = Time("2120-01-01T00:00:00") if self.max is None else self.max

        if min_time.isscalar:
            min_mask = times > min_time
            min_mask = np.tile(min_mask, len(targets))
            min_mask = min_mask.reshape(len(targets), len(times))
        else:
            min_mask = times > min_time[:, np.newaxis]

        if max_time.isscalar:
            max_mask = times < max_time
            max_mask = np.tile(max_mask, len(targets))
            max_mask = max_mask.reshape(len(targets), len(times))
        else:
            max_mask = times < max_time[:, np.newaxis]
        mask = np.logical_and(min_mask, max_mask)
        return mask


def is_always_observable(constraints, observer, targets, times=None,
                         time_range=None, time_grid_resolution=0.5*u.hour):
    """
    A function to determine whether ``targets`` are always observable throughout
    ``time_range`` given constraints in the ``constraints_list`` for a
    particular ``observer``.

    Parameters
    ----------
    constraints : list or `~astroplan.constraints.Constraint`
        Observational constraint(s)

    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``

    targets : {list, `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`}
        Target or list of targets

    times : `~astropy.time.Time` (optional)
        Array of times on which to test the constraint

    time_range : `~astropy.time.Time` (optional)
        Lower and upper bounds on time sequence, with spacing
        ``time_resolution``. This will be passed as the first argument into
        `~astroplan.time_grid_from_range`.

    time_grid_resolution : `~astropy.units.Quantity` (optional)
        If ``time_range`` is specified, determine whether constraints are met
        between test times in ``time_range`` by checking constraint at
        linearly-spaced times separated by ``time_resolution``. Default is 0.5
        hours.

    Returns
    -------
    ever_observable : list
        List of booleans of same length as ``targets`` for whether or not each
        target is observable in the time range given the constraints.
    """
    if not hasattr(constraints, '__len__'):
        constraints = [constraints]

    applied_constraints = [constraint(observer, targets, times=times,
                                      time_range=time_range,
                                      time_grid_resolution=time_grid_resolution)
                           for constraint in constraints]
    constraint_arr = np.logical_and.reduce(applied_constraints)
    return np.all(constraint_arr, axis=1)


def is_observable(constraints, observer, targets, times=None,
                  time_range=None, time_grid_resolution=0.5*u.hour):
    """
    Determines if the ``targets`` are observable during ``time_range`` given
    constraints in ``constraints_list`` for a particular ``observer``.

    Parameters
    ----------
    constraints : list or `~astroplan.constraints.Constraint`
        Observational constraint(s)

    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``

    targets : {list, `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`}
        Target or list of targets

    times : `~astropy.time.Time` (optional)
        Array of times on which to test the constraint

    time_range : `~astropy.time.Time` (optional)
        Lower and upper bounds on time sequence, with spacing
        ``time_resolution``. This will be passed as the first argument into
        `~astroplan.time_grid_from_range`.

    time_grid_resolution : `~astropy.units.Quantity` (optional)
        If ``time_range`` is specified, determine whether constraints are met
        between test times in ``time_range`` by checking constraint at
        linearly-spaced times separated by ``time_resolution``. Default is 0.5
        hours.

    Returns
    -------
    ever_observable : list
        List of booleans of same length as ``targets`` for whether or not each
        target is ever observable in the time range given the constraints.
    """
    if not hasattr(constraints, '__len__'):
        constraints = [constraints]

    applied_constraints = [constraint(observer, targets, times=times,
                                      time_range=time_range,
                                      time_grid_resolution=time_grid_resolution)
                           for constraint in constraints]
    constraint_arr = np.logical_and.reduce(applied_constraints)
    return np.any(constraint_arr, axis=1)


def months_observable(constraints, observer, targets,
                      time_grid_resolution=0.5*u.hour):
    """
    Determines which month the specified ``targets`` are observable for a
    specific ``observer``, given the supplied ``constriants``.

    Parameters
    ----------
    constraints : list or `~astroplan.constraints.Constraint`
        Observational constraint(s)

    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``

    targets : {list, `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`}
        Target or list of targets

    time_grid_resolution : `~astropy.units.Quantity` (optional)
        If ``time_range`` is specified, determine whether constraints are met
        between test times in ``time_range`` by checking constraint at
        linearly-spaced times separated by ``time_resolution``. Default is 0.5
        hours.

    Returns
    -------
    observable_months : list
        List of sets of unique integers representing each month that a target is
        observable, one set per target. These integers are 1-based so that
        January maps to 1, February maps to 2, etc.

    """
    # TODO: This method could be sped up a lot by dropping to the trigonometric
    # altitude calculations.
    if not hasattr(constraints, '__len__'):
        constraints = [constraints]

    # Calculate throughout the year of 2014 so as not to require forward
    # extrapolation off of the IERS tables
    time_range = Time(['2014-01-01', '2014-12-31'])
    times = time_grid_from_range(time_range, time_grid_resolution)

    # TODO: This method could be sped up a lot by dropping to the trigonometric
    # altitude calculations.

    applied_constraints = [constraint(observer, targets,
                                      times=times)
                           for constraint in constraints]
    constraint_arr = np.logical_and.reduce(applied_constraints)

    months_observable = []
    for target, observable in zip(targets, constraint_arr):
        s = set([t.datetime.month for t in times[observable]])
        months_observable.append(s)

    return months_observable


def observability_table(constraints, observer, targets, times=None,
                        time_range=None, time_grid_resolution=0.5*u.hour):
    """
    Creates a table with information about observability for all  the ``targets``
    over the requested ``time_range``, given the constraints in
    ``constraints_list`` for ``observer``.

    Parameters
    ----------
    constraints : list or `~astroplan.constraints.Constraint`
        Observational constraint(s)

    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``

    targets : {list, `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`}
        Target or list of targets

    times : `~astropy.time.Time` (optional)
        Array of times on which to test the constraint

    time_range : `~astropy.time.Time` (optional)
        Lower and upper bounds on time sequence, with spacing
        ``time_resolution``. This will be passed as the first argument into
        `~astroplan.time_grid_from_range`.

    time_grid_resolution : `~astropy.units.Quantity` (optional)
        If ``time_range`` is specified, determine whether constraints are met
        between test times in ``time_range`` by checking constraint at
        linearly-spaced times separated by ``time_resolution``. Default is 0.5
        hours.

    Returns
    -------
    observability_table : `~astropy.table.Table`
        A Table containing the observability information for each of the
        ``targets``. The table contains four columns with information about the
        target and it's observability: ``'target name'``, ``'ever observable'``,
        ``'always observable'``, and ``'fraction of time observable'``.  It also
        contains metadata entries ``'times'`` (with an array of all the times),
        ``'observer'`` (the `~astroplan.Observer` object), and ``'constraints'``
        (containing the supplied ``constraints``).
    """
    if not hasattr(constraints, '__len__'):
        constraints = [constraints]

    applied_constraints = [constraint(observer, targets, times=times,
                                      time_range=time_range,
                                      time_grid_resolution=time_grid_resolution)
                           for constraint in constraints]
    constraint_arr = np.logical_and.reduce(applied_constraints)

    colnames = ['target name', 'ever observable', 'always observable',
                'fraction of time observable']

    target_names = [target.name for target in targets]
    ever_obs = np.any(constraint_arr, axis=1)
    always_obs = np.all(constraint_arr, axis=1)
    frac_obs = np.sum(constraint_arr, axis=1) / constraint_arr.shape[1]

    tab = table.Table(names=colnames, data=[target_names, ever_obs, always_obs,
                                            frac_obs])

    if times is None and time_range is not None:
        times = time_grid_from_range(time_range,
                                     time_resolution=time_grid_resolution)

    tab.meta['times'] = times.datetime
    tab.meta['observer'] = observer
    tab.meta['constraints'] = constraints

    return tab


def min_best_rescale(vals, min_val, max_val, less_than_min=1):
    """
    rescales an input array ``vals`` to be a score (between zero and one),
    where the ``min_val`` goes to one, and the ``max_val`` goes to zero.

    Parameters
    ----------
    vals : array-like
        the values that need to be rescaled to be between 0 and 1
    min_val : float
        worst acceptable value (rescales to 0)
    max_val : float
        best value cared about (rescales to 1)
    less_than_min : 0 or 1
        what is returned for ``vals`` below ``min_val``. (in some cases
        anything less than ``min_val`` should also return one,
        in some cases it should return zero)

    Returns
    -------
    array of floats between 0 and 1 inclusive rescaled so that
    ``vals`` equal to ``max_val`` equal 0 and those equal to
    ``min_val`` equal 1

    Examples
    --------
    rescale airmasses to between 0 and 1, with the best (1)
    and worst (2.25). All values outside the range should
    return 0.
    >>> from astroplan.constraints import min_best_rescale
    >>> import numpy as np
    >>> airmasses = np.array([1, 1.5, 2, 3, 0])
    >>> min_best_rescale(airmasses, 1, 2.25, less_than_min = 0)
    array([ 1. ,  0.6,  0.2,  0. , 0. ])
    """
    rescaled = (vals - max_val) / (min_val - max_val)
    below = vals < min_val
    above = vals > max_val
    rescaled[below] = less_than_min
    rescaled[above] = 0

    return rescaled


def max_best_rescale(vals, min_val, max_val, greater_than_max=1):
    """
    rescales an input array ``vals`` to be a score (between zero and one),
    where the ``max_val`` goes to one, and the ``min_val`` goes to zero.

    Parameters
    ----------
    vals : array-like
        the values that need to be rescaled to be between 0 and 1
    min_val : float
        worst acceptable value (rescales to 0)
    max_val : float
        best value cared about (rescales to 1)
    greater_than_max : 0 or 1
        what is returned for ``vals`` above ``max_val``. (in some cases
        anything higher than ``max_val`` should also return one,
        in some cases it should return zero)

    Returns
    -------
    array of floats between 0 and 1 inclusive rescaled so that
    ``vals`` equal to ``min_val`` equal 0 and those equal to
    ``max_val`` equal 1

    Examples
    --------
    rescale an array of altitudes to be between 0 and 1,
    with the best (60) going to 1 and worst (35) going to
    0. For values outside the range, the rescale should
    return 0 below 35 and 1 above 60.
    >>> from astroplan.constraints import max_best_rescale
    >>> import numpy as np
    >>> altitudes = np.array([20, 30, 40, 45, 55, 70])
    >>> max_best_rescale(altitudes, 35, 60)
    array([ 0. , 0. , 0.2, 0.4, 0.8, 1. ])
    """
    rescaled = (vals - min_val) / (max_val - min_val)
    below = vals < min_val
    above = vals > max_val
    rescaled[below] = 0
    rescaled[above] = greater_than_max

    return rescaled
