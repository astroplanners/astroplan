# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.time import Time

__all__ = ['PeriodicEvent', 'EclipsingSystem']


class PeriodicEvent:
    """
    A periodic event defined by an epoch and period.
    """
    @u.quantity_input(period=u.day, duration=u.day)
    def __init__(self, epoch, period, duration=None, name=None):
        """

        Parameters
        ----------
        epoch : `~astropy.time.Time`
            Time of event
        period : `~astropy.units.Quantity`
            Period of event
        duration : `~astropy.units.Quantity` (optional)
            Duration of event
        name : str (optional)
            Name of target/event
        """
        self.epoch = epoch
        self.period = period
        self.name = name
        self.duration = duration

    def phase(self, time):
        """
        Phase of periodic event, on interval [0, 1). For example, the phase
        could be an orbital phase for an eclipsing binary system.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Evaluate the phase at this time or times

        Returns
        -------
        phase_array : `~numpy.ndarray`
            Phase at each ``time``, on range [0, 1)
        """
        return ((time - self.epoch).to(u.day).value %
                self.period.to(u.day).value) / self.period.to(u.day).value


class EclipsingSystem(PeriodicEvent):
    """
    Define parameters for an eclipsing system; useful for an eclipsing binary or
    transiting exoplanet.

    .. warning::
        There are currently two major caveats in the implementation of
        ``EclipsingSystem``. The secondary eclipse time approximation is
        only accurate when the orbital eccentricity is small, and the eclipse
        times are computed without any barycentric corrections. The current
        implementation should only be used forapproximate mid-eclipse times for
        low eccentricity orbits, with event durations longer than the
        barycentric correction error (<=16 minutes).
    """
    @u.quantity_input(period=u.day, duration=u.day)
    def __init__(self, primary_eclipse_time, orbital_period, duration=None,
                 name=None, eccentricity=None, argument_of_periapsis=None):
        """
        Parameters
        ----------
        primary_eclipse_time : `~astropy.time.Time`
            Time of primary eclipse
        orbital_period : `~astropy.units.Quantity`
            Orbital period of eclipsing system
        duration : `~astropy.units.Quantity` (optional)
            Duration of eclipse
        name : str (optional)
            Name of target/event
        eccentricity : float (optional)
            Orbital eccentricity. Default is `None`, which assumes circular
            orbit (e=0).
        argument_of_periapsis : float (optional)
            Argument of periapsis for the eclipsing system, in radians.
            Default is `None`, which assumes pi/2.
        """
        self.epoch = primary_eclipse_time
        self.period = orbital_period
        self.name = name
        self.duration = duration

        if eccentricity is None:
            eccentricity = 0
        self.eccentricity = eccentricity

        if argument_of_periapsis is None:
            argument_of_periapsis = np.pi/2
        self.argument_of_periapsis = argument_of_periapsis

    def in_primary_eclipse(self, time):
        """
        Returns `True` when ``time`` is during a primary eclipse.

        .. warning::
            Barycentric offsets are ignored in the current implementation.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Time to evaluate

        Returns
        -------
        in_eclipse : `~numpy.ndarray` or bool
            `True` if ``time`` is during primary eclipse
        """
        phases = self.phase(time)
        return ((phases < float(self.duration/self.period)/2) |
                (phases > 1 - float(self.duration/self.period)/2))

    def in_secondary_eclipse(self, time):
        r"""
        Returns `True` when ``time`` is during a secondary eclipse

        If the eccentricity of the eclipsing system is non-zero, then we compute
        the secondary eclipse time approximated to first order in eccentricity,
        as described in Winn (2010) Equation 33 [1]_:

        The time between the primary eclipse and secondary eclipse :math:`\delta t_c`
        is given by :math:`\delta t_c \approx 0.5 \left (\frac{4}{\pi} e \cos{\omega \right)`,
        where :math:`e` is the orbital eccentricity and :math:`\omega` is the
        angle of periapsis.

        .. warning::
            This approximation for the secondary eclipse time is only accurate
            when the orbital eccentricity is small; and barycentric offsets
            are ignored in the current implementation.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Time to evaluate

        Returns
        -------
        in_eclipse : `~numpy.ndarray` or bool
            `True` if ``time`` is during secondary eclipse

        References
        ----------
        .. [1] Winn (2010) https://arxiv.org/abs/1001.2010
        """
        if self.eccentricity < 1e-5:
            secondary_eclipse_phase = 0.5
        else:
            secondary_eclipse_phase = 0.5 * (1 + 4/np.pi * self.eccentricity *
                                             np.cos(self.argument_of_periapsis))
        phases = self.phase(time)
        return ((phases < secondary_eclipse_phase + float(self.duration/self.period)/2) &
                (phases > secondary_eclipse_phase - float(self.duration/self.period)/2))

    def out_of_eclipse(self, time):
        """
        Returns `True` when ``time`` is not during primary or secondary eclipse.

        .. warning::
            Barycentric offsets are ignored in the current implementation.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Time to evaluate

        Returns
        -------
        in_eclipse : `~numpy.ndarray` or bool
            `True` if ``time`` is not during primary or secondary eclipse
        """
        return np.logical_not(np.logical_or(self.in_primary_eclipse(time),
                                            self.in_secondary_eclipse(time)))

    def next_primary_eclipse_time(self, time, n_eclipses=1):
        """
        Time of the next primary eclipse after ``time``.

        .. warning::
            Barycentric offsets are ignored in the current implementation.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Find the next primary eclipse after ``time``
        n_eclipses : int (optional)
            Return the times of eclipse for the next ``n_eclipses`` after
            ``time``. Default is 1.

        Returns
        -------
        primary_eclipses : `~astropy.time.Time`
            Times of the next ``n_eclipses`` primary eclipses after ``time``
        """
        eclipse_times = ((1-self.phase(time)) * self.period + time +
                         np.arange(n_eclipses) * self.period)
        return eclipse_times

    def next_secondary_eclipse_time(self, time, n_eclipses=1):
        """
        Time of the next secondary eclipse after ``time``.

        .. warning::
            Barycentric offsets are ignored in the current implementation.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Find the next secondary eclipse after ``time``
        n_eclipses : int (optional)
            Return the times of eclipse for the next ``n_eclipses`` after
            ``time``. Default is 1.

        Returns
        -------
        secondary_eclipses : `~astropy.time.Time`
            Times of the next ``n_eclipses`` secondary eclipses after ``time``
        """
        phase = self.phase(time)
        if phase >= 0.5:
            next_eclipse_phase = 1.5
        else:
            next_eclipse_phase = 0.5
        eclipse_times = ((next_eclipse_phase - phase) * self.period + time +
                         np.arange(n_eclipses) * self.period)
        return eclipse_times

    def next_primary_ingress_egress_time(self, time, n_eclipses=1):
        """
        Calculate the times of ingress and egress for the next ``n_eclipses``
        primary eclipses after ``time``

        .. warning::
            Barycentric offsets are ignored in the current implementation.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Find the next primary ingress and egress after ``time``
        n_eclipses : int (optional)
            Return the times of eclipse for the next ``n_eclipses`` after
            ``time``. Default is 1.

        Returns
        -------
        primary_eclipses : `~astropy.time.Time` of shape (``n_eclipses``, 2)
            Times of ingress and egress for the next ``n_eclipses`` primary
            eclipses after ``time``
        """
        next_mid_eclipses = self.next_primary_eclipse_time(time, n_eclipses=n_eclipses)
        next_ingresses = next_mid_eclipses - self.duration/2
        next_egresses = next_mid_eclipses + self.duration/2

        ing_egr = np.vstack([next_ingresses.utc.jd, next_egresses.utc.jd]).T

        return Time(ing_egr, format='jd', scale='utc')

    def next_secondary_ingress_egress_time(self, time, n_eclipses=1):
        """
        Calculate the times of ingress and egress for the next ``n_eclipses``
        secondary eclipses after ``time``

        .. warning::
            Barycentric offsets are ignored in the current implementation.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Find the next secondary ingress and egress after ``time``
        n_eclipses : int (optional)
            Return the times of eclipse for the next ``n_eclipses`` after
            ``time``. Default is 1.

        Returns
        -------
        secondary_eclipses : `~astropy.time.Time` of shape (``n_eclipses``, 2)
            Times of ingress and egress for the next ``n_eclipses`` secondary
            eclipses after ``time``.
        """
        next_mid_eclipses = self.next_secondary_eclipse_time(time, n_eclipses=n_eclipses)
        next_ingresses = next_mid_eclipses - self.duration/2
        next_egresses = next_mid_eclipses + self.duration/2

        ing_egr = np.vstack([next_ingresses.utc.jd, next_egresses.utc.jd]).T

        return Time(ing_egr, format='jd', scale='utc')
