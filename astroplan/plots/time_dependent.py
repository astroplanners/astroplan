# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import numpy as np
import operator
import astropy.units as u
from astropy.time import Time
from collections.abc import Sequence
import warnings
import pytz

from ..exceptions import PlotWarning
from ..utils import _set_mpl_style_sheet

__all__ = ['plot_airmass', 'plot_schedule_airmass', 'plot_parallactic',
           'plot_altitude']


def _secz_to_altitude(secant_z):
    """
    Convert airmass (approximated as the secant of the zenith angle) to
    an altitude (aka elevation) in degrees.

    Parameters
    ----------
    secant_z : float
        Secant of the zenith angle

    Returns
    -------
    altitude : float
        Altitude [degrees]
    """
    return np.degrees(np.pi/2 - np.arccos(1./secant_z))


def _has_twin(ax):
    """
    Solution for detecting twin axes built on `ax`. Courtesy of
    Jake Vanderplas http://stackoverflow.com/a/36209590/1340208
    """
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False


def plot_airmass(targets, observer, time, ax=None, style_kwargs=None,
                 style_sheet=None, brightness_shading=False,
                 altitude_yaxis=False, min_airmass=1.0, min_region=None,
                 max_airmass=3.0, max_region=None, use_local_tz=False):
    r"""
    Plots airmass as a function of time for a given target.

    If a `~matplotlib.axes.Axes` object already exists, an additional
    airmass plot will be "stacked" on it.  Otherwise, creates a new
    `~matplotlib.axes.Axes` object and plots airmass on top of that.

    When a scalar `~astropy.time.Time` object is passed in (e.g.,
    ``Time('2000-1-1')``), the resulting plot will use a 24-hour window
    centered on the time indicated, with airmass sampled at regular
    intervals throughout.
    However, the user can control the exact number and frequency of airmass
    calculations used by passing in a non-scalar `~astropy.time.Time`
    object. For instance, ``Time(['2000-1-1 23:00:00', '2000-1-1
    23:30:00'])`` will result in a plot with only two airmass measurements.

    For examples with plots, visit the documentation of
    :ref:`plots_time_dependent`.

    Parameters
    ----------
    targets : list of `~astroplan.FixedTarget` objects
        The celestial bodies of interest.
        If a single object is passed it will be converted to a list.

    observer : `~astroplan.Observer`
        The person, telescope, observatory, etc. doing the observing.

    time : `~astropy.time.Time`
        If scalar (e.g., ``Time('2000-1-1')``), will result in plotting target
        airmasses once an hour over a 24-hour window.
        If non-scalar (e.g., ``Time(['2000-1-1'])``, ``[Time('2000-1-1')]``,
        ``Time(['2000-1-1', '2000-1-2'])``),
        will result in plotting data at the exact times specified.

    ax : `~matplotlib.axes.Axes` or None, optional.
        The `~matplotlib.axes.Axes` object to be drawn on.
        If None, uses the current ``Axes``.

    style_kwargs : dict or None, optional.
        A dictionary of keywords passed into `~matplotlib.pyplot.plot_date`
        to set plotting styles.

    style_sheet : dict or `None` (optional)
        matplotlib style sheet to use. To see available style sheets in
        astroplan, print *astroplan.plots.available_style_sheets*. Defaults
        to the light theme.

    brightness_shading : bool
        Shade background of plot to scale roughly with sky brightness. Dark
        shading signifies times when the sun is below the horizon. Default
        is `False`.

    altitude_yaxis : bool
        Add alternative y-axis on the right side of the figure with target
        altitude. Default is `False`.

    min_airmass : float
        Lower limit of y-axis airmass range in the plot. Default is ``1.0``.

    max_airmass : float
        Upper limit of y-axis airmass range in the plot. Default is ``3.0``.

    min_region : float
        If set, defines an interval between ``min_airmass`` and ``min_region``
        that will be shaded. Default is `None`.

    max_region : float
        If set, defines an interval between ``max_airmass`` and ``max_region``
        that will be shaded. Default is `None`.

    use_local_tz : bool
        If the time is specified in a local timezone, the time will be plotted
        in that timezone.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        An ``Axes`` object with added airmass vs. time plot.

    Notes
    -----
    y-axis is inverted and shows airmasses between 1.0 and 3.0 by default.
    If user wishes to change these, use ``ax.<set attribute>`` before drawing
    or saving plot:

    """
    # Import matplotlib, set style sheet
    if style_sheet is not None:
        _set_mpl_style_sheet(style_sheet)

    import matplotlib.pyplot as plt
    from matplotlib import dates

    # Set up plot axes and style if needed.
    if ax is None:
        ax = plt.gca()
    if style_kwargs is None:
        style_kwargs = {}
    style_kwargs = dict(style_kwargs)

    if 'lw' not in style_kwargs:
        style_kwargs.setdefault('linewidth', 1.5)
    if 'ls' not in style_kwargs and 'linestyle' not in style_kwargs:
        style_kwargs.setdefault('fmt', '-')

    if hasattr(time, 'utcoffset') and use_local_tz:
        tzoffset = time.utcoffset()
        tzname = time.tzname()
        tzinfo = time.tzinfo
    else:
        tzoffset = 0 * u.hour
        tzname = 'UTC'
        tzinfo = None
    # Populate time window if needed.
    # (plot against local time if that's requested)
    time_ut = Time(time)
    if time_ut.isscalar:
        time_ut = time_ut + np.linspace(-12, 12, 100)*u.hour
    elif len(time_ut) == 1:
        warnings.warn('You used a Time array of length 1.  You probably meant '
                      'to use a scalar. (Or maybe a list with length > 1?).',
                      PlotWarning)
    timetoplot = time_ut + tzoffset

    if not isinstance(targets, Sequence):
        targets = [targets]

    for target in targets:
        # Calculate airmass
        airmass = observer.altaz(time_ut, target).secz
        # Mask out nonsense airmasses
        masked_airmass = np.ma.array(airmass, mask=airmass < 1)

        # Some checks & info for labels.
        try:
            target_name = target.name
        except AttributeError:
            target_name = ''

        # Plot data (against timezone-offset time)
        ax.plot_date(timetoplot.plot_date, masked_airmass, label=target_name, **style_kwargs)

    # Format the time axis
    xlo, xhi = (timetoplot[0]), (timetoplot[-1])
    ax.set_xlim([xlo.plot_date, xhi.plot_date])
    date_formatter = dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Shade background during night time
    if brightness_shading:
        start = time_ut[0]
        end = time_ut[-1]
        nights = []

        next_sun_rise = observer.sun_rise_time(start, which='next')
        sun_set = observer.sun_set_time(next_sun_rise, which='previous')
        while (sun_set < end):

            # Calculate and order twilights and set plotting alpha for each
            twilights = [
                (sun_set, 0.0),
                (observer.twilight_evening_civil(next_sun_rise, which='previous'), 0.1),
                (observer.twilight_evening_nautical(next_sun_rise, which='previous'), 0.2),
                (observer.twilight_evening_astronomical(next_sun_rise, which='previous'), 0.3),
                (observer.twilight_morning_astronomical(next_sun_rise, which='previous'), 0.4),
                (observer.twilight_morning_nautical(next_sun_rise, which='previous'), 0.3),
                (observer.twilight_morning_civil(next_sun_rise, which='previous'), 0.2),
                (next_sun_rise, 0.1),
            ]
            nights.append(twilights)

            next_sun_rise = observer.sun_rise_time(next_sun_rise+1*u.hour, which='next')
            sun_set = observer.sun_set_time(next_sun_rise, which='previous')

        for twilights in nights:
            # add 'UTC' to each datetime object created above
            twilights = [(t[0].datetime.replace(tzinfo=pytz.utc), t[1])
                         for t in twilights]

            twilights.sort(key=operator.itemgetter(0))

            # add in left & right edges, so that if the airmass plot is requested
            # during the day, night is properly shaded
            left_edges = [(xlo.datetime.replace(tzinfo=tzinfo), twilights[0][1])] + twilights
            right_edges = twilights + [(xhi.datetime.replace(tzinfo=tzinfo), twilights[0][1])]

            for tw_left, tw_right in zip(left_edges, right_edges):
                left = tw_left[0]
                right = tw_right[0]
                if tzinfo is not None:
                    # convert to local time zone (which is plotted), then hack away the tzinfo
                    # so that matplotlib doesn't try to double down on the conversion
                    left = left.astimezone(tzinfo).replace(tzinfo=None)
                    right = right.astimezone(tzinfo).replace(tzinfo=None)
                ax.axvspan(left, right,
                           ymin=0, ymax=1, color='grey', alpha=tw_right[1])

    # Invert y-axis and set limits.
    y_lim = ax.get_ylim()
    if y_lim[1] > y_lim[0]:
        ax.invert_yaxis()
    ax.set_ylim([max_airmass, min_airmass])

    # Draw lo/hi limit regions, if present
    ymax, ymin = ax.get_ylim()       # should be (hi_limit, lo_limit)

    if max_region is not None:
        ax.axhspan(ymax, max_region, facecolor='#F9EB4E', alpha=0.10)
    if min_region is not None:
        ax.axhspan(min_region, ymin, facecolor='#F9EB4E', alpha=0.10)

    # Set labels.
    ax.set_ylabel("Airmass")
    ax.set_xlabel("Time from {0} [{1}]".format(min(timetoplot).datetime.date(), tzname))

    if altitude_yaxis and not _has_twin(ax):
        altitude_ticks = np.array([90, 60, 50, 40, 30, 20])
        airmass_ticks = 1./np.cos(np.radians(90 - altitude_ticks))

        ax2 = ax.twinx()
        ax2.invert_yaxis()
        ax2.set_yticks(airmass_ticks)
        ax2.set_yticklabels(altitude_ticks)
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('Altitude [degrees]')

    # Redraw figure for interactive sessions.
    ax.figure.canvas.draw()

    # Output.
    return ax


def plot_altitude(targets, observer, time, ax=None, style_kwargs=None,
                  style_sheet=None, brightness_shading=False,
                  airmass_yaxis=False, min_altitude=0, min_region=None,
                  max_altitude=90, max_region=None):
    r"""
    Plots altitude as a function of time for a given target.

    If a `~matplotlib.axes.Axes` object already exists, an additional
    altitude plot will be "stacked" on it.  Otherwise, creates a new
    `~matplotlib.axes.Axes` object and plots altitude on top of that.

    When a scalar `~astropy.time.Time` object is passed in (e.g.,
    ``Time('2000-1-1')``), the resulting plot will use a 24-hour window
    centered on the time indicated, with altitude sampled at regular
    intervals throughout.
    However, the user can control the exact number and frequency of altitude
    calculations used by passing in a non-scalar `~astropy.time.Time`
    object. For instance, ``Time(['2000-1-1 23:00:00', '2000-1-1
    23:30:00'])`` will result in a plot with only two altitude measurements.

    For examples with plots, visit the documentation of
    :ref:`plots_time_dependent`.

    Parameters
    ----------
    targets : list of `~astroplan.FixedTarget` objects
        The celestial bodies of interest.
        If a single object is passed it will be converted to a list.

    observer : `~astroplan.Observer`
        The person, telescope, observatory, etc. doing the observing.

    time : `~astropy.time.Time`
        If scalar (e.g., ``Time('2000-1-1')``), will result in plotting target
        altitudes once an hour over a 24-hour window.
        If non-scalar (e.g., ``Time(['2000-1-1'])``, ``[Time('2000-1-1')]``,
        ``Time(['2000-1-1', '2000-1-2'])``),
        will result in plotting data at the exact times specified.

    ax : `~matplotlib.axes.Axes` or None, optional.
        The `~matplotlib.axes.Axes` object to be drawn on.
        If None, uses the current ``Axes``.

    style_kwargs : dict or None, optional.
        A dictionary of keywords passed into `~matplotlib.pyplot.plot_date`
        to set plotting styles.

    style_sheet : dict or `None` (optional)
        matplotlib style sheet to use. To see available style sheets in
        astroplan, print *astroplan.plots.available_style_sheets*. Defaults
        to the light theme.

    brightness_shading : bool
        Shade background of plot to scale roughly with sky brightness. Dark
        shading signifies times when the sun is below the horizon. Default
        is `False`.

    altitude_yaxis : bool
        Add alternative y-axis on the right side of the figure with target
        altitude. Default is `False`.

    min_altitude : float
        Lower limit of y-axis altitude range in the plot. Default is ``1.0``.

    max_altitude : float
        Upper limit of y-axis altitude range in the plot. Default is ``3.0``.

    min_region : float
        If set, defines an interval between ``min_altitude`` and ``min_region``
        that will be shaded. Default is `None`.

    max_region : float
        If set, defines an interval between ``max_altitude`` and ``max_region``
        that will be shaded. Default is `None`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        An ``Axes`` object with added altitude vs. time plot.
    """
    # Import matplotlib, set style sheet
    if style_sheet is not None:
        _set_mpl_style_sheet(style_sheet)

    import matplotlib.pyplot as plt
    from matplotlib import dates

    # Set up plot axes and style if needed.
    if ax is None:
        ax = plt.gca()
    if style_kwargs is None:
        style_kwargs = {}
    style_kwargs = dict(style_kwargs)
    if 'ls' not in style_kwargs and 'fmt' not in style_kwargs:
        style_kwargs.setdefault('linestyle', '-')
    if 'lw' not in style_kwargs:
        style_kwargs.setdefault('linewidth', 1.5)

    # Populate time window if needed.
    time = Time(time)
    if time.isscalar:
        time = time + np.linspace(-12, 12, 100)*u.hour
    elif len(time) == 1:
        warnings.warn('You used a Time array of length 1.  You probably meant '
                      'to use a scalar. (Or maybe a list with length > 1?).',
                      PlotWarning)

    if not isinstance(targets, Sequence):
        targets = [targets]

    for target in targets:
        # Calculate airmass
        altitude = observer.altaz(time, target).alt
        # Mask out nonsense airmasses
        masked_altitude = np.ma.array(altitude, mask=altitude < 0)

        # Some checks & info for labels.
        try:
            target_name = target.name
        except AttributeError:
            target_name = ''

        # Plot data
        ax.plot_date(time.plot_date, masked_altitude, label=target_name, **style_kwargs)

    # Format the time axis
    ax.set_xlim([time[0].plot_date, time[-1].plot_date])
    date_formatter = dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Shade background during night time
    if brightness_shading:
        start = time[0].datetime

        # Calculate and order twilights and set plotting alpha for each
        twilights = [
            (observer.sun_set_time(Time(start), which='next').datetime, 0.0),
            (observer.twilight_evening_civil(Time(start), which='next').datetime, 0.1),
            (observer.twilight_evening_nautical(Time(start), which='next').datetime, 0.2),
            (observer.twilight_evening_astronomical(Time(start), which='next').datetime, 0.3),
            (observer.twilight_morning_astronomical(Time(start), which='next').datetime, 0.4),
            (observer.twilight_morning_nautical(Time(start), which='next').datetime, 0.3),
            (observer.twilight_morning_civil(Time(start), which='next').datetime, 0.2),
            (observer.sun_rise_time(Time(start), which='next').datetime, 0.1),
        ]

        twilights.sort(key=operator.itemgetter(0))
        for i, twi in enumerate(twilights[1:], 1):
            ax.axvspan(twilights[i - 1][0], twilights[i][0],
                       ymin=0, ymax=1, color='grey', alpha=twi[1])

    # Invert y-axis and set limits.
    # y_lim = ax.get_ylim()
    # if y_lim[1] > y_lim[0]:
    #     ax.invert_yaxis()
    ax.set_ylim([min_altitude, max_altitude])

    # Draw lo/hi limit regions, if present
    ymax, ymin = ax.get_ylim()       # should be (hi_limit, lo_limit)

    if max_region is not None:
        ax.axhspan(ymax, max_region, facecolor='#F9EB4E', alpha=0.10)
    if min_region is not None:
        ax.axhspan(min_region, ymin, facecolor='#F9EB4E', alpha=0.10)

    # Set labels.
    ax.set_ylabel("Altitude")
    ax.set_xlabel("Time from {0} [UTC]".format(min(time).datetime.date()))

    if airmass_yaxis and not _has_twin(ax):
        # altitude_ticks = np.array([90, 60, 50, 40, 30, 20])
        # airmass_ticks = 1./np.cos(np.radians(90 - altitude_ticks))

        airmass_ticks = np.array([1, 2, 3])
        altitude_ticks = 90 - np.degrees(np.arccos(1/airmass_ticks))

        ax2 = ax.twinx()
        # ax2.invert_yaxis()
        ax2.set_yticks(altitude_ticks)
        ax2.set_yticklabels(airmass_ticks)
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('Airmass')

    # Redraw figure for interactive sessions.
    ax.figure.canvas.draw()

    # Output.
    return ax


def plot_schedule_airmass(schedule, show_night=False):
    """
    Plots when observations of targets are scheduled to occur superimposed
    upon plots of the airmasses of the targets.

    Parameters
    ----------
    schedule : `~astroplan.Schedule`
        a schedule object output by a scheduler
    show_night : bool
        Shades the night-time on the plot

    Returns
    -------
    ax :  `~matplotlib.axes.Axes`
        An ``Axes`` object with added airmass and schedule vs. time plot.
    """
    import matplotlib.pyplot as plt
    blocks = copy.copy(schedule.scheduled_blocks)
    sorted_blocks = sorted(schedule.observing_blocks, key=lambda x: x.priority)
    targets = [block.target for block in sorted_blocks]
    ts = (schedule.start_time +
          np.linspace(0, (schedule.end_time - schedule.start_time).value, 100) * u.day)
    targ_to_color = {}
    color_idx = np.linspace(0, 1, len(targets))
    enable_brightness_shading = show_night
    # lighter, bluer colors indicate higher priority
    for target, ci in zip(set(targets), color_idx):
        plot_airmass(target, schedule.observer, ts,
                     style_kwargs=dict(color=plt.cm.cool(ci)),
                     brightness_shading=enable_brightness_shading)
        targ_to_color[target.name] = plt.cm.cool(ci)
        enable_brightness_shading = False

    for block in blocks:
        if hasattr(block, 'target'):
            plt.axvspan(block.start_time.plot_date, block.end_time.plot_date,
                        fc=targ_to_color[block.target.name], lw=0, alpha=.6)
        else:
            plt.axvspan(block.start_time.plot_date, block.end_time.plot_date,
                        color='k')
    plt.axhline(3, color='k', label='Transitions')
    # TODO: make this output a `axes` object


def plot_parallactic(target, observer, time, ax=None, style_kwargs=None,
                     style_sheet=None):
    """
    Plots parallactic angle as a function of time for a given target.

    If a `~matplotlib.axes.Axes` object already exists, an additional
    parallactic angle plot will be "stacked" on it.  Otherwise, creates a
    new `~matplotlib.axes.Axes` object and plots on top of that.

    When a scalar `~astropy.time.Time` object is passed in (e.g.,
    ``Time('2000-1-1')``), the resulting plot will use a 24-hour window
    centered on the time indicated, with parallactic angle sampled at
    regular intervals throughout.
    However, the user can control the exact number and frequency of parallactic
    angle calculations used by passing in a non-scalar `~astropy.time.Time`
    object. For instance, ``Time(['2000-1-1 23:00:00', '2000-1-1 23:30:00'])``
    will result in a plot with only two parallactic angle measurements.

    For examples with plots, visit the documentation of
    :ref:`plots_time_dependent`.

    Parameters
    ----------
    target : `~astroplan.FixedTarget`
        The celestial body of interest.

    observer : `~astroplan.Observer`
        The person, telescope, observatory, etc. doing the observing.

    time : `~astropy.time.Time`
        If scalar (e.g., ``Time('2000-1-1')``), will result in plotting target
        parallactic angle once an hour over a 24-hour window.
        If non-scalar (e.g., ``Time(['2000-1-1'])``, ``[Time('2000-1-1')]``,
        ``Time(['2000-1-1', '2000-1-2'])``),
        will result in plotting data at the exact times specified.

    ax : `~matplotlib.axes.Axes` or None, optional.
        The ``Axes`` object to be drawn on.
        If None, uses the current ``Axes``.

    style_kwargs : dict or None, optional.
        A dictionary of keywords passed into `~matplotlib.pyplot.plot_date`
        to set plotting styles.

    style_sheet : dict or `None` (optional)
        matplotlib style sheet to use. To see available style sheets in
        astroplan, print *astroplan.plots.available_style_sheets*. Defaults
        to the light theme.

    Returns
    -------
    ax :  `~matplotlib.axes.Axes`
        An ``Axes`` object with added parallactic angle vs. time plot.

    """
    # Import matplotlib, set style sheet
    if style_sheet is not None:
        _set_mpl_style_sheet(style_sheet)

    import matplotlib.pyplot as plt

    from matplotlib import dates

    # Set up plot axes and style if needed.
    if ax is None:
        ax = plt.gca()
    if style_kwargs is None:
        style_kwargs = {}
    style_kwargs = dict(style_kwargs)
    style_kwargs.setdefault('fmt', '-')
    if 'ls' not in style_kwargs and 'fmt' not in style_kwargs:
        style_kwargs.setdefault('linestyle', '-')

    # Populate time window if needed.
    time = Time(time)
    if time.isscalar:
        time = time + np.linspace(-12, 12, 100)*u.hour
    elif len(time) == 1:
        warnings.warn('You used a Time array of length 1.  You probably meant '
                      'to use a scalar. (Or maybe a list with length > 1?).',
                      PlotWarning)

    # Calculate parallactic angle.
    p_angle = observer.parallactic_angle(time, target)

    # Some checks & info for labels.
    assert len(time) == len(p_angle)

    if not hasattr(target, 'name'):
        target_name = ''
    else:
        target_name = target.name
    style_kwargs.setdefault('label', target_name)

    # Plot data.
    ax.plot_date(time.plot_date, p_angle, **style_kwargs)

    # Format the time axis
    date_formatter = dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Set labels.
    ax.set_ylabel("Parallactic Angle - Radians")
    ax.set_xlabel("Time from {0} [UTC]".format(min(time).datetime.date()))

    # Redraw figure for interactive sessions.
    ax.figure.canvas.draw()

    return ax
