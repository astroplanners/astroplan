# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

__all__ = ['plot_finder_image']


@u.quantity_input(fov_radius=u.deg)
def plot_finder_image(target, survey='DSS', fov_radius=10*u.arcmin,
                      log=False, ax=None, grid=False, reticle=False,
                      style_kwargs=None, reticle_style_kwargs=None,
                      skyview_kwargs={}
                     ):
    """
    Plot survey image centered on ``target``.

    Survey images are retrieved from NASA Goddard's SkyView service via
    ``astroquery.skyview.SkyView``.

    If a `~matplotlib.axes.Axes` object already exists, plots the finder image
    on top. Otherwise, creates a new `~matplotlib.axes.Axes`
    object with the finder image.

    Parameters
    ----------
    target : `~astroplan.FixedTarget`, `~astropy.coordinates.SkyCoord`
        Coordinates of celestial object

    survey : string
        Name of survey to retrieve image from. For dictionary of
        available surveys, use
        ``from astroquery.skyview import SkyView; SkyView.list_surveys()``.
        Defaults to ``'DSS'``, the Digital Sky Survey.

    fov_radius : `~astropy.units.Quantity`
        Radius of field of view of retrieved image. Defaults to 10 arcmin.

    log : bool, optional
        Take the natural logarithm of the FITS image if `True`.
        False by default.

    ax : `~matplotlib.axes.Axes` or None, optional.
        The `~matplotlib.axes.Axes` object to be drawn on.
        If None, uses the current `~matplotlib.axes.Axes`.

    grid : bool, optional.
        Grid is drawn if `True`. `False` by default.

    reticle : bool, optional
        Draw reticle on the center of the FOV if `True`. Default is `False`.

    style_kwargs : dict or `None`, optional.
        A dictionary of keywords passed into `~matplotlib.pyplot.imshow`
        to set plotting styles.

    reticle_style_kwargs : dict or `None`, optional
        A dictionary of keywords passed into `~matplotlib.pyplot.axvline` and
        `~matplotlib.pyplot.axhline` to set reticle style.

    skyview_kwargs : dict or `None`, optional.
        A dictionary of keywords passed into
        `~astroquery.skyview.SkyView.get_images` to set parameters of the
        retrieved image.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes with survey image centered on ``target``

    hdu : `~astropy.io.fits.PrimaryHDU`
        FITS HDU of the retrieved image


    Notes
    -----
    Dependencies:
        In addition to Matplotlib, this function makes use of astroquery.
    """

    import matplotlib.pyplot as plt
    from astroquery.skyview import SkyView

    coord = target if not hasattr(target, 'coord') else target.coord
    position = coord.icrs
    coordinates = 'icrs'
    target_name = None if isinstance(target, SkyCoord) else target.name

    hdu = SkyView.get_images(position=position, coordinates=coordinates,
                             survey=survey, radius=fov_radius,
                             **skyview_kwargs)[0][0]
    wcs = WCS(hdu.header)

    # Set up axes & plot styles if needed.
    if ax is None:
        ax = plt.gcf().add_subplot(projection=wcs)
    if style_kwargs is None:
        style_kwargs = {}
    style_kwargs = dict(style_kwargs)
    style_kwargs.setdefault('cmap', 'Greys')
    style_kwargs.setdefault('origin', 'lower')

    if log:
        image_data = np.log(hdu.data)
    else:
        image_data = hdu.data
    ax.imshow(image_data, **style_kwargs)

    # Draw reticle
    if reticle:
        pixel_width = image_data.shape[0]
        inner, outer = 0.03, 0.08

        if reticle_style_kwargs is None:
            reticle_style_kwargs = {}
        reticle_style_kwargs.setdefault('linewidth', 2)
        reticle_style_kwargs.setdefault('color', 'm')

        ax.axvline(x=0.5*pixel_width, ymin=0.5+inner, ymax=0.5+outer,
                   **reticle_style_kwargs)
        ax.axvline(x=0.5*pixel_width, ymin=0.5-inner, ymax=0.5-outer,
                   **reticle_style_kwargs)
        ax.axhline(y=0.5*pixel_width, xmin=0.5+inner, xmax=0.5+outer,
                   **reticle_style_kwargs)
        ax.axhline(y=0.5*pixel_width, xmin=0.5-inner, xmax=0.5-outer,
                   **reticle_style_kwargs)

    # Labels, title, grid
    ax.set(xlabel='RA', ylabel='DEC')
    if target_name is not None:
        ax.set_title(target_name)
    ax.grid(grid)

    # Redraw the figure for interactive sessions.
    ax.figure.canvas.draw()
    return ax, hdu
