# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.utils.exceptions import AstropyWarning

__all__ = ["TargetAlwaysUpWarning", "TargetNeverUpWarning",
           "OldEarthOrientationDataWarning", "PlotWarning",
           "PlotBelowHorizonWarning", "AstroplanWarning",
           "MissingConstraintWarning"]


class AstroplanWarning(AstropyWarning):
    """Superclass for warnings used by astroplan"""


class TargetAlwaysUpWarning(AstroplanWarning):
    """Target is circumpolar"""
    pass


class TargetNeverUpWarning(AstroplanWarning):
    """Target never rises above horizon"""
    pass


class OldEarthOrientationDataWarning(AstroplanWarning):
    """Using old Earth rotation data from IERS"""
    pass


class PlotWarning(AstroplanWarning):
    """Warnings dealing with the plotting aspects of astroplan"""
    pass


class PlotBelowHorizonWarning(PlotWarning):
    """Warning for when something is hidden on a plot because it's below the horizon"""
    pass


class MissingConstraintWarning(AstroplanWarning):
    """Triggered when a constraint is expected but not supplied"""
    pass


class InvalidTLEDataWarning(AstroplanWarning):
    """TLE data invalid for the requested time"""
    pass
