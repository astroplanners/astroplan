# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from ._version import version as __version__
except ImportError:
    __version__ = ''

from .utils import *
from .observer import *
from .target import *
from .exceptions import *
from .moon import *
from .constraints import *
from .scheduling import *
from .periodic import *

download_IERS_A()
