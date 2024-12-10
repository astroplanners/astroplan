# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
from astropy.time import Time

from astroplan.utils import stride_array, time_grid_from_range

arr10 = np.arange(10)


def test_stride_array():
    stride10by5 = stride_array(arr10, 5)
    assert len(stride10by5) == 6


def test_stride_floats():
    arr_float = np.asarray(arr10, float)
    stride_array(arr_float, 3)  # stride 10 by 3


def test_time_grid_from_range():
    times2 = ['2010-01-01 00:00:00', '2010-01-01 01:00:10']
    times3 = ['2010-01-01 00:00:00', '2010-01-01 01:00:00',
              '2010-01-01 03:00:00']
    tgrid = time_grid_from_range(Time(times2))
    assert len(tgrid) == 3
    assert np.all(tgrid.iso == Time(['2010-01-01 00:00:00',
                                     '2010-01-01 00:30:00',
                                     '2010-01-01 01:00:00']))

    with pytest.raises(ValueError):
        time_grid_from_range(Time(times3))
