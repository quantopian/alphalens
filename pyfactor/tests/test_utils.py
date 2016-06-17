from __future__ import division
from unittest import TestCase
from nose_parameterized import parameterized
from numpy import (nan)

from pandas import (
    Series,
    DataFrame,
    date_range,
    datetime,
    Panel,
    MultiIndex,
)
from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal)

from utils import (compute_forward_price_movement,
                   sector_adjust_forward_price_moves)


class UtilsTestCase(TestCase):

    def test_compute_forward_price_movement(self):
        dr = date_range(start='2015-1-1', end='2015-1-3')
        prices = DataFrame(index=dr, columns=['A', 'B'],
                           data=[[1, 1], [1, 2], [2, 1]])

        fp = compute_forward_price_movement(prices, days=[1, 2])

        ix = MultiIndex.from_product([dr, ['A', 'B']],
                                     names=['date', 'equity'])
        expected = DataFrame(index=ix, columns=[1, 2])
        expected[1] = [0., 1., 1., -0.5, nan, nan]
        expected[2] = [1., 0., nan, nan, nan, nan]

        assert_frame_equal(fp, expected)
