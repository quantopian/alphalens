from __future__ import division
from unittest import TestCase
from nose_parameterized import parameterized

from pandas import (
    Series,
    DataFrame,
    date_range,
    datetime,
    Panel,
    Index
)
from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal)

from performance import (factor_information_coefficient)


class PerformanceTestCase(TestCase):
    dr = date_range(start='2015-1-1', end='2015-1-2')
    tickers = ['A', 'B', 'C', 'D']
    factor = (DataFrame(index=dr, columns=tickers,
                        data=[[1, 2, 3, 4],
                              [4, 3, 2, 1]])
              .stack()
              .rename_axis(['date', 'symbol']))

    @parameterized.expand([(factor, [4, 3, 2, 1, 1, 2, 3, 4],
                            None, None,
                            [-1., -1.], None),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            None, None,
                            [1., 1.], None)])
    def test_information_coefficient(self, factor, fr,
                                     time_rule, by_sector,
                                     expected_ic, expected_err):
        fr_df = DataFrame(index=self.factor.index, columns=[1], data=fr)

        ic, err = factor_information_coefficient(
            factor, fr_df, time_rule=time_rule, by_sector=by_sector)

        expected_ic_df = DataFrame(index=self.dr, columns=Index(
            [1], dtype='object'), data=expected_ic).rename_axis(['date'])

        assert_frame_equal(ic, expected_ic_df)
