from __future__ import division
from unittest import TestCase
from nose_parameterized import parameterized
from numpy import (nan, inf)
from pandas import (
    Series,
    DataFrame,
    date_range,
    datetime,
    Panel,
    Index,
    MultiIndex,
    Int64Index,
    DatetimeIndex
)
from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal)

from performance import (factor_information_coefficient,
                         mean_information_coefficient,
                         quantize_factor)


class PerformanceTestCase(TestCase):
    dr = date_range(start='2015-1-1', end='2015-1-2')
    dr.name = 'date'
    tickers = ['A', 'B', 'C', 'D']
    factor = (DataFrame(index=dr, columns=tickers,
                        data=[[1, 2, 3, 4],
                              [4, 3, 2, 1]])
              .stack()
              .rename_axis(['date', 'symbol'])).rename('factor').reset_index()
    factor['sector'] = [1, 1, 2, 2, 1, 1, 2, 2]
    factor = factor.set_index(['date', 'symbol', 'sector']).factor

    @parameterized.expand([(factor, [4, 3, 2, 1, 1, 2, 3, 4],
                            False, False,
                            dr,
                            [-1., -1.],
                            [0., 0.]),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            False, False,
                            dr,
                            [1., 1.],
                            [0., 0.]),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            False, True,
                            MultiIndex.from_product(
                                [dr, [1, 2]], names=['date', 'sector']),
                            [1., 1., 1., 1.],
                            [inf, inf, inf, inf]),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            True, True,
                            MultiIndex.from_product(
                                [dr, [1, 2]], names=['date', 'sector']),
                            [1., 1., 1., 1.],
                            [inf, inf, inf, inf])])
    def test_information_coefficient(self, factor, fr,
                                     sector_adjust, by_sector,
                                     expected_ix, expected_ic_val,
                                     expected_err_val):
        fr_df = DataFrame(index=self.factor.index, columns=[1], data=fr)

        ic, err = factor_information_coefficient(
            factor, fr_df, sector_adjust=sector_adjust, by_sector=by_sector)

        expected_ic_df = DataFrame(index=expected_ix,
                                   columns=Int64Index([1], dtype='object'),
                                   data=expected_ic_val)

        expected_err_df = DataFrame(index=expected_ix,
                                    columns=Int64Index([1], dtype='object'),
                                    data=expected_err_val)

        assert_frame_equal(ic, expected_ic_df)
        assert_frame_equal(err, expected_err_df)

    @parameterized.expand([(factor, [4, 3, 2, 1, 1, 2, 3, 4],
                            'D', False,
                            dr,
                            [-1., -1.],
                            [0., 0.]),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            'W', False,
                            DatetimeIndex(['2015-01-04'],
                                          name='date', freq='W-SUN'),
                            [1.],
                            [0.]),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            None, True,
                            Int64Index([1, 2], name='sector'),
                            [1., 1.],
                            [inf, inf]),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            'W', True,
                            MultiIndex.from_product(
                                [DatetimeIndex(['2015-01-04'],
                                               name='date', freq='W-SUN'),
                                 [1, 2]], names=['date', 'sector']),
                            [1., 1.],
                            [inf, inf])])
    def test_mean_information_coefficient(self, factor, fr,
                                          by_time, by_sector,
                                          expected_ix, expected_ic_val,
                                          expected_err_val):
        fr_df = DataFrame(index=self.factor.index, columns=[1], data=fr)

        ic, err = mean_information_coefficient(
            factor, fr_df, sector_adjust=False, by_time=by_time,
            by_sector=by_sector)

        expected_ic_df = DataFrame(index=expected_ix,
                                   columns=Int64Index([1], dtype='object'),
                                   data=expected_ic_val)

        expected_err_df = DataFrame(index=expected_ix,
                                    columns=Int64Index([1], dtype='object'),
                                    data=expected_err_val)

        assert_frame_equal(ic, expected_ic_df)
        assert_frame_equal(err, expected_err_df)


    @parameterized.expand([(factor, 4, False, [1., 2., 3., 4., 4., 3., 2., 1.]),
                           (factor, 2, False, [1., 1., 2., 2., 2., 2., 1., 1.]),
                           (factor, 2, True, [1., 2., 1., 2., 2., 1., 2., 1.])])
    def test_quantize_factor(self, factor, quantiles, by_sector, expected_vals):
        quantized_factor = quantize_factor(factor, quantiles=quantiles, by_sector=by_sector)

        expected = Series(index=factor.index, data=expected_vals, name='quantile')

        assert_series_equal(quantized_factor, expected)

















