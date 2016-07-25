#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


from .. performance import (factor_information_coefficient,
                            mean_information_coefficient,
                            quantize_factor,
                            quantile_turnover,
                            factor_rank_autocorrelation,
                            factor_returns, factor_alpha_beta)


class PerformanceTestCase(TestCase):
    dr = date_range(start='2015-1-1', end='2015-1-2')
    dr.name = 'date'
    tickers = ['A', 'B', 'C', 'D']
    factor = (DataFrame(index=dr, columns=tickers,
                        data=[[1, 2, 3, 4],
                              [4, 3, 2, 1]])
              .stack())
    factor.index = factor.index.set_names(['date', 'asset'])
    factor.name = 'factor'
    factor = factor.reset_index()
    factor['sector'] = [1, 1, 2, 2, 1, 1, 2, 2]
    factor = factor.set_index(['date', 'asset', 'sector']).factor

    @parameterized.expand([(factor, [4, 3, 2, 1, 1, 2, 3, 4],
                            False, False,
                            dr,
                            [-1., -1.],
                            ),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            False, False,
                            dr,
                            [1., 1.],
                            ),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            False, True,
                            MultiIndex.from_product(
                                [dr, [1, 2]], names=['date', 'sector']),
                            [1., 1., 1., 1.],
                            ),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            True, True,
                            MultiIndex.from_product(
                                [dr, [1, 2]], names=['date', 'sector']),
                            [1., 1., 1., 1.],
                            )])
    def test_information_coefficient(self, factor, fr,
                                     sector_adjust, by_sector,
                                     expected_ix, expected_ic_val):
        fr_df = DataFrame(index=self.factor.index, columns=[1], data=fr)

        ic = factor_information_coefficient(
            factor, fr_df, sector_adjust=sector_adjust, by_sector=by_sector)

        expected_ic_df = DataFrame(index=expected_ix,
                                   columns=Int64Index([1], dtype='object'),
                                   data=expected_ic_val)

        assert_frame_equal(ic, expected_ic_df)

    @parameterized.expand([(factor, [4, 3, 2, 1, 1, 2, 3, 4],
                            'D', False,
                            dr,
                            [-1., -1.],
                            ),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            'W', False,
                            DatetimeIndex(['2015-01-04'],
                                          name='date', freq='W-SUN'),
                            [1.],
                            ),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            None, True,
                            Int64Index([1, 2], name='sector'),
                            [1., 1.],
                            ),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            'W', True,
                            MultiIndex.from_product(
                                [DatetimeIndex(['2015-01-04'],
                                               name='date', freq='W-SUN'),
                                 [1, 2]], names=['date', 'sector']),
                            [1., 1.],
                            )])
    def test_mean_information_coefficient(self, factor, fr,
                                          by_time, by_sector,
                                          expected_ix, expected_ic_val):
        fr_df = DataFrame(index=self.factor.index, columns=[1], data=fr)

        ic = mean_information_coefficient(
            factor, fr_df, sector_adjust=False, by_time=by_time,
            by_sector=by_sector)

        expected_ic_df = DataFrame(index=expected_ix,
                                   columns=Int64Index([1], dtype='object'),
                                   data=expected_ic_val)

        assert_frame_equal(ic, expected_ic_df)

    @parameterized.expand([(factor, 4, False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor, 2, False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor, 2, True,
                            [1, 2, 1, 2, 2, 1, 2, 1])])
    def test_quantize_factor(self, factor, quantiles, by_sector, expected_vals):
        quantized_factor = quantize_factor(factor,
                                           quantiles=quantiles,
                                           by_sector=by_sector)
        expected = Series(index=factor.index,
                          data=expected_vals,
                          name='quantile')
        assert_series_equal(quantized_factor, expected)

    @parameterized.expand([([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            4.0,
                            [nan, 1.0, 1.0, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            3.0,
                            [nan, 0.0, 0.0, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0]],
                            2.0,
                            [nan, 1.0, 1.0, 1.0])])
    def test_quantile_turnover(self, quantile_values, test_quantile,
                               expected_vals):

        dr = date_range(start='2015-1-1', end='2015-1-4')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']

        quantized_test_factor = Series(DataFrame(index=dr,
                                                 columns=tickers,
                                                 data=quantile_values)
                                       .stack())
        quantized_test_factor.index = quantized_test_factor.index.set_names(
            ['date', 'asset'])

        to = quantile_turnover(quantized_test_factor, test_quantile)

        expected = Series(
            index=quantized_test_factor.index.levels[0], data=expected_vals)

        assert_series_equal(to, expected)


    @parameterized.expand([([1, 2, 3, 4, 4, 3, 2, 1],

                            [4, 3, 2, 1, 1, 2, 3, 4],
                            [-0.5, -0.5]),
                           ([1, 1, 1, 1, 1, 1, 1, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            [0., 0.])])
    def test_factor_returns(self, factor_vals, fwd_return_vals, expected_vals):
        factor = Series(index=self.factor.index, data=factor_vals)

        fwd_return_df = DataFrame(index=self.factor.index,
                                  columns=[1], data=fwd_return_vals)

        factor_returns_s = factor_returns(factor, fwd_return_df)
        expected = DataFrame(index=self.dr, data=expected_vals, columns=[1])

        assert_frame_equal(factor_returns_s, expected)

    @parameterized.expand([([1, 2, 3, 4, 1, 1, 1, 1],
                            [3.5, 2.0],
                            (2.**252)-1, nan, 1.)])
    def test_factor_alpha_beta(self, fwd_return_vals, factor_returns_vals,
                               alpha, t_stat_alpha, beta):
        factor_returns = Series(index=self.dr, data=factor_returns_vals)
        fwd_return_df = DataFrame(index=self.factor.index,
                                  columns=[1], data=fwd_return_vals)

        ab = factor_alpha_beta(None, fwd_return_df,
                               factor_daily_returns=factor_returns)

        expected = DataFrame(columns=[1],
                             index=['Ann. alpha', 't-stat(alpha)', 'beta'],
                             data=[alpha, t_stat_alpha, beta])

        assert_frame_equal(ab, expected)


    @parameterized.expand([([[1.0, 2.0, 3.0, 4.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [1.0, 2.0, 3.0, 4.0]],
                            [1, 1, 2, 2, 1, 1, 2,
                             2, 1, 1, 2, 2, 1, 1, 2, 2],
                            '2015-1-4',
                            'D',
                            False,
                            [nan, 1.0, 1.0, 1.0]),
                           ([[4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 4.0]],
                            [1, 1, 2, 2, 1, 1, 2,
                             2, 1, 1, 2, 2, 1, 1, 2, 2],
                            '2015-1-4',
                            'D',
                            False,
                           [nan, -1.0, -1.0, -1.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                            [2.0, 1.0, 4.0, 3.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [2.0, 1.0, 4.0, 3.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [2.0, 1.0, 4.0, 3.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [2.0, 1.0, 4.0, 3.0],
                            [2.0, 1.0, 4.0, 3.0],
                            [4.0, 3.0, 2.0, 1.0]],
                            [1, 1, 2, 2, 1, 1, 2, 2,
                             1, 1, 2, 2, 1, 1, 2, 2,
                             1, 1, 2, 2, 1, 1, 2, 2,
                             1, 1, 2, 2, 1, 1, 2, 2,
                             1, 1, 2, 2, 1, 1, 2, 2,
                             1, 1, 2, 2, 1, 1, 2, 2],
                            '2015-1-12',
                            'W',
                            False,
                            [nan, 0.768221, -0.400819]),
                           ([[1.0, 2.0, 3.0, 4.0],
                            [2.0, 1.0, 4.0, 3.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 4.0]],
                            [1, 1, 2, 2, 1, 1, 2,
                             2, 1, 1, 2, 2, 1, 1, 2, 2],
                            '2015-1-4',
                            'D',
                            True,
                            [nan, -1.0, 1.0, -1.0])
                           ])
    def test_factor_rank_autocorrelation(self, factor_values,
                                         sector_values, end_date,
                                         time_rule, by_sector,
                                         expected_vals):
        dr = date_range(start='2015-1-1', end=end_date)
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']
        factor_df = DataFrame(index=dr, columns=tickers, data=factor_values)\
            .stack()
        factor_df.index = factor_df.index.set_names(['date', 'asset'])

        factor = Series(factor_df)
        factor.name = 'factor'
        factor = factor.reset_index()
        factor['sector'] = sector_values
        factor = factor.set_index(['date', 'asset', 'sector']).factor

        fa = factor_rank_autocorrelation(factor, time_rule, by_sector)
        expected = Series(index=fa.index, data=expected_vals)

        assert_series_equal(fa, expected)
