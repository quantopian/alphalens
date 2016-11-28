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
                            factor_returns, factor_alpha_beta,
                            average_cumulative_return_by_quantile)


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
    factor['group'] = [1, 1, 2, 2, 1, 1, 2, 2]
    factor = factor.set_index(['date', 'asset', 'group']).factor

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
                                [dr, [1, 2]], names=['date', 'group']),
                            [1., 1., 1., 1.],
                            ),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            True, True,
                            MultiIndex.from_product(
                                [dr, [1, 2]], names=['date', 'group']),
                            [1., 1., 1., 1.],
                            )])
    def test_information_coefficient(self, factor, fr,
                                     group_adjust, by_group,
                                     expected_ix, expected_ic_val):
        fr_df = DataFrame(index=self.factor.index, columns=[1], data=fr)

        ic = factor_information_coefficient(
            factor, fr_df, group_adjust=group_adjust, by_group=by_group)

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
                            Int64Index([1, 2], name='group'),
                            [1., 1.],
                            ),
                           (factor, [1, 2, 3, 4, 4, 3, 2, 1],
                            'W', True,
                            MultiIndex.from_product(
                                [DatetimeIndex(['2015-01-04'],
                                               name='date', freq='W-SUN'),
                                 [1, 2]], names=['date', 'group']),
                            [1., 1.],
                            )])
    def test_mean_information_coefficient(self, factor, fr,
                                          by_time, by_group,
                                          expected_ix, expected_ic_val):
        fr_df = DataFrame(index=self.factor.index, columns=[1], data=fr)

        ic = mean_information_coefficient(
            factor, fr_df, group_adjust=False, by_time=by_time,
            by_group=by_group)

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
    def test_quantize_factor(self, factor, quantiles, by_group, expected_vals):
        quantized_factor = quantize_factor(factor,
                                           quantiles=quantiles,
                                           by_group=by_group)
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
        expected.name = test_quantile

        assert_series_equal(to, expected)


    @parameterized.expand([([1, 2, 3, 4, 4, 3, 2, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            [-1.25000, -1.25000]),
                           ([1, 1, 1, 1, 1, 1, 1, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            [nan, nan])])
    def test_factor_returns(self, factor_vals, fwd_return_vals, expected_vals):
        factor = Series(index=self.factor.index, data=factor_vals)

        fwd_return_df = DataFrame(index=self.factor.index,
                                  columns=[1], data=fwd_return_vals)

        factor_returns_s = factor_returns(factor, fwd_return_df)
        expected = DataFrame(index=self.dr, data=expected_vals, columns=[1])

        assert_frame_equal(factor_returns_s, expected)

    @parameterized.expand([([1, 2, 3, 4, 1, 1, 1, 1],
                            [3.5, 2.0],
                            (2.**252)-1, 1.)])
    def test_factor_alpha_beta(self, fwd_return_vals, factor_returns_vals,
                               alpha, beta):
        factor_returns = Series(index=self.dr, data=factor_returns_vals)
        fwd_return_df = DataFrame(index=self.factor.index,
                                  columns=[1], data=fwd_return_vals)

        ab = factor_alpha_beta(None, fwd_return_df,
                               factor_returns=factor_returns)

        expected = DataFrame(columns=[1],
                             index=['Ann. alpha', 'beta'],
                             data=[alpha, beta])

        assert_frame_equal(ab, expected)


    @parameterized.expand([([[1.0, 2.0, 3.0, 4.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [1.0, 2.0, 3.0, 4.0]],
                            [1, 1, 2, 2, 1, 1, 2,
                             2, 1, 1, 2, 2, 1, 1, 2, 2],
                            '2015-1-4',
                            1,
                            False,
                            [nan, 1.0, 1.0, 1.0]),
                           ([[4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 4.0]],
                            [1, 1, 2, 2, 1, 1, 2,
                             2, 1, 1, 2, 2, 1, 1, 2, 2],
                            '2015-1-4',
                            1,
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
                            3,
                            False,
                            [nan, nan, nan, 1.0, 1.0,
                             1.0, 0.6, -0.6, -1.0, 1.0,
                             -0.6, -1.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                            [2.0, 1.0, 4.0, 3.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 4.0]],
                            [1, 1, 2, 2, 1, 1, 2,
                             2, 1, 1, 2, 2, 1, 1, 2, 2],
                            '2015-1-4',
                            1,
                            True,
                            [nan, -1.0, 1.0, -1.0])
                           ])
    def test_factor_rank_autocorrelation(self, factor_values,
                                         group_values, end_date,
                                         period, by_group,
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
        factor['group'] = group_values
        factor = factor.set_index(['date', 'asset', 'group']).factor

        fa = factor_rank_autocorrelation(factor, period, by_group)
        expected = Series(index=fa.index, data=expected_vals)
        expected.name = period

        assert_series_equal(fa, expected)

    @parameterized.expand([(1, 2, False,
                            [[ 1.640404,  0.0,  0.505051,  1.640404],
                             [ 1.673481,  0.0,  0.448154,  1.673481],
                             [ 0.621212,  0.0,  0.414141,  0.681818],
                             [ 0.484612,  0.0,  0.406375,  0.635145],
                             [-0.848485,  0.0, -1.121212, -0.810606],
                             [ 0.341981,  0.0,  1.373751,  0.390464],
                             [-2.060606,  0.0, -2.266667, -2.000000],
                            [ 1.473503,  0.0,  2.212841,  1.523155]]),
                           (1, 2, True,
                            [[ 2.568150,  0.0, -0.522909, -1.405378],
                             [ 2.577523,  0.0,  0.849970,  2.183635],
                             [ 0.955398,  0.0, -0.215333, -0.201407],
                             [ 0.783811,  0.0,  0.499537,  0.596399],
                             [ 0.109943,  0.0, -1.600213, -1.651631],
                             [ 0.514246,  0.0,  1.696150,  3.348696],
                             [-0.810133,  0.0, -3.256179, -3.193612],
                            [ 1.292986,  0.0,  2.282362,  4.484187]]),
                           (3, 0, False,
                            [[ 0.048485,  1.430303,  0.606061,  0.0],
                             [ 0.193845,  1.760768,  0.721950,  0.0],
                             [ 0.030303,  0.530303,  0.462121,  0.0],
                             [ 0.121153,  0.483144,  0.552414,  0.0],
                             [-0.151515, -0.439394, -1.085859,  0.0],
                             [ 0.618527,  1.058873,  0.717007,  0.0],
                             [-0.181818, -1.094949, -2.044444,  0.0],
                             [ 0.768706,  2.390434,  1.359602,  0.0]]),
                           (0, 3, True,
                            [[0.0,  0.526799, -1.364039,  0.229906],
                             [0.0,  1.320868,  0.908440,  1.758385],
                             [0.0, -0.159943, -0.499266,  0.120580],
                             [0.0,  0.488146,  0.581755,  1.068258],
                             [0.0, -0.905398, -2.386166, -3.050701],
                             [0.0,  0.850031,  1.231573,  2.066495],
                             [0.0, -2.284817, -4.168521, -6.615322],
                             [0.0,  2.894711,  1.550110,  2.999847]]),
                          ])
    def test_average_cumulative_return_by_quantile(self, before, after,
                                                  demeaned, expected_vals):
        dr = date_range(start='2015-1-1', end='2015-2-2')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']
        prices = DataFrame(index=dr, columns=tickers,
                           data=[[1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5],
                                 [1, 1, 1, 1], [0.5, 0.2, 3, 3], [0.5, 0.2, 4, 5]])
        factor = DataFrame(index=dr, columns=tickers,
                           data=[[3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4],
                                 [3, 4, 2, 1], [2, 1, 3, 4], [2, 1, 3, 4]]).stack()
        factor.index = factor.index.set_names(['date', 'asset'])
        factor.name = 'factor'
        factor = factor.reset_index()
        factor['group'] = [2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2,
                           2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2]
        factor = factor.set_index(['date', 'asset', 'group']).factor

        avgrt = average_cumulative_return_by_quantile(factor, prices, before, after, demeaned)
        arrays = [[1,1,2,2,3,3,4,4],['mean','std','mean','std','mean','std','mean','std']]
        index = MultiIndex.from_tuples(list(zip(*arrays)), names=['factor', None])
        expected = DataFrame(index=index, columns=range(-before,after+1), data=expected_vals)
        assert_frame_equal(avgrt, expected)

