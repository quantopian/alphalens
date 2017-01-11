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
                            mean_return_by_quantile,
                            average_cumulative_return_by_quantile)

from .. utils import (get_clean_factor_and_forward_returns)

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

    @parameterized.expand([(factor, 4, None, False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor, 2, None, False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor, 2, None, True,
                            [1, 2, 1, 2, 2, 1, 2, 1]),
                           (factor, [0, .25, .5, .75, 1.], None, False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor, [0, .5, .75, 1.], None, False,
                            [1, 1, 2, 3, 3, 2, 1, 1]),
                           (factor, [0, .25, .5, 1.], None, False,
                            [1, 2, 3, 3, 3, 3, 2, 1]),
                           (factor, [0, .5, 1.], None, False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor, [.25, .5, .75], None, False,
                            [nan, 1, 2, nan, nan, 2, 1, nan]),
                           (factor, [0, .5, 1.], None, True,
                            [1, 2, 1, 2, 2, 1, 2, 1]),
                           (factor, [.5, 1.], None, True,
                            [nan, 1, nan, 1, 1, nan, 1, nan]),
                           (factor, [0, 1.], None, True,
                            [1, 1, 1, 1, 1, 1, 1, 1]),
                           (factor, None, 4, False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor, None, 2, False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor, None, 3, False,
                            [1, 1, 2, 3, 3, 2, 1, 1]),
                           (factor, None, 8, False,
                            [1, 3, 6, 8, 8, 6, 3, 1]),
                           (factor, None, [0, 1, 2, 3, 5], False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor, None, [1, 2, 3], False,
                            [nan, 1, 2, nan, nan, 2, 1, nan]),
                           (factor, None, [0, 2, 5], False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor, None, [0.5, 2.5, 4.5], False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor, None, [0.5, 2.5], True,
                            [1, 1, nan, nan, nan, nan, 1, 1]),
                           (factor, None, 2, True,
                            [1, 2, 1, 2, 2, 1, 2, 1])])
    def test_quantize_factor(self, factor, quantiles, bins, by_group,
                             expected_vals):
        quantized_factor = quantize_factor(factor,
                                           quantiles=quantiles,
                                           bins=bins,
                                           by_group=by_group)
        expected = Series(index=factor.index,
                          data=expected_vals,
                          name='quantile').dropna()
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

    @parameterized.expand([([1, 2, 3, 4, 4, 3, 2, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            [-0.5, -0.5]),
                           ([1, 2, 3, 4, 1, 2, 3, 4],
                            [1, 4, 1, 2, 1, 2, 2, 1],
                            [1.0, 0.0]),                            
                           ([1, 1, 1, 1, 1, 1, 1, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            [nan, nan])])
    def test_factor_group_neutral_returns(self, factor_vals, fwd_return_vals,
                                          expected_vals):
        factor = Series(index=self.factor.index, data=factor_vals)
        fwd_return_df = DataFrame(index=self.factor.index,
                                  columns=[1], data=fwd_return_vals)
        factor_returns_s = factor_returns(factor, fwd_return_df, group_neutral=True)
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

    @parameterized.expand([(1, 2, False, 4,
                            [[ 1.00,  0.0,  -0.50,  -0.75],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ 0.00,  0.00,  0.00,  0.00],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ -0.20,  0.0,  0.25,  0.5625],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ -0.3333333,  0.0,  0.50,  1.25],
                             [ 0.0,  0.0,  0.0,  0.0]]),
                           (1, 2, True, 4,
                            [[ 0.8833333,  0.0, -0.5625, -1.015625 ],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [-0.1166667,  0.0, -0.0625, -0.265625],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [-0.3166667,  0.0,  0.1875,  0.296875],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [-0.4500000,  0.0,  0.4375,  0.984375],
                             [ 0.0,  0.0,  0.0,  0.0]]),
                           (3, 0, False, 4,
                            [[ 7.0,  3.0,  1.0,  0.0],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [-0.488, -0.36, -0.2,  0.0],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [-0.703704, -0.55555555, -0.333333333,  0.0],
                             [ 0.0,  0.0,  0.0,  0.0]]),
                           (0, 3, True, 4,
                            [[ 0.0, -0.5625, -1.015625, -1.488281],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ 0.0, -0.0625, -0.265625, -0.613281],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ 0.0, 0.1875, 0.296875, 0.339844],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ 0.0, 0.4375, 0.984375, 1.761719],
                             [ 0.0,  0.0,  0.0,  0.0]]),
                           (3, 3, False, 2,
                            [[ 3.5,  1.5,  0.5,  0.0, -0.25, -0.375,  -0.4375],
                             [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                             [-0.595852, -0.457778, -0.266667,  0.0,  0.375,  0.90625,   1.664062],
                             [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]]),
                           (3, 3, True, 2,
                            [[ 2.047926,  0.978888,  0.383333,  0.0, -0.3125, -0.640625, -1.050781],
                             [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                             [-2.047926, -0.978888, -0.383333,  0.0,  0.3125,  0.640625,  1.050781],
                             [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]]),
                          ])
    def test_average_cumulative_return_by_quantile(self, before, after,
                                                   demeaned, quantiles,
                                                   expected_vals):
        dr = date_range(start='2015-1-15', end='2015-2-1')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']
        r1, r2, r3, r4 = (1.25, 1.50, 1.00, 0.50)
        data=[[r1**i,  r2**i,  r3**i,  r4**i] for i in range(1, 19)]
        prices = DataFrame(index=dr, columns=tickers, data=data)
        dr2 = date_range(start='2015-1-21', end='2015-1-26')
        factor = DataFrame(index=dr2, columns=tickers,
                           data=[[3, 4, 2, 1], [3, 4, 2, 1], [3, 4, 2, 1],
                                 [3, 4, 2, 1], [3, 4, 2, 1], [3, 4, 2, 1]]).stack()

        factor, forward_returns = get_clean_factor_and_forward_returns(factor,
                                                                       prices,
                                                                       periods=range(0,after+1),
                                                                       filter_zscore=False)
        quantile_factor = quantize_factor(factor, by_group=False, quantiles=quantiles)
        avgrt = average_cumulative_return_by_quantile(quantile_factor, prices, before, after, demeaned)
        arrays = []
        for q in range(1, quantiles+1):
            arrays.append((q, 'mean'))
            arrays.append((q, 'std'))
        index = MultiIndex.from_tuples(arrays, names=['quantile', None])
        expected = DataFrame(index=index, columns=range(-before,after+1), data=expected_vals)
        assert_frame_equal(avgrt, expected)

    @parameterized.expand([(0, 2, False, 4,
                            [[ 0.0,  -0.50, -0.75],
                             [ 0.0,   0.0,    0.0],
                             [ 0.0,   0.0,    0.0],
                             [ 0.0,   0.0,    0.0],
                             [ 0.0,   0.25,   0.5625],
                             [ 0.0,   0.0,    0.0],
                             [ 0.0,   0.50,   1.25],
                             [ 0.0,   0.0,    0.0]]),
                           (0, 3, True, 4,
                            [[ 0.0, -0.5625, -1.015625, -1.488281],
                             [ 0.0,  0.0,     0.0,       0.0],
                             [ 0.0, -0.0625, -0.265625, -0.613281],
                             [ 0.0,  0.0,     0.0,       0.0],
                             [ 0.0,  0.1875,  0.296875,  0.339844],
                             [ 0.0,  0.0,     0.0,       0.0],
                             [ 0.0,  0.4375,  0.984375,  1.761719],
                             [ 0.0,  0.0,     0.0,       0.0]]),
                           (0, 3, False, 2,
                            [[ 0.0, -0.25, -0.375,  -0.4375],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ 0.0,  0.375,  0.90625,   1.664062],
                             [ 0.0,  0.0,  0.0,  0.0]]),
                           (0, 3, True, 2,
                            [[ 0.0, -0.3125, -0.640625, -1.050781],
                             [ 0.0,  0.0,  0.0,  0.0],
                             [ 0.0,  0.3125,  0.640625,  1.050781],
                             [ 0.0,  0.0,  0.0,  0.0]]),
                          ])
    def test_average_cumulative_return_by_quantile_2(self, before, after,
                                                   demeaned, quantiles,
                                                   expected_vals):
        """
        Test varying factor asset universe: at differnt dates there might be
        different assets
        """
        dr = date_range(start='2015-1-15', end='2015-1-25')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D', 'E', 'F']
        r1, r2, r3, r4 = (1.25, 1.50, 1.00, 0.50)
        data=[[r1**i,  r2**i,  r3**i,  r4**i,  r2**i,  r3**i] for i in range(1, 12)]
        prices = DataFrame(index=dr, columns=tickers, data=data)
        dr2 = date_range(start='2015-1-18', end='2015-1-21')
        factor = DataFrame(index=dr2, columns=tickers,
                           data=[[3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                                 [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2]]).stack()

        factor, forward_returns = get_clean_factor_and_forward_returns(factor,
                                                                       prices,
                                                                       periods=range(0,after+1),
                                                                       filter_zscore=False)
        quantile_factor = quantize_factor(factor, by_group=False, quantiles=quantiles)
        avgrt = average_cumulative_return_by_quantile(quantile_factor, prices, before, after, demeaned)
        arrays = []
        for q in range(1, quantiles+1):
            arrays.append((q, 'mean'))
            arrays.append((q, 'std'))
        index = MultiIndex.from_tuples(arrays, names=['quantile', None])
        expected = DataFrame(index=index, columns=range(-before,after+1), data=expected_vals)
        assert_frame_equal(avgrt, expected)
