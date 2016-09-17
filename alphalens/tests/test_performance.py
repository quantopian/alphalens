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


def setup_factor(dr, tickers, size=4, data=range(1,5)):
    assert size <= 4, "Cannot create factor larger than 4 yet"
    factor = (DataFrame(index=dr, columns=tickers[0:size],
                        data=[data,
                              list(reversed(data))])
              .stack())
    factor.index = factor.index.set_names(['date', 'asset'])
    factor.name = 'factor'
    factor = factor.reset_index()
    factor['group'] = [1, 1, 2, 2][0:size] * 2
    factor = factor.set_index(['date', 'asset', 'group']).factor
    return factor


class PerformanceTestCase(TestCase):
    dr = date_range(start='2015-1-1', end='2015-1-2')
    dr.name = 'date'
    tickers = ['A', 'B', 'C', 'D']
    factor = setup_factor(dr, tickers, size=4)

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
                            [1, 2, 1, 2, 2, 1, 2, 1]),
                           (setup_factor(dr, tickers, size=1, data=[1]),
                            4, False,
                            [1] * 2),
                           (setup_factor(dr, tickers, size=4, data=[1] * 4),
                            4, False,
                            [2] * 8)
                           ])
    def test_quantize_factor(self, factor, quantiles, by_group, expected_vals):
        quantized_factor = quantize_factor(factor,
                                           quantiles=quantiles,
                                           by_group=by_group)
        expected = Series(index=factor.index,
                          data=expected_vals,
                          name='quantile')
        assert_series_equal(quantized_factor, expected)

    @parameterized.expand([(setup_factor(dr, tickers, size=4, data=[1,1,2,3]),
                            4, False)
                           ])
    def test_quantize_factor_exception(self, factor, quantiles, by_group):
        """
        This documents an case when we meet confusing boundary

        Some solution discussed at:
        - http://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
        - https://github.com/pydata/pandas/issues/7751#issue-37814702

        """
        with self.assertRaises(ValueError):
            quantized_factor = quantize_factor(factor,
                                               quantiles=quantiles,
                                               by_group=by_group)

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
