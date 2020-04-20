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
from parameterized import parameterized
from numpy import nan
from pandas import (
    Series,
    DataFrame,
    date_range,
    MultiIndex,
    Int64Index,
    Index,
    DatetimeIndex,
    Timedelta
)

from pandas.tseries.offsets import (BDay, Day, CDay)

from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal)

from .. performance import (factor_information_coefficient,
                            mean_information_coefficient,
                            mean_return_by_quantile,
                            quantile_turnover,
                            factor_rank_autocorrelation,
                            factor_returns, factor_alpha_beta,
                            cumulative_returns, factor_weights,
                            common_start_returns,
                            average_cumulative_return_by_quantile)

from .. utils import (get_forward_returns_columns,
                      get_clean_factor_and_forward_returns)


class PerformanceTestCase(TestCase):
    dr = date_range(start='2015-1-1', end='2015-1-2')
    dr.name = 'date'
    tickers = ['A', 'B', 'C', 'D']
    factor = DataFrame(index=dr,
                       columns=tickers,
                       data=[[1, 2, 3, 4],
                             [4, 3, 2, 1]]).stack()
    factor.index = factor.index.set_names(['date', 'asset'])
    factor.name = 'factor'
    factor_data = DataFrame()
    factor_data['factor'] = factor
    factor_data['group'] = Series(index=factor.index,
                                  data=[1, 1, 2, 2, 1, 1, 2, 2],
                                  dtype="category")

    @parameterized.expand([(factor_data, [4, 3, 2, 1, 1, 2, 3, 4],
                            False, False,
                            dr,
                            [-1., -1.],
                            ),
                           (factor_data, [1, 2, 3, 4, 4, 3, 2, 1],
                            False, False,
                            dr,
                            [1., 1.],
                            ),
                           (factor_data, [1, 2, 3, 4, 4, 3, 2, 1],
                            False, True,
                            MultiIndex.from_product(
                                [dr, [1, 2]], names=['date', 'group']),
                            [1., 1., 1., 1.],
                            ),
                           (factor_data, [1, 2, 3, 4, 4, 3, 2, 1],
                            True, True,
                            MultiIndex.from_product(
                                [dr, [1, 2]], names=['date', 'group']),
                            [1., 1., 1., 1.],
                            )])
    def test_information_coefficient(self,
                                     factor_data,
                                     forward_returns,
                                     group_adjust,
                                     by_group,
                                     expected_ix,
                                     expected_ic_val):

        factor_data['1D'] = Series(index=factor_data.index,
                                   data=forward_returns)

        ic = factor_information_coefficient(factor_data=factor_data,
                                            group_adjust=group_adjust,
                                            by_group=by_group)

        expected_ic_df = DataFrame(index=expected_ix,
                                   columns=Index(['1D'], dtype='object'),
                                   data=expected_ic_val)

        assert_frame_equal(ic, expected_ic_df)

    @parameterized.expand([(factor_data,
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            False,
                            False,
                            'D',
                            dr,
                            [-1., -1.]),
                           (factor_data,
                            [1, 2, 3, 4, 4, 3, 2, 1],
                            False,
                            False,
                            'W',
                            DatetimeIndex(['2015-01-04'],
                                          name='date',
                                          freq='W-SUN'),
                            [1.]),
                           (factor_data,
                            [1, 2, 3, 4, 4, 3, 2, 1],
                            False,
                            True,
                            None,
                            Int64Index([1, 2], name='group'),
                            [1., 1.]),
                           (factor_data,
                            [1, 2, 3, 4, 4, 3, 2, 1],
                            False,
                            True,
                            'W',
                            MultiIndex.from_product(
                                [DatetimeIndex(['2015-01-04'],
                                               name='date',
                                               freq='W-SUN'),
                                 [1, 2]], names=['date', 'group']),
                            [1., 1.])])
    def test_mean_information_coefficient(self,
                                          factor_data,
                                          forward_returns,
                                          group_adjust,
                                          by_group,
                                          by_time,
                                          expected_ix,
                                          expected_ic_val):

        factor_data['1D'] = Series(index=factor_data.index,
                                   data=forward_returns)

        ic = mean_information_coefficient(factor_data,
                                          group_adjust=group_adjust,
                                          by_group=by_group,
                                          by_time=by_time)

        expected_ic_df = DataFrame(index=expected_ix,
                                   columns=Index(['1D'], dtype='object'),
                                   data=expected_ic_val)

        assert_frame_equal(ic, expected_ic_df)

    @parameterized.expand([([1.1, 1.2, 1.1, 1.2, 1.1, 1.2],
                            [[1, 2, 1, 2, 1, 2],
                             [1, 2, 1, 2, 1, 2],
                             [1, 2, 1, 2, 1, 2]],
                            2, False,
                            [0.1, 0.2]),
                           ([1.1, 1.2, 1.1, 1.2, 1.1, 1.2],
                            [[1, 2, 1, 2, 1, 2],
                             [1, 2, 1, 2, 1, 2],
                             [1, 2, 1, 2, 1, 2]],
                            2, True,
                            [0.1, 0.1, 0.2, 0.2]),
                           ([1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
                            [[1, 2, 3, 1, 2, 3],
                             [1, 2, 3, 1, 2, 3],
                             [1, 2, 3, 1, 2, 3]],
                            3, False,
                            [0.15, 0.15, 0.15]),
                           ([1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
                            [[1, 2, 3, 1, 2, 3],
                             [1, 2, 3, 1, 2, 3],
                             [1, 2, 3, 1, 2, 3]],
                            3, True,
                            [0.1, 0.2, 0.1, 0.2, 0.1, 0.2]),
                           ([1.5, 1.5, 1.2, 1.0, 1.0, 1.0],
                            [[1, 1, 2, 2, 2, 2],
                             [2, 2, 1, 2, 2, 2],
                             [2, 2, 1, 2, 2, 2]],
                            2, False,
                            [0.3, 0.15]),
                           ([1.5, 1.5, 1.2, 1.0, 1.0, 1.0],
                            [[1, 1, 3, 2, 2, 2],
                             [3, 3, 1, 2, 2, 2],
                             [3, 3, 1, 2, 2, 2]],
                            3, False,
                            [0.3, 0.0, 0.4]),
                           ([1.6, 1.6, 1.0, 1.0, 1.0, 1.0],
                            [[1, 1, 2, 2, 2, 2],
                             [2, 2, 1, 1, 1, 1],
                             [2, 2, 1, 1, 1, 1]],
                            2, False,
                            [0.2, 0.4]),
                           ([1.6, 1.6, 1.0, 1.6, 1.6, 1.0],
                            [[1, 1, 2, 1, 1, 2],
                             [2, 2, 1, 2, 2, 1],
                             [2, 2, 1, 2, 2, 1]],
                            2, True,
                            [0.2, 0.2, 0.4, 0.4])])
    def test_mean_return_by_quantile(self,
                                     daily_rets,
                                     factor,
                                     bins,
                                     by_group,
                                     expected_data):
        """
        Test mean_return_by_quantile
        """
        tickers = ['A', 'B', 'C', 'D', 'E', 'F']

        factor_groups = {'A': 1, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 2}

        price_data = [[daily_rets[0]**i, daily_rets[1]**i, daily_rets[2]**i,
                       daily_rets[3]**i, daily_rets[4]**i, daily_rets[5]**i]
                      for i in range(1, 5)]  # 4 days

        start = '2015-1-11'
        factor_end = '2015-1-13'
        price_end = '2015-1-14'  # 1D fwd returns

        price_index = date_range(start=start, end=price_end)
        price_index.name = 'date'
        prices = DataFrame(index=price_index, columns=tickers, data=price_data)

        factor_index = date_range(start=start, end=factor_end)
        factor_index.name = 'date'
        factor = DataFrame(index=factor_index, columns=tickers,
                           data=factor).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices,
            groupby=factor_groups,
            quantiles=None,
            bins=bins,
            periods=(1,))

        mean_quant_ret, std_quantile = \
            mean_return_by_quantile(factor_data,
                                    by_date=False,
                                    by_group=by_group,
                                    demeaned=False,
                                    group_adjust=False)

        expected = DataFrame(index=mean_quant_ret.index.copy(),
                             columns=mean_quant_ret.columns.copy(),
                             data=expected_data)
        expected.index.name = 'factor_quantile'

        assert_frame_equal(mean_quant_ret, expected)

    @parameterized.expand([([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 4.0, 1,
                            [nan, 1.0, 1.0, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 4.0, 1,
                            [nan, 1.0, 1.0, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 4.0, 2,
                            [nan, nan, 0.0, 1.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 4.0, 2,
                            [nan, nan, 0.0, 1.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 4.0, 3,
                            [nan, nan, nan, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 4.0, 3,
                            [nan, nan, nan, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 3.0, 1,
                            [nan, 0.0, 0.0, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 3.0, 1,
                            [nan, 0.0, 0.0, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 3.0, 2,
                            [nan, nan, 0.0, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 3.0, 2,
                            [nan, nan, 0.0, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 3.0, 3,
                            [nan, nan, nan, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 3.0, 3,
                            [nan, nan, nan, 0.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0]],
                            '1B', 2.0, 1,
                            [nan, 1.0, 1.0, 1.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0]],
                            '1D', 2.0, 1,
                            [nan, 1.0, 1.0, 1.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0]],
                            '1B', 3.0, 4,
                            [nan, nan, nan, nan,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0]],
                            '1D', 3.0, 4,
                            [nan, nan, nan, nan,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 3.0, 10,
                            [nan, nan, nan, nan, nan,
                             nan, nan, nan, nan, nan,
                             0., 1.]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 3.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 3.0, 10,
                            [nan, nan, nan, nan, nan,
                             nan, nan, nan, nan, nan,
                             0., 1.])
                           ])
    def test_quantile_turnover(self, quantile_values, freq, test_quantile,
                               period, expected_vals):

        dr = date_range(start='2015-1-1', periods=len(quantile_values),
                        freq=freq)
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']

        quantized_test_factor = Series(DataFrame(index=dr,
                                                 columns=tickers,
                                                 data=quantile_values)
                                       .stack())
        quantized_test_factor.index = quantized_test_factor.index.set_names(
            ['date', 'asset'])

        to = quantile_turnover(quantized_test_factor, test_quantile, period)

        expected = Series(
            index=quantized_test_factor.index.levels[0], data=expected_vals)
        expected.name = test_quantile

        assert_series_equal(to, expected)

    @parameterized.expand([([[3, 4,  2,  1, nan],
                             [3, 4, -2, -1, nan],
                             [3, nan, nan, 1, 4]],
                            ['A', 'B', 'C', 'D', 'E'],
                            {'A': 'Group1', 'B': 'Group2', 'C': 'Group1',
                             'D': 'Group2', 'E': 'Group1'},
                            False, False, False,
                            [0.30, 0.40, 0.20, 0.10,
                             0.30, 0.40, -0.20, -0.10,
                             0.375, 0.125, 0.50]),
                           ([[3, 4,  2,  1, nan],
                             [3, 4, -2, -1, nan],
                             [3, nan, nan, 1, 4]],
                            ['A', 'B', 'C', 'D', 'E'],
                            {'A': 'Group1', 'B': 'Group2', 'C': 'Group1',
                             'D': 'Group2', 'E': 'Group1'},
                            True, False, False,
                            [0.125, 0.375, -0.125, -0.375,
                             0.20, 0.30, -0.30, -0.20,
                             0.10, -0.50, 0.40]),
                           ([[3, 4,  2, 1, nan],
                             [-3, 4, -2, 1, nan],
                             [2, 2,  2, 3, 1]],
                            ['A', 'B', 'C', 'D', 'E'],
                            {'A': 'Group1', 'B': 'Group2', 'C': 'Group1',
                             'D': 'Group2', 'E': 'Group1'},
                            False, True, False,
                            [0.30, 0.40, 0.20, 0.10,
                             -0.30, 0.40, -0.20, 0.10,
                             0.20, 0.20, 0.20, 0.30, 0.10]),
                           ([[3,   4,  2,  1, nan],
                             [3,   4, -2, -1, nan],
                             [3, nan, nan, 1, 4]],
                            ['A', 'B', 'C', 'D', 'E'],
                            {'A': 'Group1', 'B': 'Group2', 'C': 'Group1',
                             'D': 'Group2', 'E': 'Group1'},
                            True, True, False,
                            [0.25,  0.25, -0.25, -0.25,
                             0.25, 0.25, -0.25, -0.25,
                             -0.50, nan, 0.50]),
                           ([[3, 4,   2,  1, 5],
                             [3, 4,  -2, -1, 5],
                             [3, nan, nan, 1, nan]],
                            ['A', 'B', 'C', 'D', 'E'],
                            {'A': 'Group1', 'B': 'Group2', 'C': 'Group1',
                             'D': 'Group2', 'E': 'Group1'},
                            False, False, True,
                            [0.20, 0.20, 0.20, 0.20, 0.20,
                             0.20, 0.20, -0.20, -0.20, 0.20,
                             0.50, 0.50]),
                           ([[1, 4,   2,   3, nan],
                             [1, 4,  -2,  -3, nan],
                             [3, nan, nan, 2, 7]],
                            ['A', 'B', 'C', 'D', 'E'],
                            {'A': 'Group1', 'B': 'Group2', 'C': 'Group1',
                             'D': 'Group2', 'E': 'Group1'},
                            True, False, True,
                            [-0.25, 0.25, -0.25, 0.25,
                             0.25, 0.25, -0.25, -0.25,
                             0., -0.50, 0.50]),
                           ([[3, 4,   2,   1, nan],
                             [-3, 4, -2,   1, nan],
                             [3, nan, nan,  1,  4],
                             [3, nan, nan, -1,  4],
                             [3, nan, nan,  1, -4]],
                            ['A', 'B', 'C', 'D', 'E'],
                            {'A': 'Group1', 'B': 'Group2', 'C': 'Group1',
                             'D': 'Group2', 'E': 'Group1'},
                            False, True, True,
                            [0.25, 0.25, 0.25, 0.25,
                             -0.25, 0.25, -0.25, 0.25,
                             0.25, 0.50, 0.25,
                             0.25, -0.50, 0.25,
                             0.25, 0.50, -0.25]),
                           ([[1, 4,   2,   3, nan],
                             [3, 4,  -2,  -1, nan],
                             [3, nan, nan, 2, 7],
                             [3, nan, nan, 2, -7]],
                            ['A', 'B', 'C', 'D', 'E'],
                            {'A': 'Group1', 'B': 'Group2', 'C': 'Group1',
                             'D': 'Group2', 'E': 'Group1'},
                            True, True, True,
                            [-0.25, 0.25, 0.25, -0.25,
                             0.25, 0.25, -0.25, -0.25,
                             -0.50, nan, 0.50,
                             0.50, nan, -0.50]),
                           ])
    def test_factor_weights(self,
                            factor_vals,
                            tickers,
                            groups,
                            demeaned,
                            group_adjust,
                            equal_weight,
                            expected_vals):

        index = date_range('1/12/2000', periods=len(factor_vals))
        factor = DataFrame(index=index,
                           columns=tickers,
                           data=factor_vals).stack()
        factor.index = factor.index.set_names(['date', 'asset'])
        factor.name = 'factor'

        factor_data = DataFrame()
        factor_data['factor'] = factor
        groups = Series(groups)
        factor_data['group'] = \
            Series(index=factor.index,
                   data=groups[factor.index.get_level_values('asset')].values)

        weights = \
            factor_weights(factor_data, demeaned, group_adjust, equal_weight)

        expected = Series(data=expected_vals,
                          index=factor_data.index,
                          name='factor')

        assert_series_equal(weights, expected)

    @parameterized.expand([([1, 2, 3, 4, 4, 3, 2, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            False,
                            [-1.25000, -1.25000]),
                           ([1, 1, 1, 1, 1, 1, 1, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            False,
                            [nan, nan]),
                           ([1, 2, 3, 4, 4, 3, 2, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            True,
                            [-0.5, -0.5]),
                           ([1, 2, 3, 4, 1, 2, 3, 4],
                            [1, 4, 1, 2, 1, 2, 2, 1],
                            True,
                            [1.0, 0.0]),
                           ([1, 1, 1, 1, 1, 1, 1, 1],
                            [4, 3, 2, 1, 1, 2, 3, 4],
                            True,
                            [nan, nan])
                           ])
    def test_factor_returns(self,
                            factor_vals,
                            fwd_return_vals,
                            group_adjust,
                            expected_vals):

        factor_data = self.factor_data.copy()
        factor_data['1D'] = fwd_return_vals
        factor_data['factor'] = factor_vals

        factor_returns_s = factor_returns(factor_data=factor_data,
                                          demeaned=True,
                                          group_adjust=group_adjust)

        expected = DataFrame(
            index=self.dr,
            data=expected_vals,
            columns=get_forward_returns_columns(
                factor_data.columns))

        assert_frame_equal(factor_returns_s, expected)

    @parameterized.expand([([1, 2, 3, 4, 1, 1, 1, 1],
                            -1,
                            5. / 6.)])
    def test_factor_alpha_beta(self, fwd_return_vals, alpha, beta):

        factor_data = self.factor_data.copy()
        factor_data['1D'] = fwd_return_vals

        ab = factor_alpha_beta(factor_data=factor_data)

        expected = DataFrame(columns=['1D'],
                             index=['Ann. alpha', 'beta'],
                             data=[alpha, beta])

        assert_frame_equal(ab, expected)

    @parameterized.expand([
        (
            [1.0, 0.5, 1.0, 0.5, 0.5],
            '1D',
            '1D',
            [2.0, 3.0, 6.0, 9.0, 13.50],
        ),
        (
            [0.1, 0.1, 0.1, 0.1, 0.1],
            '1D',
            '1D',
            [1.1, 1.21, 1.331, 1.4641, 1.61051],
        ),
        (
            [-0.1, -0.1, -0.1, -0.1, -0.1],
            '1D',
            '1D',
            [0.9, 0.81, 0.729, 0.6561, 0.59049],
        ),
        (
            [1.0, 0.5, 1.0, 0.5, 0.5],
            '1B',
            '1D',
            [2.0, 3.0, 6.0, 9.0, 13.50],
        ),
        (
            [0.1, 0.1, 0.1, 0.1, 0.1],
            '1B',
            '1D',
            [1.1, 1.21, 1.331, 1.4641, 1.61051],
        ),
        (
            [-0.1, -0.1, -0.1, -0.1, -0.1],
            '1B',
            '1D',
            [0.9, 0.81, 0.729, 0.6561, 0.59049],
        ),
        (
            [1.0, 0.5, 1.0, 0.5, 0.5],
            '1CD',
            '1D',
            [2.0, 3.0, 6.0, 9.0, 13.50],
        ),
        (
            [0.1, 0.1, 0.1, 0.1, 0.1],
            '1CD',
            '1D',
            [1.1, 1.21, 1.331, 1.4641, 1.61051],
        ),
        (
            [-0.1, -0.1, -0.1, -0.1, -0.1],
            '1CD',
            '1D',
            [0.9, 0.81, 0.729, 0.6561, 0.59049],
        ),
    ])
    def test_cumulative_returns(self,
                                returns,
                                ret_freq,
                                period_len,
                                expected_vals):
        if 'CD' in ret_freq:
            ret_freq_class = CDay(weekmask='Tue Wed Thu Fri Sun')
            ret_freq = ret_freq_class
        elif 'B' in ret_freq:
            ret_freq_class = BDay()
        else:
            ret_freq_class = Day()

        period_len = Timedelta(period_len)
        index = date_range('1/1/1999', periods=len(returns), freq=ret_freq)
        returns = Series(returns, index=index)

        cum_ret = cumulative_returns(returns)

        expected = Series(expected_vals, index=cum_ret.index)

        assert_series_equal(cum_ret, expected, check_less_precise=True)

    @parameterized.expand([([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 1,
                            [nan, 1.0, 1.0, 1.0]),
                           ([[1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 1,
                            [nan, 1.0, 1.0, 1.0]),
                           ([[4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1B', 1,
                            [nan, -1.0, -1.0, -1.0]),
                           ([[4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0],
                             [4.0, 3.0, 2.0, 1.0],
                             [1.0, 2.0, 3.0, 4.0]],
                            '1D', 1,
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
                            '1B', 3,
                            [nan, nan, nan, 1.0, 1.0,
                             1.0, 0.6, -0.6, -1.0, 1.0,
                             -0.6, -1.0]),
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
                            '1D', 3,
                            [nan, nan, nan, 1.0, 1.0,
                             1.0, 0.6, -0.6, -1.0, 1.0,
                             -0.6, -1.0])
                           ])
    def test_factor_rank_autocorrelation(self,
                                         factor_values,
                                         freq,
                                         period,
                                         expected_vals):

        dr = date_range(start='2015-1-1', periods=len(factor_values),
                        freq=freq)
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']
        factor = DataFrame(index=dr,
                           columns=tickers,
                           data=factor_values).stack()
        factor.index = factor.index.set_names(['date', 'asset'])

        factor_df = DataFrame()
        factor_df['factor'] = factor

        fa = factor_rank_autocorrelation(factor_df, period)
        expected = Series(index=dr, data=expected_vals)
        expected.name = period

        assert_series_equal(fa, expected)

    @parameterized.expand([
        (
            2, 3, False, False,
            [[4.93048307, 8.68843922], [6.60404312, 12.22369139],
             [8.92068367, 17.1794088], [12.1275523, 24.12861778],
             [16.5694159, 33.8740100], [22.7273233, 47.53995233]],
        ),
        (
            3, 2, False, True,
            [[0.0, 5.63219176], [0.0, 7.96515233],
             [0.0, 11.2420646], [0.0, 15.8458720],
             [0.0, 22.3134160], [0.0, 31.3970961]],
        ),
        (
            3, 5, True, False,
            [[3.7228318, 2.6210478], [4.9304831, 3.6296796], [6.6040431, 5.0193734],  # noqa
             [8.9206837, 6.9404046], [12.127552, 9.6023405], [16.569416, 13.297652],  # noqa
             [22.727323, 18.434747], [31.272682, 25.584180], [34.358565, 25.497254]],  # noqa
        ),
        (
            1, 4, True, True,
            [[0., 0.], [0., 0.], [0., 0.],
             [0., 0.], [0., 0.], [0., 0.]],
        ),
        (
            6, 6, False, False,
            [[2.02679565, 2.38468223], [2.38769454, 3.22602748],
             [2.85413029, 4.36044469], [3.72283181, 6.16462715],
             [4.93048307, 8.68843922], [6.60404312, 12.2236914],
             [8.92068367, 17.1794088], [12.1275523, 24.1286178],
             [16.5694159, 33.8740100], [22.7273233, 47.5399523],
             [31.2726821, 66.7013483], [34.3585654, 70.1828776],
             [37.9964585, 74.3294620]],
        ),
        (
            6, 6, False, True,
            [[0.0, 2.20770299], [0.0, 2.95942924], [0.0, 3.97022414],
             [0.0, 5.63219176], [0.0, 7.96515233], [0.0, 11.2420646],
             [0.0, 15.8458720], [0.0, 22.3134160], [0.0, 31.3970962],
             [0.0, 44.1512888], [0.0, 62.0533954], [0.0, 65.8668371],
             [0.0, 70.4306483]],
        ),
        (
            6, 6, True, False,
            [[2.0267957, 0.9562173], [2.3876945, 1.3511898], [2.8541303, 1.8856194],  # noqa
             [3.7228318, 2.6210478], [4.9304831, 3.6296796], [6.6040431, 5.0193734],  # noqa
             [8.9206837, 6.9404046], [12.127552, 9.6023405], [16.569416, 13.297652],  # noqa
             [22.727323, 18.434747], [31.272682, 25.584180], [34.358565, 25.497254],  # noqa
             [37.996459, 25.198051]],
        ),
        (
            6, 6, True, True,
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
             [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
             [0., 0.], [0., 0.], [0., 0.]],
        ),
    ])
    def test_common_start_returns(self,
                                  before,
                                  after,
                                  mean_by_date,
                                  demeaned,
                                  expected_vals):
        dr = date_range(start='2015-1-17', end='2015-2-2')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']
        r1, r2, r3, r4 = (1.20, 1.40, 0.90, 0.80)
        data = [[r1**i, r2**i, r3**i, r4**i] for i in range(1, 18)]
        returns = DataFrame(data=data, index=dr, columns=tickers)
        dr2 = date_range(start='2015-1-21', end='2015-1-29')
        factor = DataFrame(index=dr2, columns=tickers,
                           data=[[3, 4, 2, 1],
                                 [3, 4, 2, 1],
                                 [3, 4, 2, 1],
                                 [3, 4, 2, 1],
                                 [3, 4, 2, 1],
                                 [3, 4, 2, 1],
                                 [3, 4, 2, 1],
                                 [3, 4, 2, 1],
                                 [3, 4, 2, 1]]).stack()
        factor.index = factor.index.set_names(['date', 'asset'])
        factor.name = 'factor'

        cmrt = common_start_returns(
            factor,
            returns,
            before,
            after,
            cumulative=True,
            mean_by_date=mean_by_date,
            demean_by=factor if demeaned else None,
        )
        cmrt = DataFrame({'mean': cmrt.mean(axis=1), 'std': cmrt.std(axis=1)})
        expected = DataFrame(index=range(-before, after + 1),
                             columns=['mean', 'std'], data=expected_vals)
        assert_frame_equal(cmrt, expected)

    @parameterized.expand([
        (
            1, 2, False, 4,
            [[0.00512695, 0.00256348, 0.00128174, 6.40869e-4],
             [0.00579185, 0.00289592, 0.00144796, 7.23981e-4],
             [1.00000000, 1.00000000, 1.00000000, 1.00000000],
             [0.00000000, 0.00000000, 0.00000000, 0.00000000],
             [7.15814531, 8.94768164, 11.1846020, 13.9807526],
             [2.93784787, 3.67230984, 4.59038730, 5.73798413],
             [39.4519043, 59.1778564, 88.7667847, 133.150177],
             [28.3717330, 42.5575995, 63.8363992, 95.7545989]],
        ),
        (
            1, 2, True, 4,
            [[-11.898667, -17.279462, -25.236885, -37.032252],
             [7.82587034, 11.5529583, 17.0996881, 25.3636472],
             [-10.903794, -16.282025, -24.238167, -36.032893],
             [7.82140124, 11.5507268, 17.0985737, 25.3630906],
             [-4.7456488, -8.3343438, -14.053565, -23.052140],
             [4.91184665, 7.91180853, 12.5481552, 19.6734224],
             [27.5481102, 41.8958311, 63.5286176, 96.1172844],
             [20.5510133, 31.0075980, 46.7385910, 70.3923129]],
        ),
        (
            3, 0, False, 4,
            [[7.0, 3.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [-0.488, -0.36, -0.2, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [-0.703704, -0.55555555, -0.333333333, 0.0],
             [0.0, 0.0, 0.0, 0.0]],
        ),
        (
            0, 3, True, 4,
            [[-17.279462, -25.236885, -37.032252, -54.550061],
             [11.5529583, 17.0996881, 25.3636472, 37.6887906],
             [-16.282025, -24.238167, -36.032893, -53.550382],
             [11.5507268, 17.0985737, 25.3630906, 37.6885125],
             [-8.3343438, -14.053565, -23.052140, -37.074441],
             [7.91180853, 12.5481552, 19.6734224, 30.5748605],
             [41.8958311, 63.5286176, 96.1172844, 145.174884],
             [31.0075980, 46.7385910, 70.3923129, 105.944230]]),
        (
            3, 3, False, 2,
            [[0.5102539, 0.50512695, 0.50256348, 0.50128174, 0.50064087, 0.50032043, 0.50016022],  # noqa
             [0.0115837, 0.00579185, 0.00289592, 1.44796e-3, 7.23981e-4, 3.61990e-4, 1.80995e-4],  # noqa
             [11.057696, 16.0138929, 23.3050248, 34.0627690, 49.9756934, 73.5654648, 108.600603],  # noqa
             [7.2389454, 10.6247239, 15.6450367, 23.1025693, 34.1977045, 50.7264595, 75.3771641]],  # noqa
        ),
        (
            3, 3, True, 2,
            [[-5.273721, -7.754383, -11.40123, -16.78074, -24.73753, -36.53257, -54.05022],  # noqa
             [3.6239580, 5.3146000, 7.8236356, 11.551843, 17.099131, 25.363369, 37.688652],  # noqa
             [5.2737212, 7.7543830, 11.401231, 16.780744, 24.737526, 36.532572, 54.050221],  # noqa
             [3.6239580, 5.3146000, 7.8236356, 11.551843, 17.099131, 25.363369, 37.688652]],  # noqa
        ),
    ])
    def test_average_cumulative_return_by_quantile(self,
                                                   before,
                                                   after,
                                                   demeaned,
                                                   quantiles,
                                                   expected_vals):
        dr = date_range(start='2015-1-15', end='2015-2-1')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']
        r1, r2, r3, r4 = (1.25, 1.50, 1.00, 0.50)
        data = [[r1**i, r2**i, r3**i, r4**i] for i in range(1, 19)]
        returns = DataFrame(index=dr, columns=tickers, data=data)
        dr2 = date_range(start='2015-1-21', end='2015-1-26')
        dr2.name = 'date'
        factor = DataFrame(
            index=dr2, columns=tickers, data=[
                [3, 4, 2, 1],
                [3, 4, 2, 1],
                [3, 4, 2, 1],
                [3, 4, 2, 1],
                [3, 4, 2, 1],
                [3, 4, 2, 1]]).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, returns, quantiles=quantiles, periods=range(
                0, after + 1), filter_zscore=False)

        avgrt = average_cumulative_return_by_quantile(
            factor_data, returns, before, after, demeaned)
        arrays = []
        for q in range(1, quantiles + 1):
            arrays.append((q, 'mean'))
            arrays.append((q, 'std'))
        index = MultiIndex.from_tuples(arrays, names=['factor_quantile', None])
        expected = DataFrame(
            index=index, columns=range(-before, after + 1), data=expected_vals)
        assert_frame_equal(avgrt, expected)

    @parameterized.expand([
        (
            0, 2, False, 4,
            [[0.0292969, 0.0146484, 7.32422e-3],
             [0.0241851, 0.0120926, 6.04628e-3],
             [1.0000000, 1.0000000, 1.00000000],
             [0.0000000, 0.0000000, 0.00000000],
             [3.5190582, 4.3988228, 5.49852848],
             [1.0046375, 1.2557969, 1.56974616],
             [10.283203, 15.424805, 23.1372070],
             [5.2278892, 7.8418338, 11.7627508]],
        ),
        (
            0, 3, True, 4,
            [[-3.6785927, -5.1949205, -7.4034407, -10.641996],
             [1.57386873, 2.28176590, 3.33616491, 4.90228915],
             [-2.7078896, -4.2095690, -6.4107649, -9.6456583],
             [1.55205002, 2.27087143, 3.33072273, 4.89956999],
             [-0.1888313, -0.8107462, -1.9122365, -3.7724977],
             [0.55371389, 1.02143924, 1.76795263, 2.94536298],
             [6.57531357, 10.2152357, 15.7264421, 24.0601522],
             [3.67596914, 5.57112656, 8.43221341, 12.7447568]],
        ),
        (
            0, 3, False, 2,
            [[0.51464844, 0.50732422, 0.50366211, 0.50183105],
             [0.01209256, 0.00604628, 0.00302314, 0.00151157],
             [6.90113068, 9.91181374, 14.3178678, 20.7894856],
             [3.11499629, 4.54718783, 6.66416616, 9.80049950]],
        ),
        (
            0, 3, True, 2,
            [[-3.1932411, -4.7022448, -6.9071028, -10.143827],
             [1.56295067, 2.27631715, 3.33344356, 4.90092953],
             [3.19324112, 4.70224476, 6.90710282, 10.1438273],
             [1.56295067, 2.27631715, 3.33344356, 4.90092953]],
        ),
    ])
    def test_average_cumulative_return_by_quantile_2(self, before, after,
                                                     demeaned, quantiles,
                                                     expected_vals):
        """
        Test varying factor asset universe: at different dates there might be
        different assets
        """
        dr = date_range(start='2015-1-15', end='2015-1-25')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D', 'E', 'F']
        r1, r2, r3, r4 = (1.25, 1.50, 1.00, 0.50)
        data = [[r1**i, r2**i, r3**i, r4**i, r2**i, r3**i]
                for i in range(1, 12)]
        prices = DataFrame(index=dr, columns=tickers, data=data)
        dr2 = date_range(start='2015-1-18', end='2015-1-21')
        dr2.name = 'date'
        factor = DataFrame(index=dr2, columns=tickers,
                           data=[[3, 4, 2, 1, nan, nan],
                                 [3, 4, 2, 1, nan, nan],
                                 [3, nan, nan, 1, 4, 2],
                                 [3, nan, nan, 1, 4, 2]]).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices, quantiles=quantiles, periods=range(
                0, after + 1), filter_zscore=False)

        avgrt = average_cumulative_return_by_quantile(
            factor_data, prices, before, after, demeaned)
        arrays = []
        for q in range(1, quantiles + 1):
            arrays.append((q, 'mean'))
            arrays.append((q, 'std'))
        index = MultiIndex.from_tuples(arrays, names=['factor_quantile', None])
        expected = DataFrame(
            index=index, columns=range(-before, after + 1), data=expected_vals)
        assert_frame_equal(avgrt, expected)
