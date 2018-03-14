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
from numpy import (nan)

from pandas import (
    Series,
    DataFrame,
    date_range,
    MultiIndex,
    Timedelta,
    concat,
)
from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal)

from .. utils import (get_clean_factor_and_forward_returns,
                      compute_forward_returns,
                      quantize_factor)


class UtilsTestCase(TestCase):
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

    def test_compute_forward_returns(self):
        dr = date_range(start='2015-1-1', end='2015-1-3')
        prices = DataFrame(index=dr, columns=['A', 'B'],
                           data=[[1, 1], [1, 2], [2, 1]])
        factor = prices.stack()

        fp = compute_forward_returns(factor, prices, periods=[1, 2])

        ix = MultiIndex.from_product([dr, ['A', 'B']],
                                     names=['date', 'asset'])
        expected = DataFrame(index=ix, columns=['1D', '2D'])
        expected['1D'] = [0., 1., 1., -0.5, nan, nan]
        expected['2D'] = [1., 0., nan, nan, nan, nan]

        assert_frame_equal(fp, expected)

    @parameterized.expand([(factor_data, 4, None, False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor_data, 2, None, False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor_data, 2, None, True,
                            [1, 2, 1, 2, 2, 1, 2, 1]),
                           (factor_data, [0, .25, .5, .75, 1.], None, False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor_data, [0, .5, .75, 1.], None, False,
                            [1, 1, 2, 3, 3, 2, 1, 1]),
                           (factor_data, [0, .25, .5, 1.], None, False,
                            [1, 2, 3, 3, 3, 3, 2, 1]),
                           (factor_data, [0, .5, 1.], None, False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor_data, [.25, .5, .75], None, False,
                            [nan, 1, 2, nan, nan, 2, 1, nan]),
                           (factor_data, [0, .5, 1.], None, True,
                            [1, 2, 1, 2, 2, 1, 2, 1]),
                           (factor_data, [.5, 1.], None, True,
                            [nan, 1, nan, 1, 1, nan, 1, nan]),
                           (factor_data, [0, 1.], None, True,
                            [1, 1, 1, 1, 1, 1, 1, 1]),
                           (factor_data, None, 4, False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor_data, None, 2, False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor_data, None, 3, False,
                            [1, 1, 2, 3, 3, 2, 1, 1]),
                           (factor_data, None, 8, False,
                            [1, 3, 6, 8, 8, 6, 3, 1]),
                           (factor_data, None, [0, 1, 2, 3, 5], False,
                            [1, 2, 3, 4, 4, 3, 2, 1]),
                           (factor_data, None, [1, 2, 3], False,
                            [nan, 1, 2, nan, nan, 2, 1, nan]),
                           (factor_data, None, [0, 2, 5], False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor_data, None, [0.5, 2.5, 4.5], False,
                            [1, 1, 2, 2, 2, 2, 1, 1]),
                           (factor_data, None, [0.5, 2.5], True,
                            [1, 1, nan, nan, nan, nan, 1, 1]),
                           (factor_data, None, 2, True,
                            [1, 2, 1, 2, 2, 1, 2, 1])])
    def test_quantize_factor(self, factor, quantiles, bins, by_group,
                             expected_vals):
        quantized_factor = quantize_factor(factor,
                                           quantiles=quantiles,
                                           bins=bins,
                                           by_group=by_group)
        expected = Series(index=factor.index,
                          data=expected_vals,
                          name='factor_quantile').dropna()
        assert_series_equal(quantized_factor, expected)

    def test_get_clean_factor_and_forward_returns_1(self):
        """
        Test get_clean_factor_and_forward_returns with a daily factor
        """
        tickers = ['A', 'B', 'C', 'D', 'E', 'F']

        factor_groups = {'A': 1, 'B': 2, 'C': 1, 'D': 2, 'E': 1, 'F': 2}

        price_data = [[1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i]
                      for i in range(1, 7)]

        factor_data = [[3, 4, 2, 1, nan, nan],
                       [3, nan, nan, 1, 4, 2],
                       [3, 4, 2, 1, nan, nan]]

        price_index = date_range(start='2015-1-10', end='2015-1-15')
        price_index.name = 'date'
        prices = DataFrame(index=price_index, columns=tickers, data=price_data)

        factor_index = date_range(start='2015-1-10', end='2015-1-12')
        factor_index.name = 'date'
        factor = DataFrame(index=factor_index, columns=tickers,
                           data=factor_data).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices,
            groupby=factor_groups,
            quantiles=4,
            periods=(1, 2, 3))

        expected_idx = factor.index.rename(['date', 'asset'])
        expected_cols = ['1D', '2D', '3D',
                         'factor', 'group', 'factor_quantile']
        expected_data = [[0.1,  0.21,  0.331, 3.0, 1, 3],
                         [-0.5, -0.75, -0.875, 4.0, 2, 4],
                         [2.0,  8.00, 26.000, 2.0, 1, 2],
                         [-0.1, -0.19, -0.271, 1.0, 2, 1],
                         [0.1,  0.21,  0.331, 3.0, 1, 3],
                         [-0.1, -0.19, -0.271, 1.0, 2, 1],
                         [-0.5, -0.75, -0.875, 4.0, 1, 4],
                         [0.0,  0.00,  0.000, 2.0, 2, 2],
                         [0.1,  0.21,  0.331, 3.0, 1, 3],
                         [-0.5, -0.75, -0.875, 4.0, 2, 4],
                         [2.0,  8.00, 26.000, 2.0, 1, 2],
                         [-0.1, -0.19, -0.271, 1.0, 2, 1]]
        expected = DataFrame(index=expected_idx,
                             columns=expected_cols, data=expected_data)
        expected['group'] = expected['group'].astype('category')

        assert_frame_equal(factor_data, expected)

    def test_get_clean_factor_and_forward_returns_2(self):
        """
        Test get_clean_factor_and_forward_returns with a daily factor
        on a business day calendar
        """
        tickers = ['A', 'B', 'C', 'D', 'E', 'F']

        factor_groups = {'A': 1, 'B': 2, 'C': 1, 'D': 2, 'E': 1, 'F': 2}

        price_data = [[1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i]
                      for i in range(1, 7)]

        factor_data = [[3, 4, 2, 1, nan, nan],
                       [3, nan, nan, 1, 4, 2],
                       [3, 4, 2, 1, nan, nan]]

        price_index = date_range(start='2017-1-12', end='2017-1-19', freq='B')
        price_index.name = 'date'
        prices = DataFrame(index=price_index, columns=tickers, data=price_data)

        factor_index = date_range(start='2017-1-12', end='2017-1-16', freq='B')
        factor_index.name = 'date'
        factor = DataFrame(index=factor_index, columns=tickers,
                           data=factor_data).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices,
            groupby=factor_groups,
            quantiles=4,
            periods=(1, 2, 3))

        expected_idx = factor.index.rename(['date', 'asset'])
        expected_cols = ['1D', '2D', '3D',
                         'factor', 'group', 'factor_quantile']
        expected_data = [[0.1,  0.21,  0.331, 3.0, 1, 3],
                         [-0.5, -0.75, -0.875, 4.0, 2, 4],
                         [2.0,  8.00, 26.000, 2.0, 1, 2],
                         [-0.1, -0.19, -0.271, 1.0, 2, 1],
                         [0.1,  0.21,  0.331, 3.0, 1, 3],
                         [-0.1, -0.19, -0.271, 1.0, 2, 1],
                         [-0.5, -0.75, -0.875, 4.0, 1, 4],
                         [0.0,  0.00,  0.000, 2.0, 2, 2],
                         [0.1,  0.21,  0.331, 3.0, 1, 3],
                         [-0.5, -0.75, -0.875, 4.0, 2, 4],
                         [2.0,  8.00, 26.000, 2.0, 1, 2],
                         [-0.1, -0.19, -0.271, 1.0, 2, 1]]
        expected = DataFrame(index=expected_idx,
                             columns=expected_cols, data=expected_data)
        expected['group'] = expected['group'].astype('category')

        assert_frame_equal(factor_data, expected)

    def test_get_clean_factor_and_forward_returns_3(self):
        """
        Test get_clean_factor_and_forward_returns with and intraday factor
        """
        tickers = ['A', 'B', 'C', 'D', 'E', 'F']

        factor_groups = {'A': 1, 'B': 2, 'C': 1, 'D': 2, 'E': 1, 'F': 2}

        price_data = [[1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i]
                      for i in range(1, 7)]

        factor_data = [[3, 4, 2, 1, nan, nan],
                       [3, nan, nan, 1, 4, 2],
                       [3, 4, 2, 1, nan, nan]]

        price_index = date_range(start='2017-1-12', end='2017-1-19', freq='B')
        price_index.name = 'date'
        today_open = DataFrame(index=price_index + Timedelta('9h30m'),
                               columns=tickers, data=price_data)
        today_open_1h = DataFrame(index=price_index + Timedelta('10h30m'),
                                  columns=tickers, data=price_data)
        today_open_1h += today_open_1h * 0.001
        today_open_3h = DataFrame(index=price_index + Timedelta('12h30m'),
                                  columns=tickers, data=price_data)
        today_open_3h -= today_open_3h * 0.002
        prices = concat([today_open, today_open_1h, today_open_3h]) \
            .sort_index()

        factor_index = date_range(start='2017-1-12', end='2017-1-16', freq='B')
        factor_index.name = 'date'
        factor = DataFrame(index=factor_index + Timedelta('9h30m'),
                           columns=tickers, data=factor_data).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices,
            groupby=factor_groups,
            quantiles=4,
            periods=(1, 2, 3))

        expected_idx = factor.index.rename(['date', 'asset'])
        expected_cols = ['1h', '3h', '1D',
                         'factor', 'group', 'factor_quantile']
        expected_data = [[0.001, -0.002, 0.1, 3.0, 1, 3],
                         [0.001, -0.002, -0.5, 4.0, 2, 4],
                         [0.001, -0.002, 2.0, 2.0, 1, 2],
                         [0.001, -0.002, -0.1, 1.0, 2, 1],
                         [0.001, -0.002, 0.1, 3.0, 1, 3],
                         [0.001, -0.002, -0.1, 1.0, 2, 1],
                         [0.001, -0.002, -0.5, 4.0, 1, 4],
                         [0.001, -0.002, 0.0, 2.0, 2, 2],
                         [0.001, -0.002, 0.1, 3.0, 1, 3],
                         [0.001, -0.002, -0.5, 4.0, 2, 4],
                         [0.001, -0.002, 2.0, 2.0, 1, 2],
                         [0.001, -0.002, -0.1, 1.0, 2, 1]]
        expected = DataFrame(index=expected_idx,
                             columns=expected_cols, data=expected_data)
        expected['group'] = expected['group'].astype('category')

        assert_frame_equal(factor_data, expected)

    def test_get_clean_factor_and_forward_returns_4(self):
        """
        Test get_clean_factor_and_forward_returns on an event
        """
        tickers = ['A', 'B', 'C', 'D', 'E', 'F']

        factor_groups = {'A': 1, 'B': 2, 'C': 1, 'D': 2, 'E': 1, 'F': 2}

        price_data = [[1.10**i, 0.50**i, 3.00**i, 0.90**i, 0.50**i, 1.00**i]
                      for i in range(1, 9)]

        factor_data = [[1, nan, nan, nan, nan, 6],
                       [4, nan, nan, 7, nan, nan],
                       [nan, nan, nan, nan, nan, nan],
                       [nan, 3, nan, 2, nan, nan],
                       [nan, nan, 1, nan, 3, nan]]

        price_index = date_range(start='2017-1-12', end='2017-1-23', freq='B')
        price_index.name = 'date'
        prices = DataFrame(index=price_index, columns=tickers, data=price_data)

        factor_index = date_range(start='2017-1-12', end='2017-1-18', freq='B')
        factor_index.name = 'date'
        factor = DataFrame(index=factor_index, columns=tickers,
                           data=factor_data).stack()

        factor_data = get_clean_factor_and_forward_returns(
            factor, prices,
            groupby=factor_groups,
            quantiles=4,
            periods=(1, 2, 3))

        expected_idx = factor.index.rename(['date', 'asset'])
        expected_cols = ['1D', '2D', '3D',
                         'factor', 'group', 'factor_quantile']
        expected_data = [[0.1,  0.21,  0.331, 1.0, 1, 1],
                         [0.0,   0.00,  0.000, 6.0, 2, 4],
                         [0.1,  0.21,  0.331, 4.0, 1, 1],
                         [-0.1, -0.19, -0.271, 7.0, 2, 4],
                         [-0.5, -0.75, -0.875, 3.0, 2, 4],
                         [-0.1, -0.19, -0.271, 2.0, 2, 1],
                         [2.0,  8.00, 26.000, 1.0, 1, 1],
                         [-0.5, -0.75, -0.875, 3.0, 1, 4]]
        expected = DataFrame(index=expected_idx,
                             columns=expected_cols, data=expected_data)
        expected['group'] = expected['group'].astype('category')

        assert_frame_equal(factor_data, expected)
