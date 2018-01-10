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
)
from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal)

from .. utils import (compute_forward_returns,
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

        fp = compute_forward_returns(prices.index, prices, periods=[1, 2])

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
