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
                      quantize_factor,
                      common_start_returns)


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

        fp = compute_forward_returns(prices, periods=[1, 2])

        ix = MultiIndex.from_product([dr, ['A', 'B']],
                                     names=['date', 'asset'])
        expected = DataFrame(index=ix, columns=[1, 2])
        expected[1] = [0., 1., 1., -0.5, nan, nan]
        expected[2] = [1., 0., nan, nan, nan, nan]

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

    @parameterized.expand([(2, 3, False, False,
                            [[0.075, 0.241868], [0.075, 0.241868],
                             [0.075, 0.241868], [0.075, 0.241868],
                             [0.075, 0.241868], [0.075, 0.241868]]),
                           (3, 2, False, True,
                            [[0.0, 0.241868], [0.0, 0.241868],
                             [0.0, 0.241868], [0.0, 0.241868],
                             [0.0, 0.241868], [0.0, 0.241868]]),
                           (3, 5, True, False,
                            [[0.075, 0.0], [0.075, 0.0], [0.075, 0.0],
                             [0.075, 0.0], [0.075, 0.0], [0.075, 0.0],
                             [0.075, 0.0], [0.075, 0.0], [0.075, 0.0]]),
                           (1, 4, True, True,
                            [[0., 0.], [0., 0.], [0., 0.],
                             [0., 0.], [0., 0.], [0., 0.]]),
                           (6, 6, False, False,
                            [[0.075, 0.243614], [0.075, 0.242861],
                             [0.075, 0.242301], [0.075, 0.241868],
                             [0.075, 0.241868], [0.075, 0.241868],
                             [0.075, 0.241868], [0.075, 0.241868],
                             [0.075, 0.241868], [0.075, 0.241868],
                             [0.075, 0.241868], [0.075, 0.242301],
                             [0.075, 0.242861]]),
                           (6, 6, False, True,
                            [[0.0, 0.243614], [0.0, 0.242861], [0.0, 0.242301],
                             [0.0, 0.241868], [0.0, 0.241868], [0.0, 0.241868],
                             [0.0, 0.241868], [0.0, 0.241868], [0.0, 0.241868],
                             [0.0, 0.241868], [0.0, 0.241868], [0.0, 0.242301],
                             [0.0, 0.242861]]),
                           (6, 6, True, False,
                            [[0.075, 0.0], [0.075, 0.0], [0.075, 0.0],
                             [0.075, 0.0], [0.075, 0.0], [0.075, 0.0],
                             [0.075, 0.0], [0.075, 0.0], [0.075, 0.0],
                             [0.075, 0.0], [0.075, 0.0], [0.075, 0.0],
                             [0.075, 0.0]]),
                           (6, 6, True, True,
                            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
                             [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
                             [0., 0.], [0., 0.], [0., 0.]]),
                           ])
    def test_common_start_returns(self, before, after, mean_by_date, demeaned,
                                  expected_vals):
        dr = date_range(start='2015-1-17', end='2015-2-2')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']
        r1, r2, r3, r4 = (1.20, 1.40, 0.90, 0.80)
        prices = DataFrame(index=dr, columns=tickers,
                           data=[[r1**1, r2**1, r3**1, r4**1],
                                 [r1**2, r2**2, r3**2, r4**2],
                                 [r1**3, r2**3, r3**3, r4**3],
                                 [r1**4, r2**4, r3**4, r4**4],
                                 [r1**5, r2**5, r3**5, r4**5],
                                 [r1**6, r2**6, r3**6, r4**6],
                                 [r1**7, r2**7, r3**7, r4**7],
                                 [r1**8, r2**8, r3**8, r4**8],
                                 [r1**9, r2**9, r3**9, r4**9],
                                 [r1**10, r2**10, r3**10, r4**10],
                                 [r1**11, r2**11, r3**11, r4**11],
                                 [r1**12, r2**12, r3**12, r4**12],
                                 [r1**13, r2**13, r3**13, r4**13],
                                 [r1**14, r2**14, r3**14, r4**14],
                                 [r1**15, r2**15, r3**15, r4**15],
                                 [r1**16, r2**16, r3**16, r4**16],
                                 [r1**17, r2**17, r3**17, r4**17]])
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
            prices,
            before,
            after,
            False,
            mean_by_date,
            factor if demeaned else None)
        cmrt = DataFrame({'mean': cmrt.mean(axis=1), 'std': cmrt.std(axis=1)})
        expected = DataFrame(index=range(-before, after + 1),
                             columns=['mean', 'std'], data=expected_vals)
        assert_frame_equal(cmrt, expected)
