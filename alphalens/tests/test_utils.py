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
    datetime,
    Panel,
    MultiIndex,
)
from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal)

from .. utils import (compute_forward_returns,
                      demean_forward_returns,
                      common_start_returns)


class UtilsTestCase(TestCase):

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

    @parameterized.expand([(2, 4, False, False,
                            [[0.277778, 0.701250],[0.301075, 0.701552],[0.307292, 0.691290],
                            [0.307292, 0.691290],[0.284946, 0.690879],[0.277778, 0.701250],
                            [0.310345, 0.690479]]),
                           (4, 2, False, True,
                            [[0., 0.],[0., 0.],[0., 0.], [0., 0.],[0., 0.],[0., 0.],[0., 0.]]),
                           (3, 3, True, False,
                            [[0.310345, 0.699666],[0.277778, 0.710261],[0.301075, 0.710268],
                             [0.307292, 0.699602],[0.307292, 0.699602],[0.284946, 0.699462],
                             [0.277778, 0.710261]]),
                           (1, 5, True, True,
                            [[0., 0.],[0., 0.],[0., 0.], [0., 0.],[0., 0.],[0., 0.],[0., 0.]]),
                          ])
    def test_common_start_returns(self, before, after, mean_by_date, demeaned,
                                  expected_vals):
        dr = date_range(start='2015-1-1', end='2015-2-2')
        dr.name = 'date'
        tickers = ['A', 'B', 'C', 'D']
        prices = DataFrame(index=dr, columns=tickers,
                           data=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                                 [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        factor = DataFrame(index=dr, columns=tickers,
                           data=[[1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3],
                                 [1, 2, 4, 3], [1, 2, 4, 3], [1, 2, 4, 3]]).stack()
        factor.index = factor.index.set_names(['date', 'asset'])
        factor.name = 'factor'

        cmrt = common_start_returns(factor, prices, before, after, mean_by_date, demeaned)
        cmrt = DataFrame( {'mean': cmrt.mean(axis=1), 'std': cmrt.std(axis=1)} )
        expected = DataFrame(index=range(-before, after+1), columns=['mean', 'std'], data=expected_vals)
        assert_frame_equal(cmrt, expected)

