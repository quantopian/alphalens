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
                      demean_forward_returns)


class UtilsTestCase(TestCase):

    def test_compute_forward_returns(self):
        dr = date_range(start='2015-1-1', end='2015-1-3')
        prices = DataFrame(index=dr, columns=['A', 'B'],
                           data=[[1, 1], [1, 2], [2, 1]])

        fp = compute_forward_returns(prices, days=[1, 2])

        ix = MultiIndex.from_product([dr, ['A', 'B']],
                                     names=['date', 'asset'])
        expected = DataFrame(index=ix, columns=[1, 2])
        expected[1] = [0., 1., 1., -0.5, nan, nan]
        expected[2] = [0.41421356237309515, 0., nan, nan, nan, nan]

        assert_frame_equal(fp, expected)
