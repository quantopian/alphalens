from __future__ import division
from unittest import TestCase
from parameterized import parameterized
from numpy import nan
from pandas import (DataFrame, date_range, Timedelta, concat)

from .. utils import (get_clean_factor_and_forward_returns,
                      join_factor_with_factor_and_forward_returns)
from .. tears import create_factors_interaction_tear_sheet

class MultiFactorTestCase(TestCase):

    tickers = ['A', 'B', 'C', 'D', 'E', 'F']
    factor_groups = {'A': 1, 'B': 2, 'C': 1, 'D': 2, 'E': 1, 'F': 2}
    price_data = [
        [1.25 ** i, 1.50 ** i, 1.00 ** i, 0.50 ** i, 1.50 ** i, 1.00 ** i]
        for i in range(1, 51)]
    factor_1_data = [[3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
                   [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
                   [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
                   [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
                   [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
                   [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
                   [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2]]

    factor_2_data = [[2, 3, 4, 1, nan, nan], [1, 2, 3, 4, nan, nan],
                   [4, 3, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [4, 2, 1, 3, nan, nan], [2, 1, 3, 4, nan, nan],
                   [1, nan, nan, 2, 3, 4], [2, nan, nan, 4, 1, 3],
                   [1, 4, 3, 2, nan, nan], [2, 3, 4, 1, nan, nan],
                   [1, nan, nan, 2, 4, 3], [1, nan, nan, 2, 4, 3],
                   [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
                   [1, nan, nan, 4, 3, 2], [4, nan, nan, 3, 2, 1],
                   [2, nan, nan, 3, 1, 4], [2, nan, nan, 1, 3, 4],
                   [1, nan, nan, 2, 3, 4], [1, nan, nan, 3, 2, 4],
                   [4, 3, 2, 1, nan, nan], [3, 4, 1, 2, nan, nan],
                   [4, 1, 2, 3, nan, nan], [3, 4, 1, 2, nan, nan],
                   [1, 2, 3, 4, nan, nan], [2, 1, 3, 4, nan, nan],
                   [2, 3, 4, 1, nan, nan], [3, 4, 2, 1, nan, nan],
                   [4, nan, nan, 1, 2, 3], [1, nan, nan, 2, 3, 4]]

    #
    # full calendar
    #
    price_index = date_range(start='2015-1-10', end='2015-2-28')
    price_index.name = 'date'
    prices = DataFrame(index=price_index, columns=tickers, data=price_data)

    factor_index = date_range(start='2015-1-15', end='2015-2-13')
    factor_index.name = 'date'
    factor_1 = DataFrame(index=factor_index, columns=tickers,
                       data=factor_1_data).stack()

    factor_2 = DataFrame(index=factor_index, columns=tickers,
                       data=factor_2_data).stack()
    factor_2.name = 'factor_2'

    @parameterized.expand([(False, 2, (1, 5, 10), None),
                           (False, 3, (1, 2, 3, 7), 20),
                           (True, 2, (1, 5, 10), None)])
    def test_create_multi_factor_data(self,
                                      binning_by_group,
                                      quantiles,
                                      periods,
                                      filter_zscore):
        """
        Test no exceptions are thrown when creating multi_factor_data
        DataFrame
        """
        factor_data = \
            get_clean_factor_and_forward_returns(self.factor_1,
                                                 self.prices,
                                                 self.factor_groups,
                                                 binning_by_group,
                                                 quantiles=quantiles,
                                                 periods=periods,
                                                 filter_zscore=filter_zscore)

        multi_factor_data = \
            join_factor_with_factor_and_forward_returns(factor_data,
                                                        self.factor_2,
                                                        binning_by_group,
                                                        quantiles)

        print multi_factor_data.head()