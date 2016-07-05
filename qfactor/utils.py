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

import pandas as pd
import numpy as np
from IPython.display import display


MORNINGSTAR_SECTOR_MAPPING = {101: "Basic Materials",
                              102: "Consumer Cyclical",
                              103: "Financial Services",
                              104: "Real Estate",
                              205: "Consumer Defensive",
                              206: "Healthcare",
                              207: "Utilities",
                              308: "Communication Services",
                              309: "Energy",
                              310: "Industrials",
                              311: "Technology"}


def compute_forward_returns(prices, days=(1, 5, 10), filter_zscore=None):
    """
    Finds the N day forward returns (as percent change) for each asset provided.
    Parameters
    ----------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Equities as columns, dates as index.
        Pricing data must span the factor analysis time period
        plus an additional buffer window
        that is greater than the maximum number of expected days
        in the forward returns calculations.
    days : list
        Number of days forward to project returns. One column will
        be added for each value.
    filter_zscore : int
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorperates lookahead bias.
    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        DataFrame containing the N day forward returns for a security.
    """

    forward_returns = pd.DataFrame(index=pd.MultiIndex.from_product(
        [prices.index, prices.columns], names=['date', 'asset']))

    for day in days:
        delta = prices.pct_change(day).shift(-day)

        if filter_zscore is not None:
            mask = abs(delta - delta.mean()) > (filter_zscore * delta.std())
            delta[mask] = np.nan

        forward_returns[day] = delta.stack() / day

    forward_returns.index.rename(['date', 'asset'], inplace=True)

    return forward_returns


def demean_forward_returns(forward_returns, by_sector=False):
    """
    Convert forward price movements to price movements relative to mean
    daily all-universe or sector price movements.
    Sector-wise normalization incorporates the assumption of a
    sector neutral portfolio constraint and thus allows allows the
    factor to be evaluated across sectors.

    For example, if AAPL 5 day return is 0.1% and mean 5 day
    return for the Technology stocks in our universe was 0.5% in the
     same period, the sector adjusted 5 day return for AAPL
    in this period is -0.4%.


    Parameters
    ----------
    forward_returns : pd.DataFrame - MultiIndex
        DataFrame with date, asset, sector, and forward returns columns.
        See compute_forward_returns for more detail.
    by_sector : boolean
        If True, demean according to sector.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by sector.

    """
    grouper = ['date', 'sector'] if by_sector else ['date']

    return forward_returns.groupby(level=grouper).apply(lambda x: x - x.mean())


def print_table(table, name=None, fmt=None):
    """Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pandas.Series or pandas.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.

    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if fmt is not None:
        prev_option = pd.get_option('display.float_format')
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    if name is not None:
        table.columns.name = name

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def format_input_data(factor, prices, sectors=None,
                      filter_zscore=20, days=(1, 5, 10),
                      sector_names=None):
    """
    Formats the factor data, pricing data, and sector mappings
    into DataFrames and Series that contain aligned MultiIndex
    indices containing date, asset, and sector.
    ----------
    ----------
    factor : pandas.Series - MultiIndex
        A list of equities and their factor values indexed by date.
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Equities as columns, dates as index.
        Pricing data must span the factor analysis time period
        plus an additional buffer window that is greater than the
        maximum number of expected days in the forward returns
        calculations.
    sectors : pd.Series - MultiIndex
        A list of equities and their sectors
    filter_zscore : int
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorperates lookahead bias.
    days : list
        Number of days forward to project returns. One column
        will be added for each value.
    sector_names: dict
        A dictionary keyed by sector code with values corresponding
        to the display name for each sector.
        - Example:
            {101: "Basic Materials", 102: "Consumer Cyclical"}
    Returns
    -------
    factor : pd.Series
        A list of equities and their factor values indexed by date,
        asset, and optionally sector.
    forward_returns : pd.DataFrame - MultiIndex
        A DataFrame of equities and their forward returns
        indexed by date, asset, and optionally sector.
        Note: this is the same index as the factor index
    """

    factor.name = 'factor'
    factor = factor.rename_axis(['date', 'asset'], axis=0)

    forward_returns = compute_forward_returns(
        prices, days, filter_zscore=filter_zscore)

    merged_data = pd.merge(pd.DataFrame(factor),
                           forward_returns,
                           how='left',
                           left_index=True,
                           right_index=True)

    if sectors is not None:
        if isinstance(sectors, dict):
            try:
                daily_sector = map(lambda x: sectors[x], factor.reset_index().asset.values)
            except KeyError:
                diff = set(factor.index.get_level_values(
                    'asset')) - set(sectors.keys())
                raise KeyError(
                    "Assets {} not in sector mapping".format(
                        list(diff)))

            sectors = pd.Series(index=factor.index,
                                data=daily_sector)

        sectors.name = 'sector'
        sectors = sectors.rename_axis(['date', 'asset'], axis=0)

        if sector_names is not None:
            sectors = sectors.apply(
                lambda x: sector_names.get(x, x))

        merged_data = pd.merge(pd.DataFrame(sectors),
                               merged_data,
                               how='left',
                               left_index=True,
                               right_index=True)

        merged_data = merged_data.set_index('sector', append=True)

    merged_data = merged_data.dropna()

    factor = merged_data.pop("factor")
    forward_returns = merged_data

    return factor, forward_returns
