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


def compute_forward_returns(prices, periods=(1, 5, 10), filter_zscore=None):
    """
    Finds the N period forward returns (as percent change) for each asset provided.

    Parameters
    ----------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    """

    forward_returns = pd.DataFrame(index=pd.MultiIndex.from_product(
        [prices.index, prices.columns], names=['date', 'asset']))

    for period in periods:
        delta = prices.pct_change(period).shift(-period)

        if filter_zscore is not None:
            mask = abs(delta - delta.mean()) > (filter_zscore * delta.std())
            delta[mask] = np.nan

        forward_returns[period] = delta.stack()

    forward_returns.index.rename(['date', 'asset'], inplace=True)

    return forward_returns


def demean_forward_returns(forward_returns, by_group=False):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows allows the
    factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe was 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.

    Parameters
    ----------
    forward_returns : pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    by_group : bool
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    grouper = ['date', 'group'] if by_group else ['date']

    return forward_returns.groupby(level=grouper).apply(lambda x: x - x.mean())


def print_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
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


def get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=None,
                                         periods=(1, 5, 10),
                                         filter_zscore=20,
                                         groupby_labels=None):
    """
    Formats the factor data, pricing data, and group mappings
    into DataFrames and Series that contain aligned MultiIndex
    indices containing date, asset, and group.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor.
    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets
        in the columns. It is important to pass the
        correct pricing data in depending on what time of period your
        signal was generated so to avoid lookahead bias, or
        delayed calculations. Pricing data must span the factor
        analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.

    Returns
    -------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor.
    forward_returns : pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    """

    factor = factor.copy()

    factor.name = 'factor'
    factor.index = factor.index.set_names(['date', 'asset'])

    forward_returns = compute_forward_returns(
        prices, periods, filter_zscore=filter_zscore)

    merged_data = pd.merge(pd.DataFrame(factor),
                           forward_returns,
                           how='left',
                           left_index=True,
                           right_index=True)

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor.index.get_level_values(
                'asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Assets {} not in group mapping".format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor.index,
                                data=ss[factor.index.get_level_values('asset')].values)

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError(
                    "groups {} not in passed group names".format(
                        list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=factor.index,
                                data=sn[groupby.values].values)

        groupby.name = 'group'
        groupby.index = groupby.index.set_names(['date', 'asset'])

        merged_data = pd.merge(pd.DataFrame(groupby),
                               merged_data,
                               how='left',
                               left_index=True,
                               right_index=True)

        merged_data = merged_data.set_index('group', append=True)

    merged_data = merged_data.dropna()

    factor = merged_data.pop("factor")
    forward_returns = merged_data

    return factor, forward_returns


def common_start_returns(factor, returns, before, after):
    """
    A date and equity pair is extracted from each index row in the factor dataframe and for each of 
    these pairs a return series is built starting from 'before' returns before the date and ending
    'after' returns after the date specified in the pair. All those returns series are then aligned
    to a common index (-before to after) and returned as a single DataFrame

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame with at least date and equity as index, the columns are irrelevant
    returns : pd.DataFrame
        Returns for the equities in factor. Equities as columns, dates as index.
    before:
        How many returns to load before factor date
    after:
        How many returns to load after factor date
    Returns
    -------
    aligned_returns : pd.DataFrame
        Dataframe containing returns series for each factor aligned to the same index:
        -before to after
    """
   
    all_returns = []

    for timestamp, df in factor.groupby(level=0):  # group by date

        equities = df.index.get_level_values(1)

        try:
            day_zero_index = returns.index.get_loc(timestamp)
        except KeyError:
            continue

        starting_index = max(day_zero_index - before, 0)
        ending_index = min(day_zero_index + after + 1, 
                           len(returns.index) - 1)

        series = returns.ix[starting_index:ending_index, equities]
        series.index = range(starting_index - day_zero_index,
                             ending_index - day_zero_index)

        all_returns.append(series)

    return pd.concat(all_returns, axis=1)

