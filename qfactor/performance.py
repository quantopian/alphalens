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
import scipy as sp

import utils
from collections import defaultdict
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def factor_information_coefficient(factor, forward_returns,
                                   sector_adjust=False, by_sector=False):
    """
    Computes Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N day forward returns for each day in
    the factor index.

    Parameters
    ----------
    factor : pandas.Series - MultiIndex
        Factor values indexed by date and symbol.
    forward_returns : pandas.DataFrame - MultiIndex
        Daily forward returns in indexed by date and symbol.
        Separate column for each forward return window.
    sector_adjust : boolean
        Demean forward returns by sector before computing IC.
    by_sector : boolean
        If True, compute daily IC separately for each sector.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward price movement windows.

    err : pd.DataFrame
        Standard error of computed IC.

    """

    def src_ic(group):
        f = group.pop('factor')
        _ic = group.apply(lambda x: sp.stats.spearmanr(x, f)[0])

        return _ic

    if sector_adjust:
        forward_returns = utils.demean_forward_returns(forward_returns,
                                                       by_sector=True)

    factor.name = 'factor'
    factor_and_fp = pd.merge(pd.DataFrame(factor),
                             forward_returns,
                             how='left',
                             left_index=True,
                             right_index=True)

    grouper = ['date', 'sector'] if by_sector else ['date']
    ic = factor_and_fp.groupby(level=grouper).apply(src_ic)

    ic.columns = pd.Int64Index(ic.columns)

    return ic


def mean_information_coefficient(factor, forward_returns,
                                 sector_adjust=False,
                                 by_time=None, by_sector=False):
    """
    Get the mean information coefficient of specified groups.
    Answers questions like:
    What is the mean IC for each month?
    What is the mean IC for each sector for our whole timerange?
    What is the mean IC for for each sector, each week?

    Parameters
    ----------
    factor : pandas.Series - MultiIndex
        Factor values indexed by date and symbol.
    forward_returns : pandas.DataFrame - MultiIndex
        Daily forward returns in indexed by date and symbol.
        Separate column for each forward return window.
    sector_adjust : boolean
        Demean forward returns by sector before computing IC.
    by_time : string (pandas time_rule), optional
        Time window to use when taking mean IC.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        for available options.
    by_sector : boolean
        If True, take the mean IC for each sector.

    Returns
    -------
    ic : pd.DataFrame
        Mean Spearman Rank correlation between factor and provided
        forward price movement windows.
    """

    ic = factor_information_coefficient(factor,
                                        forward_returns,
                                        sector_adjust=sector_adjust,
                                        by_sector=by_sector)

    grouper = []
    if by_time is not None:
        grouper.append(pd.TimeGrouper(by_time))
    if by_sector:
        grouper.append('sector')

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        ic = (ic.reset_index()
              .set_index('date')
              .groupby(grouper)
              .mean())

    ic.columns = pd.Int64Index(ic.columns)

    return ic


def factor_returns(factor, forward_returns, long_short=True):
    """
    Computes daily returns for portfolio weighted by factor
    values. Weights are computed by demeaning factors and dividing
    by the sum of their absolute value (acheiving gross leverage of 1).

    Parameters
    ----------
    factor : pandas.Series - MultiIndex
        A list of equities and their factor values indexed by date.
    forward_returns : pandas.DataFrame - MultiIndex
        Daily forward returns in indexed by date and symbol.
        Separate column for each forward return window.

    Returns
    -------
    factor_daily_returns : pd.Series
        Daily returns of dollar neutral portfolio weighted by factor value.
    """

    def to_weights(group, long_short):
        if long_short:
            return (group - group.mean()) / abs(group).sum()
        else:
            return group / abs(group).sum()

    weights = factor.groupby(level=['date']).apply(to_weights,
                                                   long_short=long_short)
    weighted_returns = forward_returns.multiply(weights, axis=0).dropna()

    factor_daily_returns = weighted_returns.groupby(level='date').mean()

    return factor_daily_returns


def factor_alpha_beta(factor, forward_returns, factor_daily_returns=None):
    """
    Computes the alpha (excess returns), alpha t-stat (alpha significance),
    and beta (market exposure) of a factor. A regression is run with
    the daily factor universe mean return as the depedendent variable
    and mean daily return from a dollar-neutral portfolio weighted
    by factor values as the independent variable.

    Parameters
    ----------
    factor : pandas.Series - MultiIndex
        A list of equities and their factor values indexed by date.
    forward_returns : pandas.DataFrame - MultiIndex
        Daily forward returns in indexed by date and symbol.
        Separate column for each forward return window.
    factor_daily_returns : pd.Series
        Timeseries of daily factor returns. If passed, will
        use instad of long-short factor returns from factor.

    Returns
    -------
    factor_daily_returns : pd.Series
        Daily returns of dollar neutral portfolio weighted by factor value.
    """
    if factor_daily_returns is None:
        factor_daily_returns = factor_returns(factor, forward_returns)

    universe_daily_ret = (forward_returns.groupby(level='date')
                          .mean()
                          .loc[factor_daily_returns.index])

    alpha_beta = pd.DataFrame()
    for days in factor_daily_returns.columns.values:
        y = universe_daily_ret[days].values
        x = factor_daily_returns[days].values

        x = add_constant(x)
        reg_fit = OLS(y, x).fit()
        t_alpha = reg_fit.tvalues[0]
        alpha, beta = reg_fit.params

        alpha_beta.loc['alpha', days] = alpha
        alpha_beta.loc['t-stat(alpha)', days] = t_alpha
        alpha_beta.loc['beta', days] = beta

    return alpha_beta


def quantize_factor(factor, quantiles=5, by_sector=False):
    """
    Computes daily factor quantiles.

    Parameters
    ----------
    factor : pandas.Series - MultiIndex
        A list of equities and their factor values indexed by date.
    quantiles : integer
        Number of quantiles buckets to use in factor bucketing.
    by_sector : boolean
        If True, compute quantile buckets separately for each sector.

    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and symbol.
    """

    grouper = ['date', 'sector'] if by_sector else ['date']

    factor_percentile = factor.groupby(level=grouper).rank(pct=True)

    q_width = 1. / quantiles
    factor_quantile = factor_percentile.apply(
        lambda x: ((x - .000000001) // q_width) + 1)
    factor_quantile.name = 'quantile'

    return factor_quantile


def mean_return_by_quantile(quantized_factor, forward_returns,
                                   by_time=None, by_sector=False, std=False):
    """
    Computes mean demeaned returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    quantized_factor : pd.Series - MultiIndex
        DataFrame with date, equity index and factor quantile as a column.
        See quantile_bucket_factor for more detail.
    forward_returns : pandas.DataFrame - MultiIndex
        A list of equities and their N day forward returns where each column contains
        the N day forward returns
    by_sector : boolean
        If True, compute quantile bucket returns separately for each sector.
        Returns demeaning will occur on the sector level.

    Returns
    -------
    mean_returns_by_quantile : pd.DataFrame
        Sector-wise mean daily returns by specified factor quantile.
    """

    demeaned_fr = utils.demean_forward_returns(forward_returns,
                                               by_sector=by_sector)

    quantized_factor = quantized_factor.rename('quantile')
    forward_returns_quantile = (pd.DataFrame(quantized_factor)
                                .merge(demeaned_fr, how='left',
                                       left_index=True, right_index=True)
                                .set_index('quantile', append=True))

    grouper = []
    if by_time is not None:
        grouper.append(pd.TimeGrouper(by_time, level='date'))

    if by_sector:
        grouper.append(
            forward_returns_quantile.index.get_level_values('sector'))

    grouper.append(forward_returns_quantile.index.get_level_values('quantile'))

    mean_std = forward_returns_quantile.groupby(grouper).agg(['mean', 'std'])

    mean_ret = mean_std.T.xs('mean', level=1).T
    std_ret = mean_std.T.xs('std', level=1).T

    if std:
        return mean_ret, std_ret

    return mean_ret


def compute_mean_returns_spread(mean_returns, upper_quant,
                                lower_quant, std=None):
    """
    Computes the difference between the mean returns of
    two quantiles. Optionally, computes the standard deviation
    of this difference.

    Parameters
    ----------
    mean_returns : pd.DataFrame
        DataFrame of mean daily returns by quantile.
        MultiIndex containing date and quantile.
        See mean_return_by_quantile.
    upper_quant : integer
        Quantile of mean return from which we wish to subtract
        lower quantile mean return
    lower_quant : integer
        Quantile of mean return we wish to subtract from
        upper quantile mean return
    std : pd.DataFrame (optional)
        Daily standard deviation in mean return by quantile.
        Takes the same for as mean_returns.

    Returns
    -------
    mean_return_difference : pd.Series
        Daily difference in quantile returns.
    joint_std : pd.Series
        Daily standard deviation of the difference in
        quantile returns.
    """

    mean_return_difference = mean_returns.xs(upper_quant, level='quantile') - \
        mean_returns.xs(lower_quant, level='quantile')

    if std is not None:
        std1 = std.xs(upper_quant, level='quantile')
        std2 = std.xs(lower_quant, level='quantile')
        joint_std = np.sqrt(std1**2 + std2**2)

        return mean_return_difference, joint_std

    return mean_return_difference


def quantile_turnover(quantile_factor, quantile):
    """
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.

    Parameters
    ----------
    quantile_factor : pd.Series
        DataFrame with date, equity and factor quantile.
    quantile : integer
        Quantile on which to perform turnover analysis.

    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    quant_names = quantile_factor[quantile_factor == quantile]
    quant_name_sets = quant_names.groupby(level=['date']).apply(
        lambda x: set(x.index.get_level_values('equity')))
    new_names = (quant_name_sets - quant_name_sets.shift(1)).dropna()
    quant_turnover = new_names.apply(
        lambda x: len(x)) / quant_name_sets.apply(lambda x: len(x))

    return quant_turnover


def factor_rank_autocorrelation(factor, time_rule='W', by_sector=False):
    """
    Computes autocorrelation of mean factor ranks in specified timespans.
    We must compare week to week factor ranks rather than factor values to account for
    systematic shifts in the factor values of all names or names within a sector.
    This metric is useful for measuring the turnover of a factor. If the value of a factor
    for each name changes randomly from week to week, we'd expect a weekly autocorrelation of 0.

    Parameters
    ----------
    factor : pd.Series
        Series with date and equity index. Values are factor values.
    time_rule : string, optional
        Time span to use in factor grouping mean reduction.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html for available options.
    by_sector : boolean
        If True, compute autocorrelation separately for each sector.

    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of factor values.

    """

    grouper = ['date', 'sector'] if by_sector else ['date']

    daily_ranks = factor.groupby(level=grouper).apply(
        lambda x: x.rank(ascending=True))

    daily_ranks.name = "factor"
    daily_ranks = pd.DataFrame(daily_ranks)

    equity_factor_rank = daily_ranks.reset_index().pivot(
        index='date', columns='equity', values='factor')
    if time_rule is not None:
        equity_factor_rank = equity_factor_rank.resample(time_rule).mean()

    autocorr = equity_factor_rank.corrwith(equity_factor_rank.shift(1), axis=1)

    return autocorr
