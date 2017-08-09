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
from scipy import stats
from collections import OrderedDict

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from . import utils


def factor_information_coefficient(factor_data,
                                   group_adjust=False,
                                   by_group=False):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for each period,
        The factor quantile/bin that factor value belongs too, and (optionally) the group the
        asset belongs to.
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, compute period wise IC separately for each group.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    def src_ic(group):
        f = group['factor']
        _ic = group[utils.get_forward_returns_columns(factor_data.columns)].apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append('group')

    if group_adjust:
        factor_data = utils.demean_forward_returns(factor_data, grouper + ['group'])

    ic = factor_data.groupby(grouper).apply(src_ic)
    ic.columns = pd.Int64Index(ic.columns)

    return ic


def mean_information_coefficient(factor_data,
                                 group_adjust=False,
                                 by_group=False,
                                 by_time=None):
    """
    Get the mean information coefficient of specified groups.
    Answers questions like:
    What is the mean IC for each month?
    What is the mean IC for each group for our whole timerange?
    What is the mean IC for for each group, each week?

    Parameters
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for each period,
        The factor quantile/bin that factor value belongs too, and (optionally) the group the
        asset belongs to.
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, take the mean IC for each group.
    by_time : str (pd time_rule), optional
        Time window to use when taking mean IC.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        for available options.

    Returns
    -------
    ic : pd.DataFrame
        Mean Spearman Rank correlation between factor and provided
        forward price movement windows.
    """

    ic = factor_information_coefficient(factor_data, group_adjust, by_group)

    grouper = []
    if by_time is not None:
        grouper.append(pd.TimeGrouper(by_time))
    if by_group:
        grouper.append('group')

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        ic = (ic.reset_index().set_index('date').groupby(grouper).mean())

    ic.columns = pd.Int64Index(ic.columns)

    return ic


def factor_returns(factor_data, long_short=True, group_neutral=False):
    """
    Computes period wise returns for portfolio weighted by factor
    values. Weights are computed by demeaning factors and dividing
    by the sum of their absolute value (achieving gross leverage of 1).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for each period,
        The factor quantile/bin that factor value belongs too, and (optionally) the group the
        asset belongs to.
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then factor values
        will be demeaned across factor universe when factor weighting the portfolio.
    group_neutral : bool
        If True, compute group neutral returns: each group will weight
        the same and returns demeaning will occur on the group level.

    Returns
    -------
    returns : pd.DataFrame
        Period wise returns of dollar neutral portfolio weighted by factor value.
    """

    def to_weights(group, is_long_short):
        if is_long_short:
            demeaned_vals = group - group.mean()
            return demeaned_vals / demeaned_vals.abs().sum()
        else:
            return group / group.abs().sum()

    grouper = [factor_data.index.get_level_values('date')]
    if group_neutral:
        grouper.append('group')

    weights = factor_data.groupby(grouper)['factor'].apply(to_weights, long_short)

    if group_neutral:
        weights = weights.groupby(level='date').apply(to_weights, False)

    weighted_returns = factor_data[utils.get_forward_returns_columns(factor_data.columns)].multiply(weights, axis=0)

    returns = weighted_returns.groupby(level='date').sum()

    return returns


def factor_alpha_beta(factor_data, long_short=True):
    """
    Compute the alpha (excess returns), alpha t-stat (alpha significance),
    and beta (market exposure) of a factor. A regression is run with
    the period wise factor universe mean return as the independent variable
    and mean period wise return from a portfolio weighted by factor values
    as the dependent variable.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for each period,
        The factor quantile/bin that factor value belongs too, and (optionally) the group the
        asset belongs to.
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then factor values
        will be demeaned across factor universe when factor weighting the portfolio.
    Returns
    -------
    alpha_beta : pd.Series
        A list containing the alpha, beta, a t-stat(alpha)
        for the given factor and forward returns.
    """

    returns = factor_returns(factor_data, long_short=long_short)

    universe_ret = factor_data.groupby(level='date')[utils.get_forward_returns_columns(factor_data.columns)].mean().loc[returns.index]

    if isinstance(returns, pd.Series):
        returns.name = universe_ret.columns.values[0]
        returns = pd.DataFrame(returns)

    alpha_beta = pd.DataFrame()
    for period in returns.columns.values:
        x = universe_ret[period].values
        y = returns[period].values
        x = add_constant(x)

        reg_fit = OLS(y, x).fit()
        alpha, beta = reg_fit.params

        alpha_beta.loc['Ann. alpha', period] = (1 + alpha) ** (252.0/period) - 1
        alpha_beta.loc['beta', period] = beta

    return alpha_beta


def mean_return_by_quantile(factor_data,
                            by_date=False,
                            by_group=False,
                            demeaned=True):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for each period,
        The factor quantile/bin that factor value belongs too, and (optionally) the group the
        asset belongs to.
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
        Returns demeaning will occur on the group level.
    demeaned : bool
        Compute demeaned mean returns (long short portfolio)

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if demeaned:
        factor_data = utils.demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ['factor_quantile']
    if by_date:
        grouper.append(factor_data.index.get_level_values('date'))

    if by_group:
        grouper.append('group')


    group_stats = factor_data.groupby(grouper)[utils.get_forward_returns_columns(factor_data.columns)].agg(['mean', 'std', 'count'])

    mean_ret = group_stats.T.xs('mean', level=1).T

    std_error_ret = group_stats.T.xs('std', level=1).T /\
                    np.sqrt(group_stats.T.xs('count', level=1).T)

    return mean_ret, std_error_ret


def compute_mean_returns_spread(mean_returns,
                                upper_quant,
                                lower_quant,
                                std_err=None):
    """
    Computes the difference between the mean returns of
    two quantiles. Optionally, computes the standard error
    of this difference.

    Parameters
    ----------
    mean_returns : pd.DataFrame
        DataFrame of mean period wise returns by quantile.
        MultiIndex containing date and quantile.
        See mean_return_by_quantile.
    upper_quant : int
        Quantile of mean return from which we
        wish to subtract lower quantile mean return.
    lower_quant : int
        Quantile of mean return we wish to subtract
        from upper quantile mean return.
    std_err : pd.DataFrame
        Period wise standard error in mean return by quantile.
        Takes the same form as mean_returns.

    Returns
    -------
    mean_return_difference : pd.Series
        Period wise difference in quantile returns.
    joint_std_err : pd.Series
        Period wise standard error of the difference in quantile returns.
    """

    mean_return_difference = mean_returns.xs(upper_quant, level='factor_quantile') - \
        mean_returns.xs(lower_quant, level='factor_quantile')

    std1 = std_err.xs(upper_quant, level='factor_quantile')
    std2 = std_err.xs(lower_quant, level='factor_quantile')
    joint_std_err = np.sqrt(std1**2 + std2**2)

    return mean_return_difference, joint_std_err


def quantile_turnover(quantile_factor, quantile, period=1):
    """
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.

    Parameters
    ----------
    quantile_factor : pd.Series
        DataFrame with date, asset and factor quantile.
    quantile : int
        Quantile on which to perform turnover analysis.
    period: int, optional
        Period over which to calculate the turnover
    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    quant_names = quantile_factor[quantile_factor == quantile]
    quant_name_sets = quant_names.groupby(level=['date']).apply(
        lambda x: set(x.index.get_level_values('asset')))
    new_names = (quant_name_sets - quant_name_sets.shift(period)).dropna()
    quant_turnover = new_names.apply(
        lambda x: len(x)) / quant_name_sets.apply(lambda x: len(x))
    quant_turnover.name = quantile
    return quant_turnover


def factor_rank_autocorrelation(factor_data, period=1):
    """
    Computes autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a
    factor. If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for each period,
        The factor quantile/bin that factor value belongs too, and (optionally) the group the
        asset belongs to.
    period: int, optional
        Period over which to calculate the autocorrelation

    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of
        factor values.

    """

    grouper = [factor_data.index.get_level_values('date')]

    ranks = factor_data.groupby(grouper)['factor'].rank()

    asset_factor_rank = ranks.reset_index().pivot(index='date',
                                                  columns='asset',
                                                  values='factor')

    autocorr = asset_factor_rank.corrwith(asset_factor_rank.shift(period), axis=1)
    autocorr.name = period
    return autocorr


def average_cumulative_return_by_quantile(quantized_factor,
                                          prices,
                                          periods_before=10,
                                          periods_after=15,
                                          demeaned=True):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    quantized_factor : pd.Series
        Factor quantiles indexed by date and asset and
        optional a custom group.
    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets
        in the columns. Pricing data should span the factor
        analysis time period plus/minus an additional buffer window
        corresponding to periods_after/periods_before parameters.
    periods_before : int, optional
        How many periods before factor to plot
    periods_after  : int, optional
        How many periods after factor to plot
    demeaned : bool, optional
        Compute demeaned mean returns (long short portfolio)
    Returns
    -------
    pd.DataFrame indexed by quantile (level 0) and mean/std
    (level 1) and the values on the columns in range from
    -periods_before to periods_after
    """

    def average_cumulative_return(q_fact):
        demean = quantized_factor if demeaned else None
        q_returns = utils.common_start_returns(q_fact, prices,
                                               periods_before, periods_after,
                                               True, True, demean)
        return pd.DataFrame( {'mean': q_returns.mean(axis=1), 'std': q_returns.std(axis=1)} ).T

    return quantized_factor.groupby(quantized_factor).apply(average_cumulative_return)


def calc_vifs(data):
    '''
    Computes variance inflation factors (VIFs) for a set of random variables.
    For more information, see
    https://en.wikipedia.org/wiki/Variance_inflation_factor

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with observations of random variables.
        - Columns are observations of a random variable, and each row makes up
        one observation for each random variable.
    '''
    vifs = np.zeros(len(data.columns))
    for i in range(len(data.columns)):
        r2 = (OLS(data.iloc[:, i],
                  add_constant(data.drop(data.columns[i], axis='columns')))
              .fit()
              .rsquared)
        vifs[i] = 1/(1-r2)
    return pd.Series(index=data.columns, data=vifs)


def decompose_returns(algo_returns, risk_factors, hierarchy=None,
                      compute_t_stats=False, compute_vifs=False):
    '''
    Decomposes returns into returns attributable to common risk factors.

    Parameters
    ----------
    algo_returns : pd.Series
        Algorithm's daily returns, with a tz-localized index
        - Example:
        2010-01-01 00:00:00+00:00    0.812385
        2010-01-04 00:00:00+00:00    0.468386
        2010-01-05 00:00:00+00:00    0.680892
    risk_factors : pd.DataFrame
        Daily common risk factors. Index is tz-localized DatetimeIndex,
        columns are common factor names
        - Example:
                                        Mkt       SMB       HML       UMD
        2017-06-26 00:00:00+00:00    0.0004   -0.0008    0.0069   -0.0057
        2017-06-27 00:00:00+00:00   -0.0084   -0.0043    0.0130   -0.0030
        2017-06-28 00:00:00+00:00    0.0102    0.0077    0.0021    0.0091
    hierarchy : OrderedDict
        Hierarchy of common factors, for use in plotting the bar chart.
        If None, defaults to a hierarchy in which each column of risk_factors
        forms a 'level' of the hierarchy.
        - Example:
        OrderedDict([('Market', ['Mkt']),
                     ('Style', ['SMB', 'HML', 'UMD'])])
    compute_t_stats : bool
        If True, returns t-statistics for the betas. Defaults to False
    compute_vifs : bool
        If True, computes and returns variance inflation factors for the
        common factors. Defaults to False

    Returns
    -------
    returns_decomposition : pd.DataFrame
        Decomposed returns

    betas : pd.DataFrame
        Betas to the common factors

    vifs : pd.Series
        Variance inflation factors for the common factors
        - For more information, see
        https://en.wikipedia.org/wiki/Variance_inflation_factor
    '''
    if hierarchy is None:
        hierarchy = risk_factors.columns.values

    # construct a cumulative hierarchy
    cum_tiers = []
    for i in range(len(hierarchy.values())):
        cum_tiers.append([item for a in range(0, i+1)
                          for item in hierarchy.values()[a]])
    cum_hierarchy = OrderedDict()
    cum_hierarchy.update((hierarchy.keys()[i], cum_tiers[i])
                         for i in range(len(hierarchy)))

    # ensure that risk_factors and algo_rets_over_rf have the same index
    idx = algo_returns.index.intersection(risk_factors.index)
    risk_factors = risk_factors.loc[idx]
    algo_returns = algo_returns.loc[idx]

    idx = np.append(['Alpha'], risk_factors.columns.values)
    betas = pd.DataFrame(index=idx, columns=hierarchy.keys())
    returns_decomposition = pd.DataFrame(index=idx, columns=hierarchy.keys())

    if compute_t_stats:
        t_stats = pd.DataFrame(index=idx, columns=hierarchy.keys())

    for tier, factors in cum_hierarchy.iteritems():
        model_factors = add_constant(risk_factors.loc[:, factors]) \
            .rename(columns={'const': 'Alpha'})
        model = OLS(algo_returns, model_factors).fit()

        betas.loc[:, tier] = model.params
        betas.loc['Alpha', tier] *= 252

        if compute_t_stats:
            t_stats.loc[:, tier] = model.params / model.HC0_se

        returns_decomposition.loc[:, tier] = model.params.drop('Alpha') \
            * risk_factors.mean() * 252
        returns_decomposition.loc['Alpha', tier] = betas.loc['Alpha', tier]

    if compute_t_stats and compute_vifs:
        vifs = calc_vifs(risk_factors)
        return returns_decomposition, betas, t_stats, vifs
    elif compute_vifs:
        vifs = vifs(risk_factors)
        return returns_decomposition, betas, vifs
    elif compute_t_stats:
        return returns_decomposition, betas, t_stats
    else:
        return returns_decomposition, betas
