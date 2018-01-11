#
# Copyright 2017 Quantopian, Inc.
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
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
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
        _ic = group[utils.get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values('date')]

    if group_adjust:
        factor_data = utils.demean_forward_returns(factor_data,
                                                   grouper + ['group'])
    if by_group:
        grouper.append('group')

    ic = factor_data.groupby(grouper).apply(src_ic)

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
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
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
        grouper.append(pd.Grouper(freq=by_time))
    if by_group:
        grouper.append('group')

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        ic = (ic.reset_index().set_index('date').groupby(grouper).mean())

    return ic


def factor_weights(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False):
    """
    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights. 

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Should this computation happen on a long short portfolio? if True,
        weights are computed by demeaning factor values and dividing by the sum
        of their absolute value (achieving gross leverage of 1). The sum of
        positive weights will be the same as the negative weights (absolute
        value), suitable for a dollar neutral long-short portfolio
    group_adjust : bool
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral weights: each group will weight the same and
        if 'demeaned' is enabled the factor values demeaning will occur on the
        group level.
    equal_weight : bool, optional
        if True the assets will be equal-weighted instead of factor-weighted

    Returns
    -------
    returns : pd.Series
        Assets weighted by factor value.
    """

    def to_weights(group, _demeaned, _equal_weight):

        if _demeaned:
            group = group - group.mean()
        else:
            group = group.copy()

        if _equal_weight:

            negative_mask = group < 0
            group[negative_mask] = -1.0
            neg_amount = negative_mask.sum()
            if _demeaned and neg_amount > 0:
                group[negative_mask] /= neg_amount

            positive_mask = group > 0
            group[positive_mask] = 1.0
            pos_amount = positive_mask.sum()
            if _demeaned and pos_amount > 0:
                group[positive_mask] /= pos_amount

        return group / group.abs().sum()

    grouper = [factor_data.index.get_level_values('date')]
    if group_adjust:
        grouper.append('group')

    weights = factor_data.groupby(grouper)['factor'] \
        .apply(to_weights, demeaned, equal_weight)

    if group_adjust:
        weights = weights.groupby(level='date').apply(to_weights, False, False)

    # preserve freq, which contains trading calendar information
    weights.index.levels[0].freq = factor_data.index.levels[0].freq
    return weights


def factor_returns(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   by_asset=False):
    """
    Computes period wise returns for portfolio weighted by factor
    values.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    group_adjust : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    equal_weight : bool, optional
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    by_asset: bool, optional
        If True, returns are reported separately for each esset.

    Returns
    -------
    returns : pd.DataFrame
        Period wise factor returns
    """

    weights = \
        factor_weights(factor_data, demeaned, group_adjust, equal_weight)

    weighted_returns = \
        factor_data[utils.get_forward_returns_columns(factor_data.columns)] \
        .multiply(weights, axis=0)

    if by_asset:
        returns = weighted_returns
    else:
        returns = weighted_returns.groupby(level='date').sum()

    # preserve freq, which contains trading calendar information
    returns.index.freq = factor_data.index.levels[0].freq
    return returns


def factor_alpha_beta(factor_data,
                      returns=None,
                      demeaned=True,
                      group_adjust=False,
                      equal_weight=False):
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
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame, optional
        Period wise factor returns. If this is None then it will be computed
        with 'factor_returns' function and the passed flags: 'demeaned',
        'group_adjust', 'equal_weight'
    demeaned : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    group_adjust : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    equal_weight : bool, optional
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation

    Returns
    -------
    alpha_beta : pd.Series
        A list containing the alpha, beta, a t-stat(alpha)
        for the given factor and forward returns.
    """

    if returns is None:
        returns = \
            factor_returns(factor_data, demeaned, group_adjust, equal_weight)

    universe_ret = factor_data.groupby(level='date')[
        utils.get_forward_returns_columns(factor_data.columns)] \
        .mean().loc[returns.index]

    if isinstance(returns, pd.Series):
        returns.name = universe_ret.columns.values[0]
        returns = pd.DataFrame(returns)

    alpha_beta = pd.DataFrame()
    for period in returns.columns.values:
        x = universe_ret[period].values
        y = returns[period].values
        x = add_constant(x)

        reg_fit = OLS(y, x).fit()
        try:
            alpha, beta = reg_fit.params
        except ValueError:
            alpha_beta.loc['Ann. alpha', period] = np.nan
            alpha_beta.loc['beta', period] = np.nan
        else:
            freq_adjust = pd.Timedelta('252Days') / pd.Timedelta(period)

            alpha_beta.loc['Ann. alpha', period] = \
                (1 + alpha) ** freq_adjust - 1
            alpha_beta.loc['beta', period] = beta

    return alpha_beta


def cumulative_returns(returns, period, freq=None):
    """
    Builds cumulative returns from 'period' returns. This function simulate the
    cumulative effect that a series of gains or losses (the 'retuns') have on
    an original amount of capital over a period of time.

    if F is the frequency at which returns are computed (e.g. 1 day if
    'returns' contains daily values) and N is the period for which the retuns
    are computed (e.g. returns after 1 day, 5 hours or 3 days) then:
    - if N <= F the cumulative retuns are trivially computed as Compound Return
    - if N > F (e.g. F 1 day, and N is 3 days) then the returns overlap and the
      cumulative returns are computed building and averaging N interleaved sub
      portfolios (started at subsequent periods 1,2,..,N) each one rebalancing
      every N periods. This correspond to an algorithm which trades the factor
      every single time it is computed, which is statistically more robust and
      with a lower volatity compared to an algorithm that trades the factor
      every N periods and whose returns depend on the specific starting day of
      trading.

    Also note that when the factor is not computed at a specific frequency, for
    exaple a factor representing a random event, it is not efficient to create
    multiples sub-portfolios as it is not certain when the factor will be
    traded and this would result in an underleveraged portfolio. In this case
    the simulated portfolio is fully invested whenever an event happens and if
    a subsequent event occur while the portfolio is still invested in a
    previous event then the portfolio is rebalanced and split equally among the
    active events.

    Parameters
    ----------
    returns: pd.Series
        pd.Series containing factor 'period' forward returns, the index
        contains timestamps at which the trades are computed and the values
        correspond to returns after 'period' time
    period: pandas.Timedelta or string
        Length of period for which the returns are computed (1 day, 2 mins,
        3 hours etc). It can be a Timedelta or a string in the format accepted
        by Timedelta constructor ('1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        returns.index.freq will be used

    Returns
    -------
    pd.Series
        Cumulative returns series
    """

    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)

    if freq is None:
        freq = returns.index.freq

    #
    # returns index contains factor computation timestamps, then add returns
    # timestamps too (factor timestamps + period) and save them to 'full_idx'
    # Cumulative returns will use 'full_idx' index,because we want a cumulative
    # returns value for each entry in 'full_idx'
    #
    trades_idx = returns.index.copy()
    returns_idx = utils.add_custom_calendar_timedelta(trades_idx, period, freq)
    full_idx = trades_idx.union(returns_idx)

    #
    # Build N sub_returns from the single returns Series. Each sub_retuns
    # stream will contain non overlapping retuns.
    # In the next step we'll compute the portfolio returns averaging the
    # returns happening on those overlapping returns streams
    #
    sub_returns = []
    while len(trades_idx) > 0:

        #
        # select non-overlapping returns starting with first timestamp in index
        #
        sub_index = []
        next = trades_idx.min()
        while next <= trades_idx.max():
            sub_index.append(next)
            next = utils.add_custom_calendar_timedelta(next, period, freq)
            # make sure to fetch the next available entry after 'period'
            try:
                i = trades_idx.get_loc(next, method='bfill')
                next = trades_idx[i]
            except KeyError:
                break

        sub_index = pd.DatetimeIndex(sub_index, tz=full_idx.tz)
        subret = returns[sub_index]

        # make the index to have all entries in 'full_idx'
        subret = subret.reindex(full_idx)

        #
        # compute intermediate returns values for each index in subret that are
        # in between the timestaps at which the factors are computed and the
        # timestamps at which the 'period' actually returns happen
        #
        for pret_idx in reversed(sub_index):

            pret = subret[pret_idx]

            # get all timestamps between factor computation and period returns
            pret_end_idx = \
                utils.add_custom_calendar_timedelta(pret_idx, period, freq)
            slice = subret[(subret.index > pret_idx) & (
                subret.index <= pret_end_idx)].index

            if pd.isnull(pret):
                continue

            def rate_of_returns(ret, period):
                return ((np.nansum(ret) + 1)**(1. / period)) - 1

            # compute intermediate 'period' returns values, note that this also
            # moves the final 'period' returns value from trading timestamp to
            # trading timestamp + 'period'
            for slice_idx in slice:
                sub_period = (pret_end_idx - pret_idx) / (slice_idx - pret_idx)
                subret[slice_idx] = rate_of_returns(pret, sub_period)

            subret[pret_idx] = np.nan

            # transform returns as percentage change from previous value
            subret[slice[1:]] = (subret[slice] + 1).pct_change()[slice[1:]]

        sub_returns.append(subret)
        trades_idx = trades_idx.difference(sub_index)

    #
    # Compute portfolio cumulative returns averaging the returns happening on
    # overlapping returns streams. Please note that the below algorithm keeps
    # into consideration the scenario where a factor is not computed at a fixed
    # frequency (e.g. every day) and consequently the returns appears randomly
    #
    sub_portfolios = pd.concat(sub_returns, axis=1)
    portfolio = pd.Series(index=sub_portfolios.index)

    for i, (index, row) in enumerate(sub_portfolios.iterrows()):

        # check the active portfolios, count() returns non-nans elements
        active_subfolios = row.count()

        # fill forward portfolio value
        portfolio.iloc[i] = portfolio.iloc[i - 1] if i > 0 else 1.

        if active_subfolios <= 0:
            continue

        # current portfolio is the average of active sub_portfolios
        portfolio.iloc[i] *= (row + 1).mean(skipna=True)

    return portfolio


def mean_return_by_quantile(factor_data,
                            by_date=False,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
    demeaned : bool
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level.

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values('date')] + ['group']
        factor_data = utils.demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = utils.demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ['factor_quantile']
    if by_date:
        grouper.append(factor_data.index.get_level_values('date'))

    if by_group:
        grouper.append('group')

    group_stats = factor_data.groupby(grouper)[
        utils.get_forward_returns_columns(factor_data.columns)] \
        .agg(['mean', 'std', 'count'])

    mean_ret = group_stats.T.xs('mean', level=1).T

    std_error_ret = group_stats.T.xs('std', level=1).T \
        / np.sqrt(group_stats.T.xs('count', level=1).T)

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

    mean_return_difference = mean_returns.xs(upper_quant,
                                             level='factor_quantile') \
        - mean_returns.xs(lower_quant, level='factor_quantile')

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
    period: string or int, optional
        Period over which to calculate the turnover. If it is a string it must
        follow pandas.Timedelta constructor format (e.g. '1 days', '1D', '30m',
        '3h', '1D1h', etc).
    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    quant_names = quantile_factor[quantile_factor == quantile]
    quant_name_sets = quant_names.groupby(level=['date']).apply(
        lambda x: set(x.index.get_level_values('asset')))

    if isinstance(period, int):
        shift = period
    else:
        # find the frequency at which the factor is computed
        idx = quant_name_sets.index
        freq = min([idx[i] - idx[i-1] for i in range(1, min(10, len(idx)))])
        shift = int(pd.Timedelta(period) / freq)

    new_names = (quant_name_sets - quant_name_sets.shift(shift)).dropna()
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
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period: string or int, optional
        Period over which to calculate the turnover. If it is a string it must
        follow pandas.Timedelta constructor format (e.g. '1 days', '1D', '30m',
        '3h', '1D1h', etc).
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

    if isinstance(period, int):
        shift = period
    else:
        # find the frequency at which the factor is computed
        idx = asset_factor_rank.index
        freq = min([idx[i] - idx[i-1] for i in range(1, min(10, len(idx)))])
        shift = int(pd.Timedelta(period) / freq)

    autocorr = asset_factor_rank.corrwith(asset_factor_rank.shift(shift),
                                          axis=1)
    autocorr.name = period
    return autocorr


def common_start_returns(factor,
                         prices,
                         before,
                         after,
                         cumulative=False,
                         mean_by_date=False,
                         demean_by=None):
    """
    A date and equity pair is extracted from each index row in the factor
    dataframe and for each of these pairs a return series is built starting
    from 'before' the date and ending 'after' the date specified in the pair.
    All those returns series are then aligned to a common index (-before to
    after) and returned as a single DataFrame

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame with at least date and equity as index, the columns are
        irrelevant
    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets
        in the columns. Pricing data should span the factor
        analysis time period plus/minus an additional buffer window
        corresponding to after/before period parameters.
    before:
        How many returns to load before factor date
    after:
        How many returns to load after factor date
    cumulative: bool, optional
        Return cumulative returns
    mean_by_date: bool, optional
        If True, compute mean returns for each date and return that
        instead of a return series for each asset
    demean_by: pd.DataFrame, optional
        DataFrame with at least date and equity as index, the columns are
        irrelevant. For each date a list of equities is extracted from
        'demean_by' index and used as universe to compute demeaned mean
        returns (long short portfolio)

    Returns
    -------
    aligned_returns : pd.DataFrame
        Dataframe containing returns series for each factor aligned to the same
        index: -before to after
    """

    if cumulative:
        returns = prices
    else:
        returns = prices.pct_change(axis=0)

    all_returns = []

    for timestamp, df in factor.groupby(level='date'):

        equities = df.index.get_level_values('asset')

        try:
            day_zero_index = returns.index.get_loc(timestamp)
        except KeyError:
            continue

        starting_index = max(day_zero_index - before, 0)
        ending_index = min(day_zero_index + after + 1,
                           len(returns.index))

        equities_slice = set(equities)
        if demean_by is not None:
            demean_equities = demean_by.loc[timestamp] \
                .index.get_level_values('asset')
            equities_slice |= set(demean_equities)

        series = returns.loc[returns.index[starting_index:ending_index],
                             equities_slice]
        series.index = range(starting_index - day_zero_index,
                             ending_index - day_zero_index)

        if cumulative:
            series = (series / series.loc[0, :]) - 1

        if demean_by is not None:
            mean = series.loc[:, demean_equities].mean(axis=1)
            series = series.loc[:, equities]
            series = series.sub(mean, axis=0)

        if mean_by_date:
            series = series.mean(axis=1)

        all_returns.append(series)

    return pd.concat(all_returns, axis=1)


def average_cumulative_return_by_quantile(factor_data,
                                          prices,
                                          periods_before=10,
                                          periods_after=15,
                                          demeaned=True,
                                          group_adjust=False,
                                          by_group=False):
    """
    Plots average cumulative returns by factor quantiles in the period range
    defined by -periods_before to periods_after

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
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
    group_adjust : bool
        Returns demeaning will occur on the group level (group
        neutral portfolio)
    by_group : bool
        If True, compute cumulative returns separately for each group
    Returns
    -------
    cumulative returns and std deviation : pd.DataFrame
        A MultiIndex DataFrame indexed by quantile (level 0) and mean/std
        (level 1) and the values on the columns in range from
        -periods_before to periods_after
        If by_group=True the index will have an additional 'group' level
        ::
            ---------------------------------------------------
                        |       | -2  | -1  |  0  |  1  | ...
            ---------------------------------------------------
              quantile  |       |     |     |     |     |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 1      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                        | mean  |  x  |  x  |  x  |  x  |
                 2      ---------------------------------------
                        | std   |  x  |  x  |  x  |  x  |
            ---------------------------------------------------
                ...     |                 ...
            ---------------------------------------------------
    """

    def cumulative_return(q_fact, demean_by):
        return common_start_returns(q_fact, prices,
                                    periods_before,
                                    periods_after,
                                    True, True, demean_by)

    def average_cumulative_return(q_fact, demean_by):
        q_returns = cumulative_return(q_fact, demean_by)
        return pd.DataFrame({'mean': q_returns.mean(axis=1),
                             'std': q_returns.std(axis=1)}).T

    if by_group:
        #
        # Compute quantile cumulative returns separately for each group
        # Deman those returns accordingly to 'group_adjust' and 'demeaned'
        #
        returns_bygroup = []

        for group, g_data in factor_data.groupby('group'):
            g_fq = g_data['factor_quantile']
            if group_adjust:
                demean_by = g_fq  # demeans at group level
            elif demeaned:
                demean_by = factor_data['factor_quantile']  # demean by all
            else:
                demean_by = None
            #
            # Align cumulative return from different dates to the same index
            # then compute mean and std
            #
            avgcumret = g_fq.groupby(g_fq).apply(average_cumulative_return,
                                                 demean_by)
            avgcumret['group'] = group
            avgcumret.set_index('group', append=True, inplace=True)
            returns_bygroup.append(avgcumret)

        return pd.concat(returns_bygroup, axis=0)

    else:
        #
        # Compute quantile cumulative returns for the full factor_data
        # Align cumulative return from different dates to the same index
        # then compute mean and std
        # Deman those returns accordingly to 'group_adjust' and 'demeaned'
        #
        if group_adjust:
            all_returns = []
            for group, g_data in factor_data.groupby('group'):
                g_fq = g_data['factor_quantile']
                avgcumret = g_fq.groupby(g_fq).apply(cumulative_return, g_fq)
                all_returns.append(avgcumret)
            q_returns = pd.concat(all_returns, axis=1)
            q_returns = pd.DataFrame({'mean': q_returns.mean(axis=1),
                                      'std': q_returns.std(axis=1)})
            return q_returns.unstack(level=1).stack(level=0)
        elif demeaned:
            fq = factor_data['factor_quantile']
            return fq.groupby(fq).apply(average_cumulative_return, fq)
        else:
            fq = factor_data['factor_quantile']
            return fq.groupby(fq).apply(average_cumulative_return, None)


def create_pyfolio_input(factor_data,
                         period,
                         long_short=True,
                         group_neutral=False,
                         quantiles=None,
                         groups=None):
    """

    WARNING: this API is still in experimental phase and input/output
             paramenters might change in the future

    Simulate a portfolio using the factor in input and returns a DataFrames
    containing the portfolio returns formatted for pyfolio.

    For more details on how this portfolio is built see:
    - performance.factor_returns (how assets weights are computed)
    - performance.cumulative_returns (how the portfolio returns are computed)

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns
    long_short : bool, optional
        Should this computation happen on a long short portfolio? if so, then
        factor values will be demeaned across the factor universe when factor
        weighting the portfolio for cumulative returns plots
    group_neutral : bool, optional
        Should this computation happen on a group neutral portfolio? if so,
        factor values demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots
    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used
    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used


    Returns
    -------
     returns : pd.Series
        Daily returns of the strategy, noncumulative.

     benchmark : pd.Series
        Benchmark returns computed as the factor universe mean daily returns.
        If '1D' period column is not present in the factor_data the beanchmark
        returns is not computed and returned as 'None'
    """

    portfolio_data = factor_data

    if quantiles is not None:
        portfolio_data = portfolio_data[portfolio_data['factor_quantile'].isin(
            quantiles)]

    if groups is not None:
        portfolio_data = portfolio_data[portfolio_data['group'].isin(groups)]

    #
    # Build returns:
    # we don't know the frequency at which the factor returns are computed but
    # pyfolio wants daily returns. So we compute the cumulative returns of the
    # factor, then resample it at 1 day frequency and finally compute daily
    # returns
    #
    returns = factor_returns(portfolio_data, long_short, group_neutral)
    returns = cumulative_returns(returns[period], period)
    returns = returns.resample('1D').last()
    returns = returns.pct_change().fillna(0)

    #
    #
    #
    # Build benchmark returns as the factor universe mean daily return
    #
    if '1D' in utils.get_forward_returns_columns(factor_data.columns):
        benchmark_rets = factor_data.groupby(level='date')['1D'] \
                                    .mean().loc[returns.index]
    else:
        benchmark_rets = None
        print("Cannot computed benchmark daily returns: '1D' not present"
              " in 'factor_data' periods column")

    return returns, benchmark_rets
