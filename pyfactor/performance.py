import pandas as pd
import numpy as np
import scipy as sp


def factor_information_coefficient(factor, forward_returns, time_rule=None, by_sector=True):
    """
    Computes sector neutral Spearman Rank Correlation based Information Coefficient between
    factor values and N day forward returns.

    Parameters
    ----------
    factor : pandas.Series - MultiIndex
        A list of equities and their factor values indexed by date.
    forward_returns : pandas.DataFrame - MultiIndex
        A list of equities and their N day forward returns where each column contains
        the N day forward returns
    time_rule : string, optional
        Time span to use in Pandas DateTimeIndex grouping reduction.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html for available options.
    by_sector : boolean
        If True, compute ic separately for each sector

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and provided forward price movement columns.
        MultiIndex of date, sector.
    err : pd.DataFrame
        Standard error of computed IC. MultiIndex of date, sector.
        MultiIndex of date, sector.

    """

    def src_ic(group):
        _ic = pd.Series(index=forward_returns.columns)
        f = group.pop('factor')
        for days in forward_returns.columns:
            _ic[days] = sp.stats.spearmanr(f, group[days])[0]
        _ic['obs_count'] = len(f)

        return _ic

    def src_std_error(rho, n):
        return np.sqrt((1-rho**2)/(n-2))

    factor_and_fp = pd.merge(pd.DataFrame(factor.rename('factor')),
                             forward_returns,
                             how='left',
                             left_index=True,
                             right_index=True)

    grouper = ['date', 'sector_code'] if by_sector else ['date']

    ic = factor_and_fp.groupby(level=grouper).apply(src_ic)

    obs_count = ic.pop('obs_count')
    err = ic.apply(lambda x: src_std_error(x, obs_count))

    if time_rule is not None:
        ic = ic.reset_index().set_index('date')
        err = err.reset_index().set_index('date')

        grpr = [pd.TimeGrouper(time_rule), 'sector_code'] if by_sector else [pd.TimeGrouper(time_rule)]
        ic = ic.groupby(grpr).mean()
        err = err.groupby(grpr).agg(
            lambda x: np.sqrt((np.sum(np.power(x, 2))/len(x))))
    else:
        if by_sector:
            ic = ic.reset_index().groupby(['sector_code']).mean()
            err = err.reset_index().groupby(['sector_code']).agg(
                lambda x: np.sqrt((np.sum(np.power(x, 2))/len(x))))

    return ic, err


def quantize_factor(factor, by_sector=False, quantiles=5):
    """
    Computes daily factor quantiles.

    Parameters
    ----------
    factor_and_fp : pd.Series
        DataFrame with date, equity, factor, and forward price movement columns.
        Index should be integer. See add_forward_price_movement for more detail.
    by_sector : boolean
        If True, compute quantile buckets separately for each sector.
    quantiles : integer
        Number of quantiles buckets to use in factor bucketing.
    factor_name : string
        Name of factor column on which to compute quantiles.

    Returns
    -------
    factor_and_fp_ : pd.DataFrame
        Factor and forward price movements with additional factor quantile column.
    """

    g_by = ['date', 'sector_code'] if by_sector else ['date']

    factor_percentile = factor.groupby(
                level=g_by).rank(pct=True)

    q_int_width = 1. / quantiles
    factor_quantile = factor_percentile.apply(
        lambda x: ((x - .000000001) // q_int_width) + 1)

    return factor_quantile


def mean_daily_return_by_factor_quantile(quantized_factor, forward_prices, by_sector=False):
    """
    Computes mean daily returns for factor quantiles across provided forward
    price movement columns.

    Parameters
    ----------
    quantile_factor : pd.DataFrame
        DataFrame with date, equity, factor, factor quantile, and forward price movement columns.
        Index should be integer. See quantile_bucket_factor for more detail.
    by_sector : boolean
        If True, compute quintile bucket returns separately for each sector
    quantiles : integer
        Number of quantiles buckets to use in factor bucketing.


    Returns
    -------
    mean_returns_by_quantile : pd.DataFrame
        Sector-wise mean daily returns by specified factor quantile.
    """
    quant_factor_fp = pd.merge(pd.DataFrame(quantized_factor.rename('quantile')),
                               forward_prices, how='left', left_index=True,
                               right_index=True)

    quant_factor_fp = quant_factor_fp.reset_index().set_index(
        ['date', 'equity', 'sector_code', 'quantile'] if by_sector
        else ['date', 'equity', 'quantile'])


    def daily_mean_ret(group):
        mean_ret = pd.Series(index=group.columns)
        for day_num, returns in group.iteritems():
            mean_ret[day_num] = returns.mean() / day_num

        return mean_ret

    g_by = ['sector_code', 'quantile'] if by_sector else ['quantile']
    mean_ret_by_quantile = quant_factor_fp.groupby(level=g_by).apply(daily_mean_ret)

    return mean_ret_by_quantile


def surrounding_cumulative_returns_by_quantile(quantized_factor, prices, days_before, days_after, day_zero_align=True):
    """
    An equity and date pair is extracted from each row in the input dataframe and for each of
    these pairs a cumulative return time series is built starting 'days_before' days
    before and ending 'days_after' days after the date specified in the pair

    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with at least date and equity columns.
    daily_perc_ret : pd.DataFrame
        Pricing data to use in cumulative return calculation. Equities as columns, dates as index.
    day_zero_align : boolean
         Aling returns at day 0 (timeseries is 0 at day 0)
    """
    window = days_before + days_after

    surrounding_rets = pd.DataFrame(index=pd.MultiIndex.from_product(
            [prices.index, prices.columns], names=['date', 'equity']))
    for i in range(-days_before, days_after):
        delta = prices.shift(-i).pct_change(1).shift(-1)
        surrounding_rets[i] = delta.stack()

    mean_cum_returns_by_quant = (pd.merge(pd.DataFrame(
        quantized_factor.rename('quantile')),
        surrounding_rets, how='left', left_index=True, right_index=True)
        .reset_index()
        .set_index(['date', 'equity', 'quantile'])
        .groupby(level='quantile')
        .mean()
        .T.add(1).cumprod().add(-1)
        )

    # Make returns be 0 at day 0
    if day_zero_align:
        mean_cum_returns_by_quant -= mean_cum_returns_by_quant.loc[0]

    return mean_cum_returns_by_quant


def quantile_turnover(quantile_factor, quantile):
    """
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.

    Parameters
    ----------
    quantile_factor : pd.DataFrame
        DataFrame with date, equity, factor, factor quantile, and forward price movement columns.
        Index should be integer. See quantile_bucket_factor for more detail.
    quantile : integer
        Quantile on which to perform turnover analysis.

    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    quant_names = quantile_factor[quantile_factor.factor_quantile == quantile]

    quant_name_sets = quant_names.groupby(['date']).equity.apply(set)
    new_names = (quant_name_sets - quant_name_sets.shift(1)).dropna()
    quant_turnover = new_names.apply(lambda x: len(x)) / quant_name_sets.apply(lambda x: len(x))

    return quant_turnover


def factor_rank_autocorrelation(daily_factor, time_rule='W', factor_name='factor'):
    """
    Computes autocorrelation of mean factor ranks in specified timespans.
    We must compare week to week factor ranks rather than factor values to account for
    systematic shifts in the factor values of all names or names within a sector.
    This metric is useful for measuring the turnover of a factor. If the value of a factor
    for each name changes randomly from week to week, we'd expect a weekly autocorrelation of 0.

    Parameters
    ----------
    daily_factor : pd.DataFrame
        DataFrame with integer index and date, equity, factor, and sector
        code columns.
    time_rule : string, optional
        Time span to use in factor grouping mean reduction.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html for available options.
    factor_name : string
        Name of factor column on which to compute IC.

    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of factor values.

    """
    daily_ranks = daily_factor.copy()
    daily_ranks[factor_name] = daily_factor.groupby(['date', 'sector_code'])[factor_name].apply(
        lambda x: x.rank(ascending=True))

    equity_factor = daily_ranks.pivot(index='date', columns='equity', values=factor_name)
    if time_rule is not None:
        equity_factor = equity_factor.resample(time_rule, how='mean')

    autocorr = equity_factor.corrwith(equity_factor.shift(1), axis=1)

    return autocorr
