import pandas as pd
import numpy as np


def compute_forward_price_movement(prices, days=[1, 5, 10]):
    """
    Adds N day forward price movements (as percent change) to a factor value
    DataFrame.

    Parameters
    ----------
    daily_factor : pd.DataFrame
        DataFrame with, at minimum, date, equity, factor, columns. Index can be integer or
        date/equity multiIndex.
        See construct_factor_history for more detail.
    days : list
        Number of days forward to project price movement. One column will be added for each value.
    prices : pd.DataFrame, optional
        Pricing data to use in forward price calculation. Equities as columns, dates as index.


    Returns
    -------
    factor_and_fp : pd.DataFrame
        DataFrame with integer index and date, equity, factor, sector
        code columns with and an arbitary number of N day forward percentage
        price movement columns.

    """

    forward_prices = pd.DataFrame(index=pd.MultiIndex.from_product(
        [prices.index.values, prices.columns.values], names=['date', 'equity']))
    for i in days:
        delta = prices.pct_change(i).shift(-i)
        forward_prices[i] = delta.stack()

    return forward_prices


def sector_adjust_forward_price_moves(prices):
    """
    Convert forward price movements to price movements relative to mean sector price movements.
    This normalization incorperates the assumption of a sector neutral portfolio constraint
    and thus allows allows the factor to be evaluated across sectors.

    For example, if AAPL 5 day return is 0.1% and mean 5 day return for the Technology stocks
    in our universe was 0.5% in the same period, the sector adjusted 5 day return for AAPL
    in this period is -0.4%.


    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns. Index should be integer.
        See add_forward_price_movement for more detail.

    Returns
    -------
    adj_factor_and_fp : pd.DataFrame
        DataFrame with integer index and date, equity, factor, sector
        code columns with and an arbitary number of N day forward percentage
        price movement columns, each normalized by sector.

    """
    adj_factor_and_fp = factor_and_fp.copy()

    adj_factor_and_fp[pc_cols] = factor_and_fp.groupby(['date', 'sector_code'])[pc_cols].apply(
             lambda x: x - x.mean())

    return adj_factor_and_fp

def build_cumulative_returns_series(factor_and_fp, daily_perc_ret, days_before, days_after, day_zero_align=False):
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

    ret_df = pd.DataFrame()

    for index, row in factor_and_fp.iterrows():
        timestamp, equity = row['date'], row['equity']
        timestamp_idx = daily_perc_ret.index.get_loc(timestamp)
        start = timestamp_idx - days_before
        end   = timestamp_idx + days_after
        series = daily_perc_ret.ix[start:end, equity]
        ret_df = pd.concat( [ret_df, series], axis=1, ignore_index=True)

    # Reset index to have the same starting point (from datetime to day offset)
    ret_df = ret_df.apply(lambda x : x.dropna().reset_index(drop=True), axis=0)
    ret_df.index = range(-days_before, days_after)

    # From daily percent returns to comulative returns
    ret_df  = (ret_df  + 1).cumprod() - 1

    # Make returns be 0 at day 0
    if day_zero_align:
        ret_df -= ret_df.iloc[days_before,:]

    return ret_df


def get_price_move_cols(x):
    pc_cols = [col for col in x.columns.values if 'fwd_price_change' in col]
    fwd_days = map(lambda x: int(x.split('_')[0]), pc_cols)

    return fwd_days, pc_cols

def get_ic_cols(x):
    return [col for col in x.columns.values if 'day_IC' in col]