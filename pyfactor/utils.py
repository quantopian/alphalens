def add_forward_price_movement(daily_factor, days=[1, 5, 10], prices=None):
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
        If no value is passed, get pricing will be called.

    Returns
    -------
    factor_and_fp : pd.DataFrame
        DataFrame with integer index and date, equity, factor, sector
        code columns with and an arbitary number of N day forward percentage
        price movement columns.

    """
    factor_and_fp = daily_factor.copy()
    if not isinstance(factor_and_fp.index, pd.core.index.MultiIndex):
        factor_and_fp = factor_and_fp.set_index(['date', 'equity'])

    if prices is None:
        start_date = factor_and_fp.index.levels[0].values.min()
        end_date = factor_and_fp.index.levels[0].values.max()

        equities = factor_and_fp.index.levels[1].unique()

        time_buffer = pd.Timedelta(days=max(days)+5)
        prices = get_pricing(equities,
                             start_date=start_date,
                             end_date=end_date+time_buffer,
                             fields='open_price')

    col_n = '%s_day_fwd_price_change'
    for i in days:
        delta = prices.pct_change(i).shift(-i)
        factor_and_fp[col_n%i] = delta.stack()

    factor_and_fp = factor_and_fp.reset_index()

    return factor_and_fp



def sector_adjust_forward_price_moves(factor_and_fp):
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
    pc_cols = [col for col in factor_and_fp.columns.values if 'fwd_price_change' in col]

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