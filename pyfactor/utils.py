import pandas as pd
from IPython.display import display


def compute_forward_returns(prices, days=(1, 5, 10)):
    """
    Finds the N day forward returns (as percent change) for each equity provided.
    Parameters
    ----------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation. Equities as columns, dates as index.
        Pricing data must span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected days in the forward returns calculations.
    days : list
        Number of days forward to project returns. One column will be added for each value.
    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        DataFrame containg the N day forward returns for a security.
    """

    forward_returns = pd.DataFrame(index=pd.MultiIndex.from_product(
        [prices.index, prices.columns], names=['date', 'equity']))
    for day in days:
        delta = prices.pct_change(day).shift(-day)
        forward_returns[day] = delta.stack()

    return forward_returns


def sector_adjust_forward_returns(forward_returns):
    """
    Convert forward price movements to price movements relative to mean sector price movements.
    This normalization incorporates the assumption of a sector neutral portfolio constraint
    and thus allows allows the factor to be evaluated across sectors.

    For example, if AAPL 5 day return is 0.1% and mean 5 day return for the Technology stocks
    in our universe was 0.5% in the same period, the sector adjusted 5 day return for AAPL
    in this period is -0.4%.


    Parameters
    ----------
    forward_returns : pd.DataFrame - MultiIndex
        DataFrame with date, equity, sector, and forward returns columns.
        See compute_forward_returns for more detail.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each security's returns normalized by sector.

    """

    return forward_returns.groupby(level=['date', 'sector']).apply(lambda x: x - x.mean())


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
        end = timestamp_idx + days_after
        series = daily_perc_ret.ix[start:end, equity]
        ret_df = pd.concat([ret_df, series], axis=1, ignore_index=True)

    # Reset index to have the same starting point (from datetime to day offset)
    ret_df = ret_df.apply(lambda x: x.dropna().reset_index(drop=True), axis=0)
    ret_df.index = range(-days_before, days_after)

    # From daily percent returns to comulative returns
    ret_df = (ret_df + 1).cumprod() - 1

    # Make returns be 0 at day 0
    if day_zero_align:
        ret_df -= ret_df.iloc[days_before, :]

    return ret_df


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
