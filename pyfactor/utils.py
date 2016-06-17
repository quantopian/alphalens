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


def demean_forward_returns(forward_returns, by_sector=False):
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


def format_input_data(factor, prices, sectors=None, days=(1, 5, 10)):
    """
    Formats the factor data, pricing data, and sector mappings into DataFrames and Series that
    contain aligned MultiIndex indices containing date, equity, and sector.
    ----------
    ----------
    factor : pandas.Series - MultiIndex
        A list of equities and their factor values indexed by date.
    prices : pd.DataFrame
        Pricing data to use in forward price calculation. Equities as columns, dates as index.
        Pricing data must span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected days in the forward returns calculations.
    sectors : pd.Series - MultiIndex
        A list of equities and their sectors
    days : list
        Number of days forward to project returns. One column will be added for each value.
    Returns
    -------
    factor : pd.Series
        A list of equities and their factor values indexed by date, equity, and optionally sector.
    forward_returns : pd.DataFrame - MultiIndex
        A DataFrame of equities and their forward returns indexed by date, equity, and optionally sector.
        Note: this is the same index as the factor index
    """

    forward_returns = compute_forward_returns(prices, days)

    merged_data = pd.merge(pd.DataFrame(factor),
                           forward_returns,
                           how='left',
                           left_index=True,
                           right_index=True)
    if sectors is not None:
        merged_data = pd.merge(pd.DataFrame(sectors),
                               merged_data,
                               how='left',
                               left_index=True,
                               right_index=True)
        merged_data = merged_data.set_index(
            [merged_data.index, merged_data["sector"]])
        merged_data = merged_data.drop("sector", 1)

    merged_data = merged_data.dropna()

    factor = merged_data.pop("factor")
    forward_returns = merged_data

    return factor, forward_returns
