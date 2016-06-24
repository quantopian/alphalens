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

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import performance as perf
import utils
from itertools import izip
from functools import wraps


def plotting_context(func):
    """Decorator to set plotting context during function call."""
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            with context():
                # sns.set_style("whitegrid")
                sns.despine(left=True)
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return call_w_context


def context(context='notebook', font_scale=1.5, rc=None):
    """Create pyfolio default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5,
                     'axes.facecolor': '0.995',
                     'figure.facecolor': '0.97'}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    >>> with pyfolio.plotting.context(font_scale=2):
    >>>    pyfolio.create_full_tear_sheet()

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5,
                  # 'axes.facecolor': '0.995',
                  'axes.facecolor': 'white',
                  'axes.edgecolor': '1',
                  'figure.facecolor': '0.97'}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale,
                                rc=rc)


def summary_stats(ic_data, quantized_factor, mean_ret_quantile, autocorrelation_data, mean_ret_spread_quantile):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["IC Mean / Std."] = ic_data.mean() / ic_data.std()
    ic_summary_table["t-stat(IC)"] = sp.stats.ttest_1samp(ic_data, 0)[0]

    max_quantile = quantized_factor.values.max()
    min_quantile = quantized_factor.values.min()
    turnover_table = pd.DataFrame(columns=["Top Quantile", "Bottom Quantile"])
    turnover_table.loc["Mean Turnover"] = [perf.quantile_turnover(quantized_factor, max_quantile).mean(),
                                           perf.quantile_turnover(quantized_factor, min_quantile).mean()]

    returns_table = pd.DataFrame()
    returns_table["Mean Daily Return Top Quantile"] = mean_ret_quantile.loc[max_quantile]
    returns_table["Mean Daily Return Bottom Quantile"] = mean_ret_quantile.loc[min_quantile]

    auto_corr = pd.Series()
    auto_corr["Mean Factor Rank Autocorrelation"] = autocorrelation_data.mean()

    returns_table["Mean Daily Spread"] = mean_ret_spread_quantile.mean()

    print "Information Coefficient Analysis"
    utils.print_table(ic_summary_table.round(3).T)
    print "Returns Analysis"
    utils.print_table(returns_table.round(3).T)
    print "Turnover Analysis"
    utils.print_table(turnover_table.round(3))
    print auto_corr.round(3)


def plot_daily_ic_ts(daily_ic, return_ax=False):
    """
    Plots Spearman Rank Information Coefficient and IC moving average for a given factor.
    Sector neutralization of forward returns is recommended.

    Parameters
    ----------
    daily_ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    return_ax :
        The matplotlib figure object
    """

    num_plots = len(daily_ic.columns)
    f, axes = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
    axes = (a for a in axes.flatten())

    for ax, (days_num, ic) in izip(axes, daily_ic.iteritems()):
        title = "{} day IC".format(days_num)

        ic_df = (pd.DataFrame(ic.rename("{} day IC".format(days_num)))
                 .assign(**{'1 month moving avg': ic.rolling(22).mean()})
                 .plot(title=title, alpha=0.7, ax=ax))
        ax.set(ylabel='IC', xlabel="")
        ax.set_ylim([-0.25, 0.25])

    if return_ax:
        return axes


def plot_daily_ic_hist(daily_ic, return_ax=False):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.
    Sector neutralization of forward returns is recommended.

    Parameters
    ----------
    daily_ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    return_ax :
        The matplotlib figure object
    """

    num_plots = len(daily_ic.columns)

    v_spaces = num_plots // 3
    f, axes = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
    axes = (a for a in axes.flatten())

    for ax, (days_num, ic) in izip(axes, daily_ic.iteritems()):
        sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=ax)
        ax.set(title="%s day IC" % days_num, xlabel='IC')
        ax.set_xlim([-1, 1])

    if return_ax:
        return axes


def plot_quantile_returns_bar(mean_ret_by_q, by_sector=False):
    """
    Plots mean daily returns for factor quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (sector) and mean daily return values.
    by_sector : boolean
        Disagregate figures by sector.
    """

    if by_sector:
        num_sector = len(mean_ret_by_q.index.get_level_values('sector').unique())
        v_spaces = (num_sector + 1) // 2

        f, axes = plt.subplots(v_spaces, 2, sharex=False, sharey=True, figsize=(18, 6*v_spaces))
        axes = axes.flatten()

        for i, (sc, cor) in enumerate(mean_ret_by_q.groupby(level='sector')):
            cor.xs(sc, level='sector').plot(kind='bar', title=sc, ax=axes[i])
            axes[i].set_xlabel('')
            axes[i].set_ylabel('mean price % change')

        if i < len(axes):
            axes[-1] = None

        fig = plt.gcf()
        fig.suptitle("Mean Return By Factor Quantile By Sector", x=.5, y=.96)

    else:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
        mean_ret_by_q.plot(kind='bar',
                           title="Mean Return By Factor Quantile",
                           ax=ax)
        ax.set(xlabel='', ylabel='mean daily price % change')


def plot_mean_quintile_returns_spread_time_series(mean_returns_spread, std=None,
        title='Top Quintile - Bottom Quantile Mean Return'):
    if isinstance(mean_returns_spread, pd.DataFrame):
        for name, fr_column in mean_returns_spread.iteritems():
            stdn = None if std is None else std[name]
            plot_mean_quintile_returns_spread_time_series(fr_column, std=stdn,
                title=str(name) + " Day Forward Return " + title)
        return

    f, ax = plt.subplots(figsize=(18, 6))
    (pd.DataFrame(mean_returns_spread.rename('mean_return_spread'))
        .assign(**{'1 month moving avg': mean_returns_spread.rolling(22).mean()})
        .plot(alpha=0.7, ax=ax))
    mean_returns_spread.rolling(22).mean()
    if std is not None:
        upper = mean_returns_spread.values + std
        lower = mean_returns_spread.values - std
        ax.fill_between(mean_returns_spread.index, lower, upper, alpha=0.3)

    ax.set(ylabel='Difference in Quantile Mean Return (%)')
    ax.set(title=title, ylim=(-0.05, 0.05))


def plot_ic_by_sector(ic_sector):

    """
    Plots Spearman Rank Information Coefficient for a given factor over provided forward price
    movement windows. Separates by sector.

    Parameters
    ----------
    ic_sector : pd.DataFrame
        Sector-wise mean daily returns.
    """
    f, ax = plt.subplots(1, 1, figsize=(18, 6))
    ic_sector.plot(kind='bar', ax=ax)

    ax.set(title="Information Coefficient by Sector")
    

def plot_ic_by_sector_over_time(ic_time):
    """
    Plots sector-wise time window mean daily Spearman Rank Information Coefficient
    for a given factor over provided forward price movement windows.

    Parameters
    ----------
    ic_time : pd.DataFrame
        Sector-wise mean daily returns.
    """

    ic_time = ic_time.reset_index()

    f, axes = plt.subplots(6, 2, sharex=False, sharey=True, figsize=(18, 45))
    axes = axes.flatten()
    i = 0
    for sc, data in ic_time.groupby(['sector']):
        data.drop('sector', axis=1).set_index('date').plot(kind='bar',
                                                           title=sc,
                                                           ax=axes[i])
        i += 1
    fig = plt.gcf()
    fig.suptitle("Monthly Information Coefficient by Sector", fontsize=16, x=.5, y=.93)
    

def plot_factor_rank_auto_correlation(factor_autocorrelation):
    """
    Plots factor rank autocorrelation over time. See factor_rank_autocorrelation for more details.

    Parameters
    ----------
    daily_factor : pd.DataFrame
        DataFrame with date, equity, and factor value columns.
    time_rule : string, optional
        Time span to use in time grouping reduction prior to autocorrelation calculation.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html for available options.
    """

    f, ax = plt.subplots(1, 1, figsize=(18, 6))
    factor_autocorrelation.plot(title='Factor Rank Autocorrelation', ax=ax)
    ax.set(ylabel='autocorrelation coefficient')
    

def plot_top_bottom_quantile_turnover(quantized_factor):
    """
    Plots daily top and bottom quantile factor turnover.

    Parameters
    ----------
    quantized_factor : pd.Series
        Factor quantiles indexed by date and symbol.
    """

    max_quantile = quantized_factor.values.max()
    turnover = pd.DataFrame()
    turnover['top quantile turnover'] = perf.quantile_turnover(quantized_factor, max_quantile)
    turnover['bottom quantile turnover'] = perf.quantile_turnover(quantized_factor, 1)

    f, ax = plt.subplots(1, 1, figsize=(18, 6))
    turnover.plot(title='Top and Bottom Quantile Daily Turnover', ax=ax, alpha=0.6)
    ax.set(ylabel='proportion of names new to quantile', xlabel="")
    

def plot_monthly_ic_heatmap(mean_monthly_ic):
    """
    Plots a heatmap of the information coefficient by month.
    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N days forward.
    """

    num_plots = len(mean_monthly_ic.columns)

    v_spaces = num_plots // 3
    f, axes = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
    axes = (a for a in axes.flatten())

    for current_subplot, (days_num, ic) in zip(axes, mean_monthly_ic.iteritems()):

        formatted_ic = pd.Series(index=pd.MultiIndex.from_product([np.unique(ic.index.year), np.unique(ic.index.month)],
                                                     names=["year", "month"])[:len(ic)], data=ic.values)

        sns.heatmap(
            formatted_ic.unstack().round(3),
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 9},
            cbar=False,
            ax=current_subplot)
        current_subplot.set_ylabel("Year")
        current_subplot.set_xlabel("Month")
        current_subplot.set_title("Monthly Mean %s Day IC" % days_num)

