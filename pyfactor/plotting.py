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
import seaborn as sns
import matplotlib.pyplot as plt
import performance as perf
import utils
from itertools import izip


sns.set(font_scale=2)


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
    f, axes = plt.subplots(num_plots, 1, figsize=(28, num_plots * 12))
    axes = (a for a in axes.flatten())

    summary_stats = pd.DataFrame(columns=['mean', 'std'])
    for ax, (days_num, ic) in izip(axes, daily_ic.iteritems()):
        title = "{} day IC".format(days_num)
        summary_stats.loc["%i day IC" % days_num] = [ic.mean(), ic.std()]

        ic_df = (pd.DataFrame(ic.rename("{} day IC".format(days_num)))
                 .assign(**{'1 month moving avg': ic.rolling(22).mean()})
                 .plot(title=title, alpha=0.7, ax=ax))
        ax.set(ylabel='IC', xlabel="")
        ax.set_ylim([-1, 1])

    summary_stats['mean/std'] = summary_stats['mean'] / summary_stats['std']
    utils.print_table(summary_stats)
    plt.show()

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
    f, axes = plt.subplots(v_spaces, 3, figsize=(28, v_spaces * 8))
    axes = (a for a in axes.flatten())

    for ax, (days_num, ic) in izip(axes, daily_ic.iteritems()):
        sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=ax)
        ax.set(title="%s day IC" % days_num, xlabel='IC')
        ax.set_xlim([-1, 1])
    plt.show()

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
        v_spaces = (num_sector) // 3

        f, axes = plt.subplots(v_spaces, 2, sharex=False, sharey=True, figsize=(20, 8*v_spaces))
        axes = axes.flatten()

        for i, (sc, cor) in enumerate(mean_ret_by_q.groupby(level='sector')):
            cor.xs(sc, level='sector').plot(kind='bar', title=sc, ax=axes[i])
            axes[i].set_xlabel('factor quantile')
            axes[i].set_ylabel('mean price % change')

        if i < len(axes):
            axes[-1] = None

        fig = plt.gcf()
        fig.suptitle("Mean Return By Factor Quantile By Sector", x=.5, y=.96)

    else:
        f, ax = plt.subplots(1, 1, figsize=(28, 12))
        mean_ret_by_q.plot(kind='bar',
                           title="Mean Return By Factor Quantile",
                           ax=ax)
        ax.set(xlabel='', ylabel='mean daily price % change')

    plt.show()



def plot_mean_quintile_returns_spread_time_series(mean_returns_spread, std=None,
        title='Top Quintile - Bottom Quantile Mean Return'):
    if isinstance(mean_returns_spread, pd.DataFrame):
        for name, fr_column in mean_returns_spread.iteritems():
            stdn = None if std is None else std[name]
            plot_mean_quintile_returns_spread_time_series(fr_column, std=stdn,
                title=str(name) + " Day Forward Return " + title)
        return

    f, ax = plt.subplots(figsize=(20, 8))
    (pd.DataFrame(mean_returns_spread.rename('mean_return_spread'))
        .assign(**{'1 month moving avg': mean_returns_spread.rolling(22).mean()})
        .plot(alpha=0.7, ax=ax))
    mean_returns_spread.rolling(22).mean()
    if std is not None:
        upper = mean_returns_spread.values + std
        lower = mean_returns_spread.values - std
        ax.fill_between(mean_returns_spread.index, lower, upper, alpha=0.3)

    ax.set(ylabel='Difference in Quantile Mean Return')
    ax.set(title=title)

    plt.show()


def plot_ic_by_sector(ic_sector):

    """
    Plots Spearman Rank Information Coefficient for a given factor over provided forward price
    movement windows. Separates by sector.

    Parameters
    ----------
    ic_sector : pd.DataFrame
        Sector-wise mean daily returns.
    """
    f, ax = plt.subplots(1, 1, figsize=(28, 12))
    ic_sector.plot(kind='bar', ax=ax)
    fig = plt.gcf()
    fig.suptitle("Information Coefficient by Sector", fontsize=16, x=.5, y=.93)
    plt.show()


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

    f, axes = plt.subplots(6, 2, sharex=False, sharey=True, figsize=(20, 45))
    axes = axes.flatten()
    i = 0
    for sc, data in ic_time.groupby(['sector']):
        data.drop('sector', axis=1).set_index('date').plot(kind='bar',
                                                           title=sc,
                                                           ax=axes[i])
        i += 1
    fig = plt.gcf()
    fig.suptitle("Monthly Information Coefficient by Sector", fontsize=16, x=.5, y=.93)
    plt.show()


def plot_factor_rank_auto_correlation(daily_factor, time_rule='W'):
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

    fa = perf.factor_rank_autocorrelation(daily_factor, time_rule=time_rule)
    print "Mean rank autocorrelation: " + str(fa.mean())
    f, ax = plt.subplots(1, 1, figsize=(28, 12))
    fa.plot(title='Factor Rank Autocorrelation', ax=ax)
    ax.set(ylabel='autocorrelation coefficient')
    plt.show()


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

    f, ax = plt.subplots(1, 1, figsize=(28, 12))
    turnover.plot(title='Top and Bottom Quantile Turnover', ax=ax)
    ax.set(ylabel='proportion of names not present in quantile in previous period', xlabel="")
    plt.show()

