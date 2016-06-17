import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import performance as perf
import utils
from itertools import izip


# DONE
def plot_daily_ic_ts(daily_ic, return_ax=False):
    """
    Plots Spearman Rank Information Coefficient and IC moving average for a given factor.
    Sector neturalization of forward price movements with sector_adjust_forward_price_moves is
    recommended.

    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns.
    factor_name : string
        Name of factor column on which to compute IC.
    """
    num_plots = len(daily_ic.columns)

    f, axes = plt.subplots(num_plots, 1, figsize=(28, num_plots * 12))
    axes = (a for a in axes.flatten())

    summary_stats = pd.DataFrame(columns=['mean', 'std'])
    for ax, (days_num, ic) in izip(axes, daily_ic.iteritems()):
        title = "{} day IC {}".format(days_num)
        summary_stats.loc["%i day IC" % days_num] = [ic.mean(), ic.std()]

        ic_df = (pd.DataFrame(ic.rename("{} day IC".format(days_num)))
                 .assign(**{'1 month moving avg': ic.rolling(22).mean()})
                 .plot(title=title, alpha=0.7, ax=ax))
        ax.set(ylabel='IC', xlabel='date')

    summary_stats['mean/std'] = summary_stats['mean'] / summary_stats['std']
    utils.print_table(summary_stats)
    plt.show()

    if return_ax:
        return axes

# DONE
def plot_daily_ic_hist(daily_ic, return_ax=False):
    num_plots = len(daily_ic.columns)

    v_spaces = num_plots // 3
    f, axes = plt.subplots(v_spaces, 3, figsize=(28, v_spaces * 8))
    axes = (a for a in axes.flatten())

    for ax, (days_num, ic) in izip(axes, daily_ic.iteritems()):
        sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=ax)
        ax.set(title="%s day IC" % days_num, xlabel='IC')
    plt.show()

    if return_ax:
        return axes

#DONE
#DONE FOR SECTORS
def plot_quantile_returns_bar(mean_ret_by_q, by_sector=False):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns.
    by_sector : boolean
        Disagregate figures by sector.
    """

    if by_sector:
        f, axes = plt.subplots(6, 2, sharex=False, sharey=True, figsize=(20, 45))
        axes = axes.flatten()

        for i, (sc, cor) in enumerate(mean_ret_by_q.groupby(level='sector')):
            cor.plot(kind='bar', title=sc, ax=axes[i])
            axes[i].set_xlabel('factor quantile')
            axes[i].set_ylabel('mean price % change')

        fig = plt.gcf()
        fig.suptitle("Mean Return By Factor Quantile", fontsize=24, x=.5, y=.93)

    else:
        f, ax = plt.subplots(1, 1, figsize=(28, 12))
        mean_ret_by_q.plot(kind='bar',
                           title="Mean Return By Factor Quantile",
                           ax=ax)
        ax.set(xlabel='factor quantile', ylabel='mean daily price % change')

    plt.show()


def plot_ic_by_sector(ic_sector, factor_name='factor'):
    """
    Plots Spearman Rank Information Coefficient for a given factor over provided forward price
    movement windows. Separates by sector.

    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns.
    factor_name : string
        Name of factor column on which to compute IC.
    """
    ic_sector.plot(kind='bar') #yerr=err_sector
    fig = plt.gcf()
    fig.suptitle("Information Coefficient by Sector", fontsize=16, x=.5, y=.93)
    plt.show()


def plot_ic_by_sector_over_time(ic_time):
    """
    Plots sector-wise time window mean daily Spearman Rank Information Coefficient
    for a given factor over provided forward price movement windows.

    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns.
    time_rule : string, optional
        Time span to use in time grouping reduction.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html for available options.
    factor_name : string
        Name of factor column on which to compute IC.
    """

    ic_time = ic_time.reset_index()
    err_time = err_time.reset_index()

    f, axes = plt.subplots(6, 2, sharex=False, sharey=True, figsize=(20, 45))
    axes = axes.flatten()
    i = 0
    for sc, data in ic_time.groupby(['sector']):
        e = err_time[err_time.sector == sc].set_index('date')
        data.drop('sector', axis=1).set_index('date').plot(kind='bar',
                                                                title=sc,
                                                                ax=axes[i],
                                                                ) # yerr=e
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
    factor_name : string
        Name of factor column on which to compute IC.
    """

    fa = perf.factor_rank_autocorrelation(daily_factor, time_rule=time_rule)
    print "Mean rank autocorrelation: " + str(fa.mean())
    fa.plot(title='Week-to-Week Factor Rank Autocorrelation')
    plt.ylabel('autocorrelation coefficient')
    plt.show()

# DONE
def plot_top_bottom_quantile_turnover(quantized_factor):
    """
    Plots daily top and bottom quantile factor turnover.

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame with date, equity, and factor value columns.
    quantiles : integer
        Number of quantiles to use in quantile bucketing.
    """
    max_quantile = quantized_factor.values.max()
    turnover = pd.DataFrame()
    turnover['top quintile turnover'] = perf.quantile_turnover(quantized_factor, max_quantile)
    turnover['bottom quintile turnover'] = perf.quantile_turnover(quantized_factor, 1)

    turnover.plot(title='Top and Bottom Quintile Turnover')
    plt.ylabel('proportion of names not present in quantile in previous period')
    plt.show()


