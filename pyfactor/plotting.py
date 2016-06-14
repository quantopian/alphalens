import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import performance as perf
import utils
from itertools import izip


#### Begin Refactored ####

def plot_daily_ic_ts(daily_ic, return_ax=False, is_sector_adjusted=False):
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
        title = "{} day IC {}".format(days_num,
                   "(sector adjusted)" if is_sector_adjusted else "")
        summary_stats.loc["%i day IC" % days_num] = [ic.mean(), ic.std()]

        ic_df = (pd.DataFrame(ic.rename("{} day IC".format(days_num)))
                 .assign(**{'1 month moving avg': ic.rolling(22).mean()})
                 .plot(title=title, alpha=0.7, ax=ax))
        # ax.text(.01, .95, 'mean: {0:.4f}'.format(mean_ic),
        #     ha='left', va='center', transform=ax.transAxes)
        # ax.text(.01, .91, 'stdev: {0:.4f}'.format(std_ic),
        #     ha='left', va='center', transform=ax.transAxes)
        # ax.text(.01, .87, 'mean/stdev: {0:.4f}'.format(mean_ic/std_ic),
        #     ha='left', va='center', transform=ax.transAxes)
        ax.set(ylabel='IC', xlabel='date')

    summary_stats['mean/std'] = summary_stats['mean'] / summary_stats['std']
    utils.print_table(summary_stats)
    plt.show()

    if return_ax:
        return axes

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

def plot_quantile_returns_bar(mean_ret_by_q, by_sector=False, quantiles=5):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns.
    by_sector : boolean
        Disagregate figures by sector.
    quantiles : integer
        Number of quantiles buckets to use in factor bucketing.
    factor_name : string
        Name of factor column on which to compute IC.
    """
    if by_sector:
        f, axes = plt.subplots(6,2, sharex=False, sharey=True, figsize=(20,45))
        axes = axes.flatten()

        for i, (sc, cor) in enumerate(mean_ret_by_q.groupby(level='sector_code')):
            cor = cor.reset_index().drop('sector_code', axis=1).set_index('factor_quantile')
            cor.plot(kind='bar', title=sc, ax=axes[i])
            axes[i].set_xlabel('factor quantile')
            axes[i].set_ylabel('mean price % change')

        fig = plt.gcf()
        fig.suptitle(factor_name + ": Mean Return By Factor Quantile", fontsize=24, x=.5, y=.93)

    else:
        f, ax = plt.subplots(1,1, figsize=(28,12))
        mean_ret_by_q.plot(kind='bar',
                           title="Mean Return By Factor Quantile",
                           ax=ax)
        ax.set(xlabel='factor quantile', ylabel='mean daily price % change')

    plt.show()


def plot_quantile_cumulative_return(cum_ret_by_q):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns.
    daily_perc_ret : pd.DataFrame
        Pricing data to use in cumulative return calculation. Equities as columns, dates as index.
    quantiles : integer
        Number of quantiles buckets to use in factor bucketing.
    by_quantile : boolean
        Disagregate figures by quantile.
    factor_name : string
        Name of factor column on which to compute IC.
    days_before : int
        How many days to plot before the factor is calculated
    days_after  : int
        How many days to plot after the factor is calculated
    day_zero_align : boolean
         Aling returns at day 0 (timeseries is 0 at day 0)
    std_bar : boolean
        Plot standard deviation plot
    """

    palette = sns.color_palette("coolwarm", len(cum_ret_by_q.columns))

    f, ax = plt.subplots(1,1, figsize=(28,12))
    for i, (quantile, cum_returns) in enumerate(cum_ret_by_q.iteritems()):
        label = 'Quantile ' + str(quantile)
        sns.tsplot(ax=ax, data=cum_returns.values, condition=label,
                        legend=True, color=palette[i], time=cum_returns.index)

    # mark day zero with a vertical line
    ax.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Days')
    plt.ylabel('% return')
    plt.title("Cumulative returns by quantile")

    plt.show()

#### End Refactored #####

def plot_quantile_returns_box(factor_and_fp, by_sector=True, quantiles=5, factor_name='factor'):
    """
    Plots sector-wise mean daily returns as boxplot for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns.
    by_sector : boolean
        Disagregate figures by sector.
    quantiles : integer
        Number of quantiles buckets to use in factor bucketing.
    factor_name : string
        Name of factor column on which to compute IC.
    """
    decile_factor = quantile_bucket_factor(factor_and_fp, by_sector=by_sector, quantiles=quantiles,
                                           factor_name=factor_name)
    fwd_days, pc_cols = utils.get_price_move_cols(decile_factor)

    if by_sector:
        f, axes = plt.subplots(6,2, sharex=False, sharey=True, figsize=(20,45))
        axes = axes.flatten()
        i = 0
        for sc, cor in decile_factor.groupby(by='sector_code'):
            cor_box_plot = pd.melt(cor, var_name='fwd_days_price', value_name='%_price_change',
                                             id_vars=['factor_quantile'], value_vars=pc_cols)
            # boxplot doesn't sort 'x' by itself
            cor_box_plot = cor_box_plot.sort(columns='factor_quantile', ascending=True)
            sns.boxplot(ax=axes[i], x="factor_quantile", y="%_price_change", hue="fwd_days_price", data=cor_box_plot)
            axes[i].set_xlabel('factor quantile')
            axes[i].set_ylabel('mean price % change')
            axes[i].set_title(sc)
            i+=1
        fig = plt.gcf()
        fig.suptitle(factor_name + ": Mean Return By Factor Quantile", fontsize=24, x=.5, y=.93)

    else:

        decile_factor_box_plot = pd.melt(decile_factor, var_name='fwd_days_price', value_name='%_price_change',
                                 id_vars=['factor_quantile'], value_vars=pc_cols)
        # boxplot doesn't sort 'x' by itself
        decile_factor_box_plot = decile_factor_box_plot.sort(columns='factor_quantile', ascending=True)
        sns.boxplot(x="factor_quantile", y="%_price_change", hue="fwd_days_price", data=decile_factor_box_plot)

        plt.title("Mean Return By Factor Quantile (sector adjusted)")
        plt.xlabel('factor quantile')
        plt.ylabel('mean price % change')

    plt.show()


def plot_ic_by_sector(factor_and_fp, factor_name='factor'):
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
    ic_sector, err_sector = factor_spearman_rank_IC(factor_and_fp, factor_name=factor_name)

    ic_sector.plot(kind='bar') #yerr=err_sector
    fig = plt.gcf()
    fig.suptitle("Information Coefficient by Sector", fontsize=16, x=.5, y=.93)
    plt.show()


def plot_ic_by_sector_over_time(factor_and_fp, time_rule=None, factor_name='factor'):
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
    ic_time, err_time = factor_spearman_rank_IC(factor_and_fp, time_rule=time_rule,
                                                factor_name=factor_name)
    ic_time = ic_time.reset_index()
    err_time = err_time.reset_index()

    f, axes = plt.subplots(6,2, sharex=False, sharey=True, figsize=(20,45))
    axes = axes.flatten()
    i = 0
    for sc, data in ic_time.groupby(['sector_code']):
        e = err_time[err_time.sector_code == sc].set_index('date')
        data.drop('sector_code', axis=1).set_index('date').plot(kind='bar',
                                                                title=sc,
                                                                ax=axes[i],
                                                                ) # yerr=e
        i+=1
    fig = plt.gcf()
    fig.suptitle("Monthly Information Coefficient by Sector", fontsize=16, x=.5, y=.93)
    plt.show()

def plot_factor_rank_auto_correlation(daily_factor, time_rule='W', factor_name='factor'):
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

    fa = factor_rank_autocorrelation(daily_factor, time_rule=time_rule, factor_name=factor_name)
    print "Mean rank autocorrelation: " + str(fa.mean())
    fa.plot(title='Week-to-Week Factor Rank Autocorrelation')
    plt.ylabel('autocorrelation coefficient')
    plt.show()

def plot_top_bottom_quantile_turnover(daily_factor, num_quantiles=5, factor_name='factor'):
    """
    Plots daily top and bottom quantile factor turnover. See quantile_bucket_factor for more
    details.

    Parameters
    ----------
    daily_factor : pd.DataFrame
        DataFrame with date, equity, and factor value columns.
    num_quantiles : integer
        Number of quantiles to use in quantile bucketing.
    factor_name : string
        Name of factor column on which to compute IC.
    """

    quint_buckets = quantile_bucket_factor(daily_factor, by_sector=True,
                                           quantiles=5, factor_name=factor_name)
    turnover = pd.DataFrame()
    turnover['top quintile turnover'] = quantile_turnover(quint_buckets, num_quantiles)
    turnover['bottom quintile turnover'] = quantile_turnover(quint_buckets, 1)

    turnover.plot(title='Top and Bottom Quintile Turnover (Quantiles Computed by Sector)')
    plt.ylabel('proportion of names not present in quantile in previous period')
    plt.show()


def plot_factor_vs_fwdprice_distribution(factor_and_fp, factor_name='factor', remove_outliers = False):
    """
    Plots distribuion of factor vs forward price.
    This is useful to visually spot linear or non linear relationship between factor and fwd prices

    Parameters
    ----------
    factor_and_fp : pd.DataFrame
        DataFrame with date, equity, factor, and forward price movement columns.
    factor_name : string
        Name of factor column
    remove_outliers : boolean
        Remove outliers before plotting the distribution
    """
    fwd_days, pc_cols = get_price_move_cols(factor_and_fp)
    for col in pc_cols:

        if remove_outliers:
            data = factor_and_fp[ ~is_outlier(factor_and_fp[factor_name].values) & \
                                  ~is_outlier(factor_and_fp[col].values)]
        else:
            data = factor_and_fp

        jg = sns.jointplot(data[factor_name], data[col], kind="kde")
        jg.fig.suptitle('Factor/returns kernel density estimation' +
                        (' NO OUTLIERS' if remove_outliers else '') )
        plt.show()

        jg = sns.jointplot(data[factor_name], data[col], kind="reg")
        jg.fig.suptitle('Factor/returns regression' +
                        (' NO OUTLIERS' if remove_outliers else '') )

        plt.show()




