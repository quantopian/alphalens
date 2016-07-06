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
from scipy import stats
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import performance as perf
import utils
from itertools import izip
from functools import wraps

DECIMAL_TO_BPS = 10000


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
    with pyfolio.plotting.context(font_scale=2):
        pyfolio.create_full_tear_sheet()

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


def summary_stats(ic_data, alpha_beta, quantized_factor, mean_ret_quantile,
                  autocorrelation_data, mean_ret_spread_quantile):
    """
    Generates a pretty printed table of summary statistics for the alpha factor.

    Parameters
    ----------
    ic_data : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward price movement windows.
    alpha_beta : pd.Series
        A list containing the alpha, beta, a t-stat(alpha) for the given factor and forward returns.
    quantized_factor : pd.Series
        Factor quantiles indexed by date and symbol.
    mean_ret_quantile : pd.DataFrame
        Mean daily returns by specified factor quantile.
    autocorrelation_data : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of factor values.
    mean_ret_spread_quantile : pd.Series
        Daily difference in quantile returns.
    """

    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)
    ic_summary_table["Ann. IR"] = (
        ic_data.mean() / ic_data.std()) * np.sqrt(252)

    max_quantile = quantized_factor.values.max()
    min_quantile = quantized_factor.values.min()
    turnover_table = pd.DataFrame(columns=["Top Quantile", "Bottom Quantile"])
    turnover_table.loc["Mean Turnover"] = [perf.quantile_turnover(quantized_factor, max_quantile).mean(),
                                           perf.quantile_turnover(quantized_factor, min_quantile).mean()]

    auto_corr = pd.Series()
    auto_corr["Mean Factor Rank Autocorrelation"] = autocorrelation_data.mean()

    returns_table = pd.DataFrame()
    returns_table = returns_table.append(alpha_beta)
    returns_table.loc["Mean Daily Return Top Quantile (bps)"] = mean_ret_quantile.loc[
        max_quantile] * DECIMAL_TO_BPS
    returns_table.loc["Mean Daily Return Bottom Quantile (bps)"] = mean_ret_quantile.loc[
        min_quantile] * DECIMAL_TO_BPS
    returns_table.loc[
        "Mean Daily Spread (bps)"] = mean_ret_spread_quantile.mean() * DECIMAL_TO_BPS

    print "Returns Analysis"
    utils.print_table(returns_table.round(3))
    print "Information Analysis"
    utils.print_table(ic_summary_table.round(3).T)
    print "Turnover Analysis"
    utils.print_table(turnover_table.round(3))
    print auto_corr.round(3)


def plot_daily_ic_ts(daily_ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient and IC moving average for a given factor.
    Sector neutralization of forward returns is recommended.

    Parameters
    ----------
    daily_ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    axes : matplotlib.Axes
        The axes that were plotted on.
    """

    num_plots = len(daily_ic.columns)
    if ax is None:
        f, axes = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        axes = (a for a in axes.flatten())

    else:
        axes = (a for a in ax)

    for a, (days_num, ic) in zip(axes, daily_ic.iteritems()):
        ic.plot(alpha=0.7, ax=a, lw=0.7, color='steelblue')
        ic.rolling(22).mean().plot(ax=a,
                                   color='forestgreen', lw=2, alpha=0.8)

        a.set(ylabel='IC', xlabel="")
        a.set_ylim([-0.25, 0.25])
        a.set_title("{} Day Information Coefficient (IC)".format(days_num))
        a.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        a.legend(['IC', '1 month moving avg'], loc='upper right')
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
                fontsize=16,
                bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                transform=a.transAxes,
                verticalalignment='top')

    return axes

def plot_daily_ic_hist(daily_ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.
    Sector neutralization of forward returns is recommended.

    Parameters
    ----------
    daily_ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    num_plots = len(daily_ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, axes = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        axes = (a for a in axes.flatten())
    else:
        axes = (a for a in ax)

    for a, (days_num, ic) in zip(axes, daily_ic.iteritems()):
        sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=a)
        a.set(title="%s Day IC" % days_num, xlabel='IC')
        a.set_xlim([-0.25, 0.25])
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
                fontsize=16,
                bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                transform=a.transAxes,
                verticalalignment='top')
        a.axvline(ic.mean(), color='w', linestyle='dashed', linewidth=2)

    return axes


def plot_quantile_returns_bar(mean_ret_by_q, by_sector=False, ax=None):
    """
    Plots mean daily returns for factor quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (sector) and mean daily return values.
    by_sector : boolean
        Disagregate figures by sector.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ymin = mean_ret_by_q.min().min() * DECIMAL_TO_BPS
    ymax = mean_ret_by_q.max().max() * DECIMAL_TO_BPS

    if by_sector:
        num_sector = len(
            mean_ret_by_q.index.get_level_values('sector').unique())

        if ax is None:
            v_spaces = ((num_sector - 1) // 2) + 1
            f, axes = plt.subplots(v_spaces, 2, sharex=False,
                                   sharey=True, figsize=(18, 6*v_spaces))
            axes = axes.flatten()
        else:
            axes = (a for a in ax)

        for a, (sc, cor) in zip(axes, mean_ret_by_q.groupby(level='sector')):
            (cor.xs(sc, level='sector')
                .multiply(DECIMAL_TO_BPS)
                .plot(kind='bar', title=sc, ax=a))

            a.set(xlabel='', ylabel='Mean Daily Return (bps)',
                  ylim=(ymin, ymax))

        if num_sector < len(list(axes)):
            axes[-1].set_visible(False)

        return axes

    else:
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        mean_ret_by_q.multiply(DECIMAL_TO_BPS).plot(kind='bar',
            title="Mean Return By Factor Quantile", ax=ax)
        ax.set(xlabel='', ylabel='Mean Daily Return (bps)',
               ylim=(ymin, ymax))

        return ax


def plot_mean_quantile_returns_spread_time_series(mean_returns_spread,
                                                  std=None,
                                                  bandwidth=0.5,
                                                  ax=None):
    """
    Plots mean daily returns for factor quantiles.

    Parameters
    ----------
    mean_returns_spread : pd.Series
        Series with difference between quantile mean returns by day.
    std : pd.Series
        Series with standard devation of difference between quantile
        mean returns each day.
    bandwidth : float
        Width of displayed error bands in standard deviations.
    title : string
        Plot title.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if isinstance(mean_returns_spread, pd.DataFrame):
        if ax is not None:
            axs = (a for a in ax)
        else:
            axs = (None for a in mean_returns_spread.columns)

        for a, (name, fr_column) in zip(axs, mean_returns_spread.iteritems()):
            stdn = None if std is None else std[name]
            ret_ax = plot_mean_quantile_returns_spread_time_series(fr_column,
                                                                   std=stdn, ax=a)

        return axs

    days = mean_returns_spread.name
    title = ('Top Minus Bottom Quantile Mean Return ({} Day Forward Return)'
             .format(days if days is not None else ""))

    if ax is None:
        f, ax = plt.subplots(figsize=(18, 6))

    mean_returns_spread *= DECIMAL_TO_BPS

    mean_returns_spread.plot(alpha=0.4, ax=ax, lw=0.7, color='forestgreen')
    pd.rolling_mean(mean_returns_spread, 22).plot(color='orangered', alpha=0.7)
    ax.legend(['mean returns spread', '1 month moving avg'], loc='upper right')

    if std is not None:
        std *= DECIMAL_TO_BPS
        upper = mean_returns_spread.values + (std * bandwidth)
        lower = mean_returns_spread.values - (std * bandwidth)
        ax.fill_between(
            mean_returns_spread.index, lower, upper, alpha=0.3, color='steelblue')

    ax.set(ylabel='Difference In Quantile Mean Return (bps)', xlabel='')
    ax.set(title=title, ylim=(-5., 5.))
    ax.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)

    return ax


def plot_ic_by_sector(ic_sector, ax=None):
    """
    Plots Spearman Rank Information Coefficient for a given factor over provided forward price
    movement windows. Separates by sector.

    Parameters
    ----------
    ic_sector : pd.DataFrame
        Sector-wise mean daily returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    ic_sector.plot(kind='bar', ax=ax)

    ax.set(title="Information Coefficient By Sector")

    return ax


def plot_factor_rank_auto_correlation(factor_autocorrelation, ax=None):
    """
    Plots factor rank autocorrelation over time. See factor_rank_autocorrelation for more details.

    Parameters
    ----------
    factor_autocorrelation : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of factor values.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_autocorrelation.plot(title='Factor Rank Autocorrelation', ax=ax)
    ax.set(ylabel='Autocorrelation Coefficient', xlabel='')
    ax.axhline(0.0, linestyle='-', color='black', lw=1)
    ax.text(.05, .95, "Mean %.3f" % factor_autocorrelation.mean(),
            fontsize=16,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')

    return ax


def plot_top_bottom_quantile_turnover(quantized_factor, ax=None):
    """
    Plots daily top and bottom quantile factor turnover.

    Parameters
    ----------
    quantized_factor : pd.Series
        Factor quantiles indexed by date and symbol.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    max_quantile = quantized_factor.values.max()
    turnover = pd.DataFrame()
    turnover['top quantile turnover'] = perf.quantile_turnover(
        quantized_factor, max_quantile)
    turnover['bottom quantile turnover'] = perf.quantile_turnover(
        quantized_factor, 1)

    turnover.plot(
        title='Top and Bottom Quantile Daily Turnover', ax=ax, alpha=0.6, lw=0.8)
    ax.set(ylabel='Proportion Of Names New To Quantile', xlabel="")

    return ax


def plot_monthly_ic_heatmap(mean_monthly_ic, ax=None):
    """
    Plots a heatmap of the information coefficient or returns by month.

    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N days forward.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    num_plots = len(mean_monthly_ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, axes = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        axes = (a for a in axes.flatten())
    else:
        axes = (a for a in ax)

    for a, (days_num, ic) in zip(axes, mean_monthly_ic.iteritems()):

        formatted_ic = (pd.Series(index=pd.MultiIndex.from_product([np.unique(ic.index.year),
                                                                    np.unique(ic.index.month)],
                                                                   names=["year", "month"])
                                  [:len(ic)], data=ic.values))

        sns.heatmap(
            formatted_ic.unstack(),
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 7},
            linewidths=0.01,
            linecolor='white',
            cmap=cm.RdYlGn,
            cbar=False,
            ax=a)
        a.set(ylabel='', xlabel='')

        a.set_title(
            "Monthly Mean {} Day Return IC".format(days_num))

    return axes


def plot_cumulative_returns(factor_returns, ax=None):
    """
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    factor_returns : pd.Series
        Daily returns of dollar neutral portfolio weighted by factor value.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_returns.add(1).cumprod().plot(
        ax=ax, lw=3, color='forestgreen', alpha=0.6)
    ax.set(ylabel='Cumulative Returns',
           title='Factor Weighted Long/Short Portfolio Cumulative Return', xlabel='')
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


def plot_cumulative_returns_by_quantile(quantile_daily_returns, ax=None):
    """
    Plots the cumulative returns of various factor quantiles.

    Parameters
    ----------
    mean_returns_by_quantile : pd.Series -- MultiIndex
        Mean daily returns by specified factor quantile.
        MultiIndex of date, quantile.
        See performance.mean_returns_by_quantile.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    daily_ret_wide = quantile_daily_returns.reset_index().pivot(
        index='date', columns='quantile', values=1)
    cum_ret = daily_ret_wide.add(1).cumprod()
    cum_ret = cum_ret.loc[:, ::-1]
    num_quant = len(cum_ret.columns)

    colors = cm.RdYlGn_r(np.linspace(0, 1, num_quant))

    cum_ret.plot(lw=2, ax=ax, color=colors)
    ax.legend()
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    ax.set(ylabel='Log Cumulative Returns',
           title='Cumulative Return by Quantile',
           xlabel='',
           yscale='symlog',
           yticks=np.linspace(ymin, ymax, 5),
           ylim=(ymin, ymax))

    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax