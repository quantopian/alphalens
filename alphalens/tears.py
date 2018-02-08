#
# Copyright 2017 Quantopian, Inc.
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


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from functools import wraps

from . import plotting
from . import performance as perf
from . import utils


class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None


def create_full_tear_sheet_api_change_warning(func):
    """
    Decorator used to help API transition: maintain the function backward
    compatible and warn the user about the API change.
    Old API:
        create_full_tear_sheet(factor_data,
                               long_short=True,
                               group_adjust=False,
                               by_group=False)
    New API:
        create_full_tear_sheet(factor_data,
                               long_short=True,
                               group_neutral=False,
                               by_group=False)

    Eventually this function can be deleted
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        group_adjust = kwargs.pop('group_adjust', None)
        if group_adjust is not None:
            kwargs['group_neutral'] = group_adjust
            warnings.warn("create_full_tear_sheet: 'group_adjust' argument "
                          "is now deprecated and replaced by 'group_neutral'",
                          category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return call_w_context


def create_information_tear_sheet_api_change_warning(func):
    """
    Decorator used to help API transition: maintain the function backward
    compatible and warn the user about the API change.
    Old API:
        create_information_tear_sheet(factor_data,
                                      group_adjust=False,
                                      by_group=False)
    New API:
        create_information_tear_sheet(factor_data,
                                      group_neutral=False,
                                      by_group=False)

    Eventually this function can be deleted
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        group_adjust = kwargs.pop('group_adjust', None)
        if group_adjust is not None:
            kwargs['group_neutral'] = group_adjust
            warnings.warn("create_information_tear_sheet: 'group_adjust' "
                          "argument is now deprecated and replaced by "
                          "'group_neutral'",
                          category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return call_w_context


def create_returns_tear_sheet_api_change_warning(func):
    """
    Decorator used to help API transition: maintain the function backward
    compatible and warn the user about the API change.
    Old API:
        create_returns_tear_sheet(factor_data,
                                  long_short=True,
                                  by_group=False)
    New API:
        create_returns_tear_sheet(factor_data,
                                  long_short=True,
                                  group_neutral=False,
                                  by_group=False)

    Eventually this function can be deleted
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        if len(args) == 3 and args[2] is True:
            warnings.warn("create_returns_tear_sheet: a new argument "
                          "'group_neutral' has been added. Please consider "
                          "using keyword arguments instead of positional ones "
                          " to avoid unexpected behaviour.",
                          category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return call_w_context


def create_event_returns_tear_sheet_api_change_warning(func):
    """
    Decorator used to help API transition: maintain the function backward
    compatible and warn the user about the API change.
    Old API:
        create_event_returns_tear_sheet(factor_data,
                                        prices,
                                        avgretplot=(5, 15),
                                        long_short=True,
                                        by_group=False)
    New API:
        create_event_returns_tear_sheet(factor_data,
                                        prices,
                                        avgretplot=(5, 15),
                                        long_short=True,
                                        group_neutral=False,
                                        std_bar=True,
                                        by_group=False)

    Eventually this function can be deleted
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        if len(args) == 5 and args[4] is True:
            warnings.warn("create_event_returns_tear_sheet: two new arguments"
                          " has been added: 'group_neutral' and 'std_bar'. "
                          "Please consider using keyword arguments instead "
                          "of positional ones to avoid unexpected behaviour.",
                          category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return call_w_context


@plotting.customize
def create_summary_tear_sheet(factor_data,
                              long_short=True,
                              group_neutral=False):
    """
    Creates a small summary tear sheet with returns, information, and turnover
    analysis.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    """

    # Returns Analysis
    mean_quant_ret, std_quantile = \
        perf.mean_return_by_quantile(factor_data,
                                     by_group=False,
                                     demeaned=long_short,
                                     group_adjust=group_neutral)

    mean_quant_rateret = \
        mean_quant_ret.apply(utils.rate_of_return, axis=0,
                             base_period=mean_quant_ret.columns[0])

    mean_quant_ret_bydate, std_quant_daily = \
        perf.mean_return_by_quantile(factor_data,
                                     by_date=True,
                                     by_group=False,
                                     demeaned=long_short,
                                     group_adjust=group_neutral)

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0]
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0,
        base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(factor_data,
                                        demeaned=long_short,
                                        group_adjust=group_neutral)

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data['factor_quantile'].max(),
        factor_data['factor_quantile'].min(),
        std_err=compstd_quant_daily)

    periods = utils.get_forward_returns_columns(factor_data.columns)

    fr_cols = len(periods)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_quantile_statistics_table(factor_data)

    plotting.plot_returns_table(alpha_beta,
                                mean_quant_rateret,
                                mean_ret_spread_quant)

    plotting.plot_quantile_returns_bar(mean_quant_rateret,
                                       by_group=False,
                                       ylim_percentiles=None,
                                       ax=gf.next_row())

    # Information Analysis
    ic = perf.factor_information_coefficient(factor_data)
    plotting.plot_information_table(ic)

    # Turnover Analysis
    quantile_factor = factor_data['factor_quantile']

    quantile_turnover = \
        {p: pd.concat([perf.quantile_turnover(quantile_factor, q, p)
                       for q in range(1, int(quantile_factor.max()) + 1)],
                      axis=1)
            for p in periods}

    autocorrelation = pd.concat(
        [perf.factor_rank_autocorrelation(factor_data, period) for period in
         periods], axis=1)

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    plt.show()
    gf.close()


@create_returns_tear_sheet_api_change_warning
@plotting.customize
def create_returns_tear_sheet(factor_data,
                              long_short=True,
                              group_neutral=False,
                              by_group=False):
    """
    Creates a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
        Additionally factor values will be demeaned across the factor universe
        when factor weighting the portfolio for cumulative returns plots
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots
    by_group : bool
        If True, display graphs separately for each group.
    """

    factor_returns = perf.factor_returns(factor_data,
                                         long_short,
                                         group_neutral)

    mean_quant_ret, std_quantile = \
        perf.mean_return_by_quantile(factor_data,
                                     by_group=False,
                                     demeaned=long_short,
                                     group_adjust=group_neutral)

    mean_quant_rateret = \
        mean_quant_ret.apply(utils.rate_of_return, axis=0,
                             base_period=mean_quant_ret.columns[0])

    mean_quant_ret_bydate, std_quant_daily = \
        perf.mean_return_by_quantile(factor_data,
                                     by_date=True,
                                     by_group=False,
                                     demeaned=long_short,
                                     group_adjust=group_neutral)

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return, axis=0,
        base_period=mean_quant_ret_bydate.columns[0]
    )

    compstd_quant_daily = \
        std_quant_daily.apply(utils.std_conversion, axis=0,
                              base_period=std_quant_daily.columns[0])

    alpha_beta = perf.factor_alpha_beta(factor_data,
                                        factor_returns,
                                        long_short,
                                        group_neutral)

    mean_ret_spread_quant, std_spread_quant = \
        perf.compute_mean_returns_spread(mean_quant_rateret_bydate,
                                         factor_data['factor_quantile'].max(),
                                         factor_data['factor_quantile'].min(),
                                         std_err=compstd_quant_daily)

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_returns_table(alpha_beta,
                                mean_quant_rateret,
                                mean_ret_spread_quant)

    plotting.plot_quantile_returns_bar(mean_quant_rateret,
                                       by_group=False,
                                       ylim_percentiles=None,
                                       ax=gf.next_row())

    plotting.plot_quantile_returns_violin(mean_quant_rateret_bydate,
                                          ylim_percentiles=(1, 99),
                                          ax=gf.next_row())

    for p in factor_returns:

        title = ('Factor Weighted '
                 + ('Group Neutral ' if group_neutral else '')
                 + ('Long/Short ' if long_short else '')
                 + "Portfolio Cumulative Return ({} Period)".format(p))

        plotting.plot_cumulative_returns(factor_returns[p],
                                         period=p,
                                         title=title,
                                         ax=gf.next_row())

        plotting.plot_cumulative_returns_by_quantile(mean_quant_ret_bydate[p],
                                                     period=p,
                                                     ax=gf.next_row())

    ax_mean_quantile_returns_spread_ts = [gf.next_row()
                                          for x in range(fr_cols)]
    plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts
    )

    plt.show()
    gf.close()

    if by_group:
        mean_return_quantile_group, mean_return_quantile_group_std_err = \
            perf.mean_return_by_quantile(factor_data,
                                         by_date=False,
                                         by_group=True,
                                         demeaned=long_short,
                                         group_adjust=group_neutral)

        mean_quant_rateret_group = mean_return_quantile_group.apply(
            utils.rate_of_return, axis=0,
            base_period=mean_return_quantile_group.columns[0]
        )

        num_groups = len(mean_quant_rateret_group.index
                         .get_level_values('group').unique())

        vertical_sections = 1 + (((num_groups - 1) // 2) + 1)
        gf = GridFigure(rows=vertical_sections, cols=2)

        ax_quantile_returns_bar_by_group = [gf.next_cell()
                                            for _ in range(num_groups)]
        plotting.plot_quantile_returns_bar(mean_quant_rateret_group,
                                           by_group=True,
                                           ylim_percentiles=(5, 95),
                                           ax=ax_quantile_returns_bar_by_group)
        plt.show()
        gf.close()


@create_information_tear_sheet_api_change_warning
@plotting.customize
def create_information_tear_sheet(factor_data,
                                  group_neutral=False,
                                  by_group=False):
    """
    Creates a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_neutral : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, display graphs separately for each group.
    """

    ic = perf.factor_information_coefficient(factor_data, group_neutral)

    plotting.plot_information_table(ic)

    columns_wide = 2
    fr_cols = len(ic.columns)
    rows_when_wide = (((fr_cols - 1) // columns_wide) + 1)
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    if not by_group:

        mean_monthly_ic = \
            perf.mean_information_coefficient(factor_data,
                                              group_adjust=group_neutral,
                                              by_group=False,
                                              by_time="M")
        ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
        plotting.plot_monthly_ic_heatmap(mean_monthly_ic,
                                         ax=ax_monthly_ic_heatmap)

    if by_group:
        mean_group_ic = \
            perf.mean_information_coefficient(factor_data,
                                              group_adjust=group_neutral,
                                              by_group=True)

        plotting.plot_ic_by_group(mean_group_ic, ax=gf.next_row())

    plt.show()
    gf.close()


@plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None):
    """
    Creates a tear sheet for analyzing the turnover properties of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    """

    if turnover_periods is None:
        turnover_periods = utils.get_forward_returns_columns(
            factor_data.columns)

    quantile_factor = factor_data['factor_quantile']

    quantile_turnover = \
        {p: pd.concat([perf.quantile_turnover(quantile_factor, q, p)
                       for q in range(1, int(quantile_factor.max()) + 1)],
                      axis=1)
            for p in turnover_periods}

    autocorrelation = pd.concat(
        [perf.factor_rank_autocorrelation(factor_data, period) for period in
         turnover_periods], axis=1)

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = (((fr_cols - 1) // 1) + 1)
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        plotting.plot_top_bottom_quantile_turnover(quantile_turnover[period],
                                                   period=period,
                                                   ax=gf.next_row())

    for period in autocorrelation:
        plotting.plot_factor_rank_auto_correlation(autocorrelation[period],
                                                   period=period,
                                                   ax=gf.next_row())

    plt.show()
    gf.close()


@create_full_tear_sheet_api_change_warning
@plotting.customize
def create_full_tear_sheet(factor_data,
                           long_short=True,
                           group_neutral=False,
                           by_group=False):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
    group_neutral : bool
        Should this computation happen on a group neutral portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
        - See tears.create_information_tear_sheet for details on how this
        flag affects information analysis
    by_group : bool
        If True, display graphs separately for each group.
    """

    plotting.plot_quantile_statistics_table(factor_data)
    create_returns_tear_sheet(factor_data,
                              long_short,
                              group_neutral,
                              by_group,
                              set_context=False)
    create_information_tear_sheet(factor_data,
                                  group_neutral,
                                  by_group,
                                  set_context=False)
    create_turnover_tear_sheet(factor_data, set_context=False)


@create_event_returns_tear_sheet_api_change_warning
@plotting.customize
def create_event_returns_tear_sheet(factor_data,
                                    prices,
                                    avgretplot=(5, 15),
                                    long_short=True,
                                    group_neutral=False,
                                    std_bar=True,
                                    by_group=False):
    """
    Creates a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, the factor
        quantile/bin that factor value belongs to and (optionally) the group
        the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    prices : pd.DataFrame
        A DataFrame indexed by date with assets in the columns containing the
        pricing data.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after)
        If not None, plot quantile average cumulative returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so then
        factor returns will be demeaned across the factor universe
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile
    by_group : bool
        If True, display graphs separately for each group.
    """

    before, after = avgretplot

    avg_cumulative_returns = \
        perf.average_cumulative_return_by_quantile(
            factor_data,
            prices,
            periods_before=before,
            periods_after=after,
            demeaned=long_short,
            group_adjust=group_neutral)

    num_quantiles = int(factor_data['factor_quantile'].max())

    vertical_sections = 1
    if std_bar:
        vertical_sections += (((num_quantiles - 1) // 2) + 1)
    cols = 2 if num_quantiles != 1 else 1
    gf = GridFigure(rows=vertical_sections, cols=cols)
    plotting.plot_quantile_average_cumulative_return(avg_cumulative_returns,
                                                     by_quantile=False,
                                                     std_bar=False,
                                                     ax=gf.next_row())
    if std_bar:
        ax_avg_cumulative_returns_by_q = [gf.next_cell()
                                          for _ in range(num_quantiles)]
        plotting.plot_quantile_average_cumulative_return(
            avg_cumulative_returns,
            by_quantile=True,
            std_bar=True,
            ax=ax_avg_cumulative_returns_by_q)

    plt.show()
    gf.close()

    if by_group:
        groups = factor_data['group'].unique()
        num_groups = len(groups)
        vertical_sections = ((num_groups - 1) // 2) + 1
        gf = GridFigure(rows=vertical_sections, cols=2)

        avg_cumret_by_group = \
            perf.average_cumulative_return_by_quantile(
                factor_data,
                prices,
                periods_before=before,
                periods_after=after,
                demeaned=long_short,
                group_adjust=group_neutral,
                by_group=True)

        for group, avg_cumret in avg_cumret_by_group.groupby(level='group'):
            avg_cumret.index = avg_cumret.index.droplevel('group')
            plotting.plot_quantile_average_cumulative_return(
                avg_cumret,
                by_quantile=False,
                std_bar=False,
                title=group,
                ax=gf.next_cell())

        plt.show()
        gf.close()


@plotting.customize
def create_event_study_tear_sheet(factor_data,
                                  prices=None,
                                  avgretplot=(5, 15)):
    """
    Creates an event study tear sheet for analysis of a specific event.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
    prices : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing the
        pricing data.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).
    """

    long_short = False

    plotting.plot_quantile_statistics_table(factor_data)

    gf = GridFigure(rows=1, cols=1)
    plotting.plot_events_distribution(events=factor_data['factor'],
                                      ax=gf.next_row())
    plt.show()
    gf.close()

    if prices is not None and avgretplot is not None:

        create_event_returns_tear_sheet(factor_data=factor_data,
                                        prices=prices,
                                        avgretplot=avgretplot,
                                        long_short=long_short,
                                        group_neutral=False,
                                        std_bar=True,
                                        by_group=False)

    factor_returns = perf.factor_returns(factor_data,
                                         demeaned=False,
                                         equal_weight=True)

    mean_quant_ret, std_quantile = \
        perf.mean_return_by_quantile(factor_data,
                                     by_group=False,
                                     demeaned=long_short)
    mean_quant_ret = \
        mean_quant_ret.apply(utils.rate_of_return, axis=0,
                             base_period=mean_quant_ret.columns[0])

    mean_quant_ret_bydate, std_quant_daily = \
        perf.mean_return_by_quantile(factor_data,
                                     by_date=True,
                                     by_group=False,
                                     demeaned=long_short)

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return, axis=0,
        base_period=mean_quant_ret_bydate.columns[0]
    )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 1
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_quantile_returns_bar(mean_quant_ret,
                                       by_group=False,
                                       ylim_percentiles=None,
                                       ax=gf.next_row())

    plotting.plot_quantile_returns_violin(mean_quant_rateret_bydate,
                                          ylim_percentiles=(1, 99),
                                          ax=gf.next_row())

    for p in factor_returns:

        plotting.plot_cumulative_returns(factor_returns[p],
                                         period=p,
                                         ax=gf.next_row())

    plt.show()
    gf.close()
