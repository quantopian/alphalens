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

from . import plotting
from . import performance as perf
from . import utils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd


@plotting.plotting_context
def create_factor_tear_sheet(factor,
                             prices,
                             groupby=None,
                             show_groupby_plots=True,
                             periods=(1, 5, 10),
                             quantiles=5,
                             bins=None,
                             filter_zscore=10,
                             groupby_labels=None,
                             long_short=True,
                             avgretplot=(5, 15),
                             turnover_for_all_periods=False):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets
        in the columns. It is important to pass the
        correct pricing data in depending on what time your
        signal was generated so to avoid lookahead bias, or
        delayed calculations. Pricing data must span the factor
        analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    show_groupby_plots : bool
        If True create group specific plots.
    periods : sequence[int]
        periods to compute forward returns on.
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, e.g. [0, .10, .5, .90, 1.]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        Only one of 'quantiles' or 'bins' can be not-None
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    long_short : bool
        Should this computation happen on a long short portfolio?
    avgretplot: tuple (int, int) - (before, after)
        If not None, plot quantile average cumulative returns
    turnover_for_all_periods: boolean, optional
        If True, diplay quantile turnover and factor autocorrelation
        plots for every periods. If False, only period of 1 is
        plotted
    """

    periods = sorted(periods)
    turnover_periods = list(periods) if turnover_for_all_periods else [1]

    can_group_adjust = groupby is not None
    factor, forward_returns = utils.get_clean_factor_and_forward_returns(factor,
                                                                         prices,
                                                                         groupby=groupby,
                                                                         periods=periods,
                                                                         filter_zscore=filter_zscore,
                                                                         groupby_labels=groupby_labels)

    ic = perf.factor_information_coefficient(factor,
                                             forward_returns,
                                             group_adjust=False,
                                             by_group=False)

    mean_monthly_ic = perf.mean_information_coefficient(factor,
                                                        forward_returns,
                                                        by_time="M")

    factor_returns = perf.factor_returns(factor, forward_returns, long_short)

    alpha_beta = perf.factor_alpha_beta(factor,
                                        forward_returns,
                                        factor_returns=factor_returns)

    quantile_factor = perf.quantize_factor(factor,
                                           by_group=False,
                                           quantiles=quantiles,
                                           bins=bins)

    num_quant = quantile_factor.max()

    def compound_returns(period_ret):
        period = int(period_ret.name)
        return period_ret.add(1).pow(1./period).sub(1)

    mean_ret_quantile, std_quantile = perf.mean_return_by_quantile(quantile_factor,
                                                                   forward_returns,
                                                                   by_group=False,
                                                                   demeaned=long_short)
                                                                   
    mean_compret_quantile = mean_ret_quantile.apply(compound_returns, axis=0)

    mean_ret_quant_daily, std_quant_daily = perf.mean_return_by_quantile(quantile_factor,
                                                                         forward_returns,
                                                                         by_date=True,
                                                                         by_group=False,
                                                                         demeaned=long_short)

    mean_compret_quant_daily = mean_ret_quant_daily.apply(compound_returns, axis=0)
    compstd_quant_daily = std_quant_daily.apply(compound_returns, axis=0)

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(mean_compret_quant_daily,
                                                                               num_quant,
                                                                               1,
                                                                               std_err=compstd_quant_daily)

    quantile_turnover = {p: pd.concat([perf.quantile_turnover(
        quantile_factor, q, p) for q in range(1, num_quant + 1)], axis=1) for p in turnover_periods}

    factor_autocorrelation = pd.concat(
        [perf.factor_rank_autocorrelation(factor, period=p) for p in turnover_periods], axis=1)

    ## PLOTTING ##
    plotting.summary_stats(factor,
                           ic,
                           alpha_beta,
                           quantile_factor,
                           mean_compret_quantile,
                           quantile_turnover,
                           factor_autocorrelation,
                           mean_ret_spread_quant)

    # it makes life easier with grid plots
    class GridFigure(object):
    
        def __init__(self, rows, cols):
            self.rows = rows
            self.cols = cols
            self.fig  = plt.figure(figsize=(14, rows * 7))
            self.gs   = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
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
            
    # Returns
    fr_cols = len(periods)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_quantile_returns_bar(mean_compret_quantile,
                                       by_group=False,
                                       ylim_percentiles=None,
                                       ax=gf.next_row())

    plotting.plot_quantile_returns_violin(mean_compret_quant_daily,
                                          ylim_percentiles=(1, 99),
                                          ax=gf.next_row())
    
    for p in periods:
    
        plotting.plot_cumulative_returns(factor_returns[p], period=p,
                                         ax=gf.next_row())
                                         
        plotting.plot_cumulative_returns_by_quantile(mean_ret_quant_daily[p], period=p,
                                                     ax=gf.next_row())

    ax_mean_quantile_returns_spread_ts = [ gf.next_row() for x in range(fr_cols) ]
    plotting.plot_mean_quantile_returns_spread_time_series(mean_ret_spread_quant,
                                                           std_err=std_spread_quant,
                                                           bandwidth=0.5,
                                                           ax=ax_mean_quantile_returns_spread_ts)

    # Average Cumulative Returns
    if avgretplot is not None:
    
        before, after = avgretplot
        after = max(after, max(periods) + 1)
        avgretplot = before, after
        
        avg_cumulative_returns = perf.average_cumulative_return_by_quantile(quantile_factor,
                                                                            prices,
                                                                            periods_before=before,
                                                                            periods_after=after,
                                                                            demeaned=long_short)

        vertical_sections = 1 + (((num_quant - 1) // 2) + 1)
        gf = GridFigure(rows=vertical_sections, cols=2)
        plotting.plot_quantile_average_cumulative_return(avg_cumulative_returns, by_quantile=False,
                                                         std_bar=False, ax=gf.next_row())

        ax_avg_cumulative_returns_by_q = [ gf.next_cell() for x in range(num_quant) ]
        plotting.plot_quantile_average_cumulative_return(avg_cumulative_returns, by_quantile=True,
                                                         std_bar=True, ax=ax_avg_cumulative_returns_by_q)

    # IC
    columns_wide = 2
    rows_when_wide = (((fr_cols - 1) // columns_wide) + 1)
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * len(periods)
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [ gf.next_row() for x in range(fr_cols) ]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [ gf.next_cell() for x in range(fr_cols * 2) ]
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    ax_monthly_ic_heatmap = [ gf.next_cell() for x in range(fr_cols) ]
    plotting.plot_monthly_ic_heatmap(mean_monthly_ic, ax=ax_monthly_ic_heatmap)

    for p in turnover_periods:

        plotting.plot_top_bottom_quantile_turnover(quantile_turnover[p], period=p, ax=gf.next_row())

        plotting.plot_factor_rank_auto_correlation(
            factor_autocorrelation[p], period=p, ax=gf.next_row())

    # Group Specific Breakdown
    if can_group_adjust and show_groupby_plots:
        ic_by_group = perf.mean_information_coefficient(factor,
                                                        forward_returns,
                                                        by_group=True)

        mean_return_quantile_group, mean_return_quantile_group_std_err = perf.mean_return_by_quantile(quantile_factor,
                                                                                                      forward_returns,
                                                                                                      by_group=True,
                                                                                                      demeaned=True)
        mean_compret_quantile_group = mean_return_quantile_group.apply(compound_returns, axis=0)
        
        num_groups = len(ic_by_group.index.get_level_values('group').unique())
        group_rows = (((num_groups - 1) // 2) + 1) if avgretplot is None else num_groups
        gf = GridFigure(rows=1+group_rows, cols=2)

        plotting.plot_ic_by_group(ic_by_group, ax=gf.next_row())

        gaxes = [ gf.next_cell() for _ in range(group_rows*2) ]

        ax_quantile_returns_bar_by_group = gaxes if avgretplot is None else gaxes[0::2] 
        plotting.plot_quantile_returns_bar(mean_compret_quantile_group,
                                           by_group=True,
                                           ylim_percentiles=(5, 95),
                                           ax=ax_quantile_returns_bar_by_group)

        if avgretplot is not None:
            before, after = avgretplot
            
            for ax, (group, g_factor) in zip(gaxes[1::2], quantile_factor.groupby(level='group')):
            
                avg_cumulative_returns = perf.average_cumulative_return_by_quantile(g_factor,
                                                                                    prices,
                                                                                    periods_before=before,
                                                                                    periods_after=after,
                                                                                    demeaned=True)

                plotting.plot_quantile_average_cumulative_return(avg_cumulative_returns, by_quantile=False,
                                                                 std_bar=False, title=group, ax=ax)

