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
from itertools import product


@plotting.plotting_context
def create_factor_tear_sheet(factor,
                             prices,
                             groupby=None,
                             show_groupby_plots=True,
                             periods=(1, 5, 10),
                             quantiles=5,
                             filter_zscore=10,
                             groupby_labels=None,
                             long_short=True,
                             avgretplot=(5, 15) ):
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
    quantiles : int
        The number of buckets to parition the data into for analysis.
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
    """

    periods = list(periods)
    if 1 not in periods:
        periods.insert(0, 1)
    periods.sort()

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
                                           quantiles=quantiles)

    mean_ret_quantile, std_quantile = perf.mean_return_by_quantile(quantile_factor,
                                                                   forward_returns,
                                                                   by_group=False,
                                                                   demeaned=long_short)

    mean_ret_quant_daily, std_quant_daily = perf.mean_return_by_quantile(quantile_factor,
                                                                         forward_returns,
                                                                         by_time='D',
                                                                         by_group=False,
                                                                         demeaned=long_short)

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(mean_ret_quant_daily,
                                                                               quantiles,
                                                                               1,
                                                                               std_err=std_quant_daily)

    factor_autocorrelation = perf.factor_rank_autocorrelation(factor,
                                                              time_rule='D')

    ## PLOTTING ##
    plotting.summary_stats(ic,
                           alpha_beta,
                           quantile_factor,
                           mean_ret_quantile,
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
    vertical_sections = 4 + fr_cols
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_quantile_returns_bar(mean_ret_quantile,
                                       by_group=False,
                                       ylim_percentiles=None,
                                       ax=gf.next_row())

    plotting.plot_quantile_returns_violin(mean_ret_quant_daily,
                                          ylim_percentiles=(1, 99),
                                          ax=gf.next_row())

    plotting.plot_cumulative_returns(factor_returns[1],
                                     ax=gf.next_row())

    plotting.plot_cumulative_returns_by_quantile(mean_ret_quant_daily[1],
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
        
        vertical_sections = 1 + (((quantiles - 1) // 2) + 1)
        gf = GridFigure(rows=vertical_sections, cols=2)
        plotting.plot_quantile_average_cumulative_return(quantile_factor, forward_returns[1],
                                                         by_quantile=False,
                                                         periods_before=before, periods_after=after,
                                                         std_bar=False, demeaned=long_short,
                                                         ax=gf.next_row())

        ax_avg_cumulative_returns_by_q = [ gf.next_cell() for x in range(quantiles) ]
        plotting.plot_quantile_average_cumulative_return(quantile_factor, forward_returns[1],
                                                         by_quantile=True,
                                                         periods_before=before, periods_after=after,
                                                         std_bar=True, demeaned=long_short,
                                                         ax=ax_avg_cumulative_returns_by_q)
                                                             
    # IC
    columns_wide = 2
    rows_when_wide = (((fr_cols - 1) // columns_wide) + 1)
    vertical_sections = fr_cols + 3 * rows_when_wide + 2    
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [ gf.next_row() for x in range(fr_cols) ]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [ gf.next_cell() for x in range(fr_cols * 2) ]
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    ax_monthly_ic_heatmap = [ gf.next_cell() for x in range(fr_cols) ]
    plotting.plot_monthly_ic_heatmap(mean_monthly_ic, ax=ax_monthly_ic_heatmap)

    plotting.plot_top_bottom_quantile_turnover(quantile_factor, ax=gf.next_row())

    plotting.plot_factor_rank_auto_correlation(factor_autocorrelation, ax=gf.next_row())

    # Group Specific Breakdown
    if can_group_adjust and show_groupby_plots:
        ic_by_group = perf.mean_information_coefficient(factor,
                                                        forward_returns,
                                                        by_group=True)

        mean_return_quantile_group, mean_return_quantile_group_std_err = perf.mean_return_by_quantile(quantile_factor,
                                                                                                      forward_returns,
                                                                                                      by_group=True,
                                                                                                      demeaned=True)

        num_groups = len(ic_by_group.index.get_level_values('group').unique())
        rows_when_2_wide = (((num_groups - 1) // 2) + 1)
        vertical_sections = 1 + rows_when_2_wide
        gf = GridFigure(rows=vertical_sections, cols=2)
        
        plotting.plot_ic_by_group(ic_by_group, ax=gf.next_row())

        ax_quantile_returns_bar_by_group = [ gf.next_cell() for x in range(num_groups) ]
        plotting.plot_quantile_returns_bar(mean_return_quantile_group,
                                           by_group=True,
                                           ylim_percentiles=(5, 95),
                                           ax=ax_quantile_returns_bar_by_group)
