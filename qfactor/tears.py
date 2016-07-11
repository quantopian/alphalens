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

from plotting import *
import performance as perf
import utils
import matplotlib.gridspec as gridspec
from itertools import product


@plotting_context
def create_factor_tear_sheet(factor,
                             prices,
                             sectors=None,
                             sector_plots=True,
                             days=(1, 5, 10),
                             quantiles=5,
                             filter_zscore=10,
                             sector_names=None
                             ):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factors.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by date and asset, containing
        the values for a single alpha factor.
    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with equities
        in the columns. It is important to pass the
        correct pricing data in depending on what time of day your
        signal was generated so to avoid lookahead bias, or
        delayed calculations. Pricing data must span the factor
        analysis time period plus an additional buffer window
        that is greater than the maximum number of expected days
        in the forward returns calculations.
    sectors : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the daily sector codes for each asset, or
        a dict of asset to sector mappings. If a dict is passed,
        it is assumed that sector mappings are unchanged for the
        entire time period of the passed factor data.
    sector_plots : bool
        If True create sector specific plots.
    days : list
        Days to compute forward returns on.
    quantiles : int
        The number of buckets to parition the data into for analysis.
    filter_zscore : int
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.
    sector_names : dict
        A dictionary keyed by sector code with values corresponding
        to the display name for each sector.
    """
    if 1 not in days:
        days.insert(0, 1)

    if sector_names == 'morningstar':
        sector_names = utils.MORNINGSTAR_SECTOR_MAPPING


    can_sector_adjust = sectors is not None
    factor, forward_returns = utils.format_input_data(
        factor,
        prices,
        sectors=sectors,
        days=days,
        filter_zscore=filter_zscore,
        sector_names=sector_names)

    daily_ic = perf.factor_information_coefficient(
        factor, forward_returns,
        sector_adjust=False,
        by_sector=False)

    mean_monthly_ic = perf.mean_information_coefficient(
        factor, forward_returns, by_time="M")

    factor_returns = perf.factor_returns(factor, forward_returns)

    alpha_beta = perf.factor_alpha_beta(factor, forward_returns,
                                        factor_daily_returns=factor_returns)

    quantile_factor = perf.quantize_factor(
        factor, by_sector=False, quantiles=quantiles)

    mean_ret_quantile, std_quantile = perf.mean_return_by_quantile(
        quantile_factor, forward_returns, by_sector=False, std_err=True)

    mean_ret_quant_daily, std_quant_daily = perf.mean_return_by_quantile(
        quantile_factor, forward_returns, by_time='D',
        by_sector=False, std_err=True)

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_ret_quant_daily, quantiles, 1, std_err=std_quant_daily)

    factor_autocorrelation = perf.factor_rank_autocorrelation(
        factor, time_rule='D')

    ## PLOTTING ##
    summary_stats(daily_ic, alpha_beta, quantile_factor, mean_ret_quantile,
                  factor_autocorrelation, mean_ret_spread_quant)

    fr_cols = len(days)

    # Returns
    vertical_sections = 4 + fr_cols
    fig = plt.figure(figsize=(14, vertical_sections * 7))
    ret_gs = gridspec.GridSpec(vertical_sections, 2, wspace=0.4, hspace=0.3)

    i = 0
    ax_quantile_returns_bar = plt.subplot(ret_gs[i, :])
    i += 1
    plot_quantile_returns_bar(mean_ret_quantile, by_sector=False,
        ylim_percentiles=None,
        ax=ax_quantile_returns_bar)

    ax_quantile_returns_violin = plt.subplot(ret_gs[i, :])
    i += 1
    plot_quantile_returns_violin(mean_ret_quant_daily,
        ylim_percentiles=(1,99),
        ax=ax_quantile_returns_violin)

    ax_cumulative_returns = plt.subplot(ret_gs[i, :])
    i += 1
    plot_cumulative_returns(factor_returns[1], ax=ax_cumulative_returns)

    ax_cumulative_returns_by_quantile = plt.subplot(ret_gs[i, :])
    i += 1
    plot_cumulative_returns_by_quantile(mean_ret_quant_daily[1],
        ax=ax_cumulative_returns_by_quantile)

    ax_mean_quantile_returns_spread_ts = []
    for j in range(fr_cols):
        p = plt.subplot(ret_gs[i, :])
        ax_mean_quantile_returns_spread_ts.append(p)
        i += 1

    plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts)

    # IC
    columns_wide = 2

    rows_when_wide = (((fr_cols - 1) // columns_wide) + 1)
    ix_wide = list(product(range(rows_when_wide), range(columns_wide)))
    vertical_sections = fr_cols + 3 * rows_when_wide + 2
    fig = plt.figure(figsize=(14, vertical_sections * 7))
    ic_gs = gridspec.GridSpec(vertical_sections, 2, wspace=0.4, hspace=0.3)

    i = 0
    ax_daily_ic_ts = []
    for j in range(fr_cols):
        p = plt.subplot(ic_gs[i, :])
        ax_daily_ic_ts.append(p)
        i += 1
    plot_daily_ic_ts(daily_ic, ax=ax_daily_ic_ts)

    ax_daily_ic_hist = []
    ax_daily_ic_qq = []
    for j in range(fr_cols):
        p_hist = plt.subplot(ic_gs[j+i, 0])
        p_qq = plt.subplot(ic_gs[j+i, 1])
        ax_daily_ic_hist.append(p_hist)
        ax_daily_ic_qq.append(p_qq)

    i += fr_cols
    plot_daily_ic_hist(daily_ic, ax=ax_daily_ic_hist)
    plot_daily_ic_qq(daily_ic, ax=ax_daily_ic_qq)

    ax_monthly_ic_heatmap = []
    for j, k in ix_wide:
        p = plt.subplot(ic_gs[j+i, k])
        ax_monthly_ic_heatmap.append(p)
    i += rows_when_wide
    plot_monthly_ic_heatmap(mean_monthly_ic,
        ax=ax_monthly_ic_heatmap)

    ax_top_bottom_quantile_turnover = plt.subplot(ic_gs[i, :])
    plot_top_bottom_quantile_turnover(quantile_factor,
        ax=ax_top_bottom_quantile_turnover)
    i += 1

    ax_factor_rank_auto_correlation = plt.subplot(ic_gs[i, :])
    plot_factor_rank_auto_correlation(factor_autocorrelation,
        ax=ax_factor_rank_auto_correlation)

    # Sector Specific Breakdown
    if can_sector_adjust and sector_plots:
        ic_by_sector = perf.mean_information_coefficient(
            factor, forward_returns, by_sector=True)

        mean_return_quantile_sector = perf.mean_return_by_quantile(
            quantile_factor, forward_returns, by_sector=True)

        num_sectors = len(
            ic_by_sector.index.get_level_values('sector').unique())
        rows_when_2_wide = (((num_sectors - 1) // 2) + 1)
        ix_2_wide = product(range(rows_when_2_wide), range(2))
        vertical_sections = 1 + rows_when_2_wide
        fig = plt.figure(figsize=(14, vertical_sections * 7))

        s_gs = gridspec.GridSpec(vertical_sections, 2, wspace=0.4, hspace=0.3)
        i = 0

        ax_ic_by_sector = plt.subplot(s_gs[i, :])
        i += 1
        plot_ic_by_sector(ic_by_sector, ax=ax_ic_by_sector)

        ax_quantile_returns_bar_by_sector = []
        for j, k in ix_2_wide:
            p = plt.subplot(s_gs[j+i, k])
            ax_quantile_returns_bar_by_sector.append(p)
        i += rows_when_wide
        plot_quantile_returns_bar(mean_return_quantile_sector,
            by_sector=True, ylim_percentiles=(5, 95),
            ax=ax_quantile_returns_bar_by_sector)