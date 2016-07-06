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
    sector_plots : boolean
        If True create sector specific plots.
    days: list
        Days to compute forward returns on.
    quantiles: int:
        The number of buckets to parition the data into for analysis.
    filter_zscore : int
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.
    sector_names: dict
        A dictionary keyed by sector code with values corresponding
        to the display name for each sector.
        - Example:
            {101: "Basic Materials", 102: "Consumer Cyclical"}
    """
    if 1 not in days:
        days.insert(0, 1)

    if sector_mappings == 'morningstar':
        sector_mappings = utils.MORNINGSTAR_SECTOR_MAPPING


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
        sector_adjust=can_sector_adjust,
        by_sector=False)

    mean_monthly_ic = perf.mean_information_coefficient(
        factor, forward_returns, by_time="M")

    factor_returns = perf.factor_returns(factor, forward_returns)

    alpha_beta = perf.factor_alpha_beta(factor, forward_returns,
                                        factor_daily_returns=factor_returns)

    quantile_factor = perf.quantize_factor(
        factor, by_sector=False, quantiles=quantiles)

    mean_ret_quantile, std_quantile = perf.mean_return_by_quantile(
        quantile_factor, forward_returns, by_sector=False, std=True)

    mean_ret_quant_daily, std_quant_daily = perf.mean_return_by_quantile(
        quantile_factor, forward_returns, by_time='D',
        by_sector=False, std=True)

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_ret_quant_daily, quantiles, 1, std=std_quant_daily)

    factor_autocorrelation = perf.factor_rank_autocorrelation(
        factor, time_rule='D')

    ## PLOTTING ##

    summary_stats(daily_ic, alpha_beta, quantile_factor, mean_ret_quantile,
                  factor_autocorrelation, mean_ret_spread_quant)

    plot_quantile_returns_bar(mean_ret_quantile, by_sector=False)

    plot_cumulative_returns(factor_returns[1])
    plot_cumulative_returns_by_quantile(mean_ret_quant_daily[1])

    plot_daily_ic_ts(daily_ic)
    plot_daily_ic_hist(daily_ic)

    plot_monthly_IC_heatmap(mean_monthly_ic)

    plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std=std_spread_quant,
        bandwidth=0.5,
        title="Top Quantile - Bottom Quantile Mean Return (0.5 std. error band)")

    plot_top_bottom_quantile_turnover(quantile_factor)
    plot_factor_rank_auto_correlation(factor_autocorrelation)

    # Sector Specific Breakdown
    if can_sector_adjust and sector_plots:
        ic_by_sector = perf.mean_information_coefficient(
            factor, forward_returns, by_sector=True)

        mean_return_quantile_sector = perf.mean_return_by_quantile(
            quantile_factor, forward_returns, by_sector=True)

        plot_ic_by_sector(ic_by_sector)

        plot_quantile_returns_bar(mean_return_quantile_sector, by_sector=True, sector_mapping=sector_mappings)
