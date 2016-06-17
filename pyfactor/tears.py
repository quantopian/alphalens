from plotting import *
import performance as perf
import utils
import pandas as pd


def create_factor_tear_sheet(factor,
                             prices,
                             factor_name='factor',
                             sector_plots=True,
                             sector_adjust=False,
                             sector_names=None,
                             days=(1, 5, 10),
                             nquantiles = 10,
                             ret_type='normal' # normal, market_excess or beta_excess
                            ):
    start_date = factor.index.levels[0].min()
    end_date = factor.index.levels[0].max()

    if sector_adjust:
        pricing = utils.sector_adjust_forward_returns(prices)

    forward_prices = utils.compute_forward_returns(prices, days=days)

    daily_ic, _ = perf.factor_information_coefficient(factor, forward_prices, by_sector=False)

    quantile_factor = perf.quantize_factor(factor, by_sector=sector_adjust,
                                           quantiles=nquantiles)

    mean_ret_by_q = perf.mean_daily_return_by_factor_quantile(quantile_factor, forward_prices,
                                                         by_sector=sector_adjust)

    cum_ret_by_q = perf.surrounding_cumulative_returns_by_quantile(quantile_factor, prices,
        days_before=5, days_after=10)

    # What is the sector-netural rolling mean IC for our different forward price windows?
    plot_daily_ic_ts(daily_ic, is_sector_adjusted=sector_adjust)
    plot_daily_ic_hist(daily_ic)

    # What are the sector-neutral factor quantile mean returns for our different forward price windows?
    plot_quantile_returns_bar(mean_ret_by_q, by_sector=False)

    # Plot comulative returns over time for each quantile
    plot_quantile_cumulative_return(cum_ret_by_q)



    # # Plot comulative returns over time, one plot for each quantile
    # plot_quantile_cumulative_return(factor_and_fp, daily_perc_ret, quantiles=nquantiles, by_quantile=True,
    #                 factor_name=factor_name, days_before=days_before, days_after=days_after, std_bar=True)



    # As above but more detailed, we want to know the volatility of returns
    # plot_quantile_returns_box(factor_and_fp, by_sector=False, quantiles=nquantiles, factor_name=factor_name)

    # let's have a look at the relationship between factor and returns
    # plot_factor_vs_fwdprice_distribution(factor_and_fp, factor_name=factor_name)
    # plot_factor_vs_fwdprice_distribution(factor_and_fp, factor_name=factor_name, remove_outliers=True)

    # # How much is the contents of the the top and bottom quintile changing each day?
    # plot_top_bottom_quantile_turnover(factor, num_quantiles=5, factor_name=factor_name)

    # # What is the autocorrelation in factor rank? Should this be autocorrelation in sector-neutralized
    # # factor value?
    # plot_factor_rank_auto_correlation(factor, factor_name=factor_name)

    # # What is IC decay for each sector?
    # plot_ic_by_sector(factor_and_fp, factor_name=factor_name)

    # if end_date - start_date > pd.Timedelta(days=70):
    #     tr = 'M'
    # else:
    #     tr = 'W'
    # # What is the IC decay for each sector over time, not assuming sector neturality?
    # plot_ic_by_sector_over_time(adj_factor_and_fp, time_rule=tr, factor_name=factor_name)


    # if sector_plots:
    #     # What are the factor quintile returns for each sector, not assuming sector neutrality?
    #     plot_quantile_returns(adj_factor_and_fp, by_sector=True, quantiles=5, factor_name=factor_name)

    #     # As above but more detailed, we want to know the volatility of returns
    #     plot_quantile_returns_box(adj_factor_and_fp, by_sector=True, quantiles=5, factor_name=factor_name)