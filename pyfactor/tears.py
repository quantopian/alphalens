def create_factor_tear_sheet(factor_and_fp,
                             sector_adjust=False,
                             sector_names=None,
                             days=[1, 5, 10],
                             nquantiles = 10,
                             ret_type='normal' # normal, market_excess or beta_excess
                            ):

    adj_factor_and_fp = sector_adjust_forward_price_moves(factor_and_fp)

    # What is the sector-netural rolling mean IC for our different forward price windows?
    plot_daily_ic(adj_factor_and_fp, factor_name=factor_name)

    # Plot comulative returns over time for each quantile
    plot_quantile_cumulative_return(factor_and_fp, daily_perc_ret, quantiles=nquantiles, by_quantile=False,
                    factor_name=factor_name, days_before=days_before, days_after=days_after, std_bar=False)

    # Plot comulative returns over time, one plot for each quantile
    plot_quantile_cumulative_return(factor_and_fp, daily_perc_ret, quantiles=nquantiles, by_quantile=True,
                    factor_name=factor_name, days_before=days_before, days_after=days_after, std_bar=True)

    # What are the sector-neutral factor decile mean returns for our different forward price windows?
    plot_quantile_returns(adj_factor_and_fp, by_sector=False, quantiles=nquantiles, factor_name=factor_name)

    # As above but more detailed, we want to know the volatility of returns
    plot_quantile_returns_box(adj_factor_and_fp, by_sector=False, quantiles=nquantiles, factor_name=factor_name)

    # let's have a look at the relationship between factor and returns
    plot_factor_vs_fwdprice_distribution(adj_factor_and_fp, factor_name=factor_name)
    plot_factor_vs_fwdprice_distribution(adj_factor_and_fp, factor_name=factor_name, remove_outliers=True)

    # How much is the contents of the the top and bottom quintile changing each day?
    plot_top_bottom_quantile_turnover(factor, num_quantiles=5, factor_name=factor_name)

    # What is the autocorrelation in factor rank? Should this be autocorrelation in sector-neutralized
    # factor value?
    plot_factor_rank_auto_correlation(factor, factor_name=factor_name)

    # What is IC decay for each sector?
    plot_ic_by_sector(factor_and_fp, factor_name=factor_name)

    if pd.to_datetime(end_date) - pd.to_datetime(start_date) > pd.Timedelta(days=70):
        tr = 'M'
    else:
        tr = 'W'
    # What is the IC decay for each sector over time, not assuming sector neturality?
    plot_ic_by_sector_over_time(adj_factor_and_fp, time_rule=tr, factor_name=factor_name)

    # What are the factor quintile returns for each sector, not assuming sector neutrality?
    plot_quantile_returns(adj_factor_and_fp, by_sector=True, quantiles=5, factor_name=factor_name)

    # As above but more detailed, we want to know the volatility of returns
    plot_quantile_returns_box(adj_factor_and_fp, by_sector=True, quantiles=5, factor_name=factor_name)