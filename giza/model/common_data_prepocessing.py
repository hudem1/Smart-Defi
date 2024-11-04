from giza.datasets import DatasetsLoader
import polars as pl
from datetime import datetime, timedelta


loader = DatasetsLoader()

def common_preprocess():
    prices = loader.load('tokens-daily-prices-mcap-volume') # FROM Feb 25th 2015 TO Feb 5th 2024
    liquidations = loader.load('aave-liquidationsV3')  # FROM Feb 2nd 2023 TO Feb 5th 2024

    p_tokens = prices.select('token').unique()

    # drop useless features
    print(liquidations.shape) # (625, 13)
    liquidations = liquidations.drop(["liquidator", "user", "col_contract_address", "col_current_value_USD", "debt_contract_address", "debt_current_value_USD"])
    print(liquidations.shape) # (625, 7)

    # keep only liquidations whose both collat & debt tokens exist in `prices`
    liquidations = liquidations.filter(
        pl.col('token_col').is_in(p_tokens['token']),
        pl.col('token_debt').is_in(p_tokens['token']),
    )

    # get the unique token symbols (both colateral and debt combined)
    unique_token_symbols = pl.concat([
        pl.col('token_col'),
        pl.col('token_debt'),
    ]).alias('token').unique()
    # shape: 9 tokens in common b/w `prices` and `liquidations`
    liquid_tokens = liquidations.select(unique_token_symbols)

    # convert `day` column from type `datetime[ns]` to `date`
    liquidations = liquidations.with_columns([
        pl.col("day").dt.date()
    ])

    # sort data by date
    liquidations = liquidations.sort('day')

    # Count the number of identical collat_token, and same for the debt_token
    # liquidations.select('token_col').unique()
    # liquidations.select('token_debt').unique()

    # Get the number of collat & debt tokens combinations and sort from highest to lowest
    combinations_count = liquidations.group_by(['token_col', 'token_debt']).count().sort('count', descending=True)

    # keep the 5 most frequent token combinations
    top_combinations = combinations_count.head().drop('count')

    # Keep only the liquidations for the most frequent token combinations, shape: (342, 7)
    liquidations = liquidations.join(top_combinations, on=['token_col', 'token_debt'], how="inner")

    # We end up with 5 unique tokens for the liquidations data
    liquid_tokens = top_combinations.select(unique_token_symbols)

    # Get the number of collat & debt tokens combinations and sort from highest to lowest
    combinations_count = liquidations.group_by(['token_col', 'token_debt']).count().sort('count', descending=True)

    # keep `prices` of tokens that exist in the remaining `liquidations`
    # >> for 5 tokens: 10 739 rows of prices data
    prices = prices.filter(pl.col('token').is_in(liquid_tokens['token']))

    prices = prices.sort('date')

    # For each token, display the first and last occurrences' date
    first_and_last = prices.group_by('token').agg([
        pl.first('date').alias('first_date'),
        pl.last('date').alias('last_date')
    ]).sort('first_date')

    common_start_date = first_and_last.select(
        pl.last('first_date').alias('common_date')
    ).item()

    # prices.group_by(['token', 'date']).all()

    # prices.groupby(["token", "date"]).count().filter(pl.col('count') > 1)
    # datetime(2019, 11, 19)

    # just to have the price for each token under the token name (but without keeping the `market_cap` and `volumes_last_24h`)
    # prices.pivot(
    #     values='price',
    #     index='date',
    #     columns='token'
    # )

    # Pivot the dataframe according to the `date` column to have it under the wanted format
    prices = prices.pivot(
        values=['price', 'market_cap', 'volumes_last_24h'],
        index='date',
        columns='token'
    )

    # rename suffix given to columns to lighten the column names
    prices = prices.rename({
        col: col.replace("_token_", "_") for col in prices.columns
    })

    prices = prices.filter(pl.col('date') >= common_start_date)

    # all.filter(pl.col('date') == datetime(2024, 2, 5)).select(['price_token_DAI', 'market_cap_token_DAI', 'volumes_last_24h_token_DAI'])

    return prices, liquidations