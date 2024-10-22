from giza.datasets import DatasetsLoader
import polars as pl

loader = DatasetsLoader()

prices = loader.load('tokens-daily-prices-mcap-volume') # FROM Feb 1st 2019 TO Feb 5th 2024
liquidations = loader.load('aave-liquidationsV3')  # FROM Feb 2nd 2023 TO Feb 5th 2024

p_tokens = prices.select('token').unique()

# drop useless features
print(liquidations.shape) # (625, 13)
liquidations = liquidations.drop(["liquidator", "user", "col_contract_address", "col_current_value_USD", "debt_contract_address", "debt_current_value_USD"])
print(liquidations.shape) # (625, 7)


print(liquidations.head())
print(liquidations.tail())


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

# keep `prices` of tokens that exist in the remaining `liquidations`
# >> for 9 tokens: 17 716 rows of prices data
prices.filter(pl.col('token').is_in(liquid_tokens['token']))


# convert `day` column from type `datetime[ns]` to `date`
liquidations = liquidations.with_columns([
    pl.col("day").dt.date()
])

# sort data by date
liquidations = liquidations.sort('day')


# what could be interesting:
#  - check the different pairs/combinations of collat_tokens and debt_tokens, and count the number of each pair
#  - count the number of identical collat_token, and same for the debt_token maybe ?