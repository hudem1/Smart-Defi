from giza.datasets import DatasetsLoader
import polars as pl
from datetime import datetime, timedelta

from common_data_prepocessing import common_preprocess

loader = DatasetsLoader()

def add_lagged_prices(liquidations: pl.DataFrame, prices: pl.DataFrame, token_column: str):
    # Create a temporary column in prices that matches the token_column in liquidations
    token_price = pl.col(f"price_{token_column}").alias("price")
    # Join prices to liquidations on date criteria
    result = (liquidations
              .join(
                  prices.with_columns(token_price),
                  left_on="day",
                  right_on=pl.col("date"),
                  how="left"
              )
              .sort(["day", "date"])
              .with_columns([
                  # Create lagged features for 30 days before the liquidation date
                  [pl.col("price").shift(i).alias(f"{token_column}_t_{i}") for i in range(1, 31)]
              ])
              .filter(pl.col("date") == pl.col("day"))  # Ensure we're aligning to the liquidation day
              .drop("date", "price")  # Clean up unnecessary columns
            )
    return result


# This function fetches the last 30 days' prices for a given token and day
def get_historical_prices(prices: pl.DataFrame, current_day: pl.Date, token_col: str, token_debt: str):
    # current_day_date = pl.lit(current_day).str.strptime(pl.Date)
    start_day = current_day - pl.duration(days=30)
    price_data = (prices
                  .filter((pl.col("date") < current_day) & (pl.col("date") >= start_day))
                  .select(["date", f"price_{token_col}", f"price_{token_debt}"])
                  .sort("date", descending=True)  # Ensure we are getting the most recent 30 days
                 )

    return price_data


def preprocess_data():
    prices, liquidations = common_preprocess()

    # # Process date
    # liquidations = liquidations.with_column(
    #     pl.col("day").str.strptime(pl.Date).cast(pl.Int32).alias("day_ordinal")
    # )

    # # Encode categorical data
    # liquidations = liquidations.with_columns([
    #     pl.col("token_col").cast(pl.Categorical).cast(pl.Int32).alias("token_col"),
    #     pl.col("token_debt").cast(pl.Categorical).cast(pl.Int32).alias("token_debt")
    # ])

    # Initialize a list to collect the enhanced data frames
    results = []
    # Iterate over each row in the liquidations DataFrame
    for row in liquidations.rows():
        # Fetch prices for token_col and token_debt
        # row = pl.DataFrame([row], schema=liquidations.schema)
        day, token_col, token_debt = row[0], row[1], row[2]
        historical_prices = get_historical_prices(prices, day, token_col, token_debt)
        print(row)
        print(historical_prices)
        row = pl.DataFrame([row], schema=liquidations.schema)

        # Create new columns for each price day
        for i in range(30):
            price_token_col, price_token_debt = historical_prices.select("price_" + token_col, "price_" + token_debt)[i]
            row = row.with_columns(pl.lit(price_token_col).alias(f"token_col_t_{i+1}"))
            row = row.with_columns(pl.lit(price_token_debt).alias(f"token_debt_t_{i+1}"))

        # Append the row with new data to results
        results.append(row)

    # Convert results back to DataFrame
    lagged_liquidations: pl.DataFrame = pl.concat(results)

    lagged_liquidations = lagged_liquidations.with_columns([
        lagged_liquidations["day"].cast(pl.Int32).alias("day"),
        # Columns treated independently:
        # lagged_liquidations["token_col"].cast(pl.Categorical).cast(pl.UInt32).alias("token_col"),
        # lagged_liquidations["token_debt"].cast(pl.Categorical).cast(pl.UInt32).alias("token_debt")
    ])

    # Find unique tokens in both columns
    unique_token_symbols = pl.concat([
        pl.col('token_col'),
        pl.col('token_debt'),
    ]).alias('token').unique()
    unique_tokens = lagged_liquidations.select(unique_token_symbols).to_series()

    # Create a dictionary with consistent encoding
    token_mapping = {token: idx for idx, token in enumerate(unique_tokens)}

    # Display the mapping
    print("Token Mapping:", token_mapping)

    lagged_liquidations = lagged_liquidations.with_columns([
        # lagged_liquidations["token_col"].replace(lambda x: token_mapping[x]),
        # lagged_liquidations["token_debt"].replace(token_mapping),
        lagged_liquidations["token_col"].map_elements(lambda x: token_mapping[x]).alias("token_col"),
        lagged_liquidations["token_debt"].map_elements(lambda x: token_mapping[x]).alias("token_debt")
    ])