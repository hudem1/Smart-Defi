from giza.datasets import DatasetsLoader
import polars as pl
from datetime import datetime, timedelta

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from common_data_prepocessing import common_preprocess

from giza.zkcook import mcr, serialize_model

loader = DatasetsLoader()


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

    results = []

    for row in liquidations.rows():
        # Fetch prices for token_col and token_debt
        day, token_col, token_debt, collateral_amount, _, debt_amount = row[0], row[1], row[2], row[3], row[4], row[5]
        historical_prices = get_historical_prices(prices, day, token_col, token_debt)

        row = pl.DataFrame([row], schema=liquidations.schema)

        row = row.with_columns(
            (pl.col("debt_value_USD") / pl.col("col_value_USD")).alias("debt_to_collat_ratio")
        )

        row = row.drop(['debt_value_USD', 'col_value_USD', 'debt_amount', 'collateral_amount'])

        # Create new columns for each price day
        for i in range(30):
            price_token_col, price_token_debt = historical_prices.select("price_" + token_col, "price_" + token_debt)[i]

            debt_to_collat_ratio = (price_token_debt * debt_amount) / (price_token_col * collateral_amount)
            row = row.with_columns(pl.lit(debt_to_collat_ratio).alias(f"debt_to_collat_ratio_t_{i+1}"))

        results.append(row)

    lagged_liquidations: pl.DataFrame = pl.concat(results)

    lagged_liquidations = lagged_liquidations.with_columns([
        lagged_liquidations["day"].cast(pl.Int32).alias("day"),
        # Columns treated independently:
        # lagged_liquidations["token_col"].cast(pl.Categorical).cast(pl.UInt32).alias("token_col"),
        # lagged_liquidations["token_debt"].cast(pl.Categorical).cast(pl.UInt32).alias("token_debt")
    ])

    lagged_liquidations = categorical_encoding(lagged_liquidations)

    # test = liquidations.with_columns(
    #     (pl.col("debt_value_USD") / pl.col("col_value_USD")).alias("debt_to_collateral_ratio")
    # )
    # ratios_per_pair = test.group_by(['token_col', 'token_debt']).agg(
    #     pl.col('debt_to_collateral_ratio')
    # )

    # ratios_per_pair.with_columns(
    #     pl.col("debt_to_collateral_ratio").arr.eval(pl.element().mean(), dtype_out=pl.Float64).alias("ratio_mean")
    #     # pl.col('debt_to_collateral_ratio').ar.mean().alias('ratio mean')
    # )

    # Drop the columns col_value_USD and debt_value_USD
    # lagged_liquidations = lagged_liquidations.drop(["col_value_USD", "debt_value_USD"])
    # lagged_liquidations = lagged_liquidations.with_columns([
    #     (pl.col("token_col_t") * pl.col("collateral_amount")).alias("col_value_USD"),
    #     (pl.col("token_debt_t") * pl.col("debt_amount")).alias("debt_value_USD"),
    # ])


    # Add a column to have the first most recent day where the ratio is superior to the liquidation ratio
    # test = lagged_liquidations.with_columns(
    #     pl.struct([f"debt_to_collat_ratio_t_{i}" for i in range(1, 31)] + ['debt_to_collat_ratio']).alias("ratios").map_elements(find_start_of_borrow).alias("start_of_borrow")
    # )

    ratio_columns = [f"debt_to_collat_ratio_t_{i}" for i in range(1, 31)]  # Adjust according to your actual range

    # Maybe a better approach would be to set to None all later values AS SOON AS we encounter
    # lagged_liquidations = lagged_liquidations.with_columns([
    #     pl.when(pl.col(col) > pl.col("debt_to_collat_ratio"))
    #     .then(None)
    #     .otherwise(pl.col(col))
    #     .alias(col) for col in ratio_columns
    # ])

    # grouped_df = lagged_liquidations.group_by(['token_col', 'token_debt']).agg([
    #     pl.col('debt_to_collat_ratio').list().alias('ratios')
    # ])

    # lagged_liquidations.group_by(['token_col', 'token_debt']).map_groups(split_group)

    return lagged_liquidations


def categorical_encoding(lagged_liquidations):
    # Find unique tokens in both columns
    unique_token_symbols = pl.concat([
        pl.col('token_col'),
        pl.col('token_debt'),
    ]).alias('token').unique()
    unique_tokens = lagged_liquidations.select(unique_token_symbols).to_series()

    # Create a dictionary with consistent encoding
    token_mapping = {token: idx for idx, token in enumerate(unique_tokens)}
    # {'DAI': 0, 'WETH': 1, 'WBTC': 2, 'USDC': 3, 'USDT': 4}

    print("Token Mapping:", token_mapping)

    lagged_liquidations = lagged_liquidations.with_columns([
        lagged_liquidations["token_col"].map_elements(lambda x: token_mapping[x]).alias("token_col"),
        lagged_liquidations["token_debt"].map_elements(lambda x: token_mapping[x]).alias("token_debt")
    ])

    return lagged_liquidations


def one_hot_encoding(lagged_liquidations):
    # Find unique tokens in both columns
    unique_token_symbols = pl.concat([
        pl.col('token_col'),
        pl.col('token_debt'),
    ]).alias('token').unique()
    unique_tokens = lagged_liquidations.select(unique_token_symbols).to_series()

    for token in unique_tokens:
        lagged_liquidations = lagged_liquidations.with_columns([
            (pl.col('token_col') == token).cast(pl.UInt8).alias(f"{token}_col"),
            (pl.col('token_debt') == token).cast(pl.UInt8).alias(f"{token}_debt")
        ])

    # lagged_liquidations = lagged_liquidations.drop(["token_col", "token_debt"])

    return lagged_liquidations


# Find the first earlier day where the debt-to-collateral ratio was greater
def find_start_of_borrow(row):
    liquidation_ratio = row['debt_to_collat_ratio']

    for i in range(1, 31):
        day_ratio = row[f'debt_to_collat_ratio_t_{i}']

        if day_ratio > liquidation_ratio:
            return f't-{i}'

    return None  # Return None if no previous day's ratio is higher


def split_data(lagged_liquidations: pl.DataFrame):
    lagged_liquidations = lagged_liquidations.with_columns([
        pl.col("day").sort().over(["token_col", "token_debt"]),  # Ensure data is sorted by date within each group if needed
        pl.count().over(["token_col", "token_debt"]).alias("group_size"),
        pl.arange(0, pl.count()).over(["token_col", "token_debt"]).alias("row_index")
        # pl.cumcount.over(["token_col", "token_debt"]).alias("row_number")
    ])

    train_liquidations = lagged_liquidations.filter(pl.col("row_index") < (pl.col("group_size") * 0.8))
    test_liquidations = lagged_liquidations.filter(pl.col("row_index") >= (pl.col("group_size") * 0.8))

    X_train = train_liquidations.drop(["group_size", "row_index", "day", "debt_to_collat_ratio"])
    y_train = train_liquidations.select(["day", "debt_to_collat_ratio"])

    X_test = test_liquidations.drop(["group_size", "row_index", "day", "debt_to_collat_ratio"])
    y_test = test_liquidations.select(["day", "debt_to_collat_ratio"])

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()


# def train_model(X_train: np.ndarray, y_train: np.ndarray):
def train_model(liquidations: pl.DataFrame):
    X = liquidations.drop(['day', 'debt_to_collat_ratio']).to_numpy()
    y = liquidations.select(['day', 'debt_to_collat_ratio']).to_numpy()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Setup and train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # return model
    return model, X_train, X_test, y_train, y_test


def simplify_model(model, X_train, y_train, X_test, y_test):
    model, transformer = mcr(model = model,
                         X_train = X_train,
                         y_train = y_train,
                         X_eval = X_test,
                         y_eval = y_test,
                         eval_metric = 'rmse',
                         transform_features = True)

    return model, transformer


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    # Predict and evaluate
    predictions = model.predict(X_test)
    print(f"-- predictions: {predictions}")

    # mse = mean_squared_error(y_test, predictions)
    mse = root_mean_squared_error(y_test, predictions)

    print(f"Mean Squared Error: {mse}")


if __name__ == "__main__":
    liquidations = preprocess_data()

    # X_train, y_train, X_test, y_test = split_data(liquidations)
    # model = train_model(X_train, y_train)

    model, X_train, X_test, y_train, y_test = train_model(liquidations)

    evaluate_model(model, X_test, y_test)

    print(model.get_params())

    model, transformer = simplify_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(model, transformer.transform(X_test), y_test)

    print(model.get_params())

    # model.save("token_prices.keras")
    # model.save("pred_liquid_mod.h5")

    serialize_model(model, "predict_liquidations_model.json")


# Scalers:

#   - token_col_t_i --> minmax
#   - token_debt_t_i --> minmax

#   - collateral_amount
#   - debt_amount

#   - col_value_USD & debt_value_USD