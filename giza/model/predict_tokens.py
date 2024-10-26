from giza.datasets import DatasetsLoader
import polars as pl
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf

import tf2onnx
import onnxruntime as ort

loader = DatasetsLoader()

def preprocess_data():
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

    return prices


# Create sequences for LSTM
def create_sequences(data, seq_length):
    x, y = [], []

    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])

    return np.array(x), np.array(y)


# Split the data into training, validation, and testing sets
def split_data(x: np.ndarray, y: np.ndarray):
    total_samples = x.shape[0]

    train_end = int(total_samples * 0.85)
    val_end = int(total_samples * 0.95)

    x_train = x[:train_end]
    y_train = y[:train_end]

    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]

    x_test = x[val_end:]
    y_test = y[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test


# Build the LSTM model
def create_lstm_model(input_shape: tuple, y_train):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, name='lstm_2'))
    model.add(Dense(y_train.shape[1]))  # Predicting all features (price, market cap, volume)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_model(prices: pl.DataFrame, scaler: MinMaxScaler):
    # Normalize the data
    scaled_prices = scaler.fit_transform(prices[:, 1:].to_numpy()) # first column is date

    # Prepare the sequences
    sequence_length = 30
    x, y = create_sequences(scaled_prices, sequence_length)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)

    # print("--- input shape to model ---")
    # print((sequence_length, x_train.shape[2]))

    # model = create_lstm_model((sequence_length, x_train.shape[2]), y_train)

    # # Train the model
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32)

    # # Evaluate the model on the test set
    # model.evaluate(x_test, y_test)

    return x_test[-1]


def predict_future_prices(model: Sequential, last_sequence, scaler, last_date, columns, future_days=30):
    # To predict future values, use the last available sequence and predict step by step, updating the input sequence at each step.
    # last_sequence = x_test[-1]

    future_predictions = []

    for _ in range(future_days):
        last_prediction = model.predict(last_sequence[np.newaxis, :])
        future_predictions.append(last_prediction)
        last_sequence = np.vstack((last_sequence[1:], last_prediction))

    # Inverse transform the predictions to get them back on the original scale
    future_predictions = scaler.inverse_transform(np.squeeze(np.array(future_predictions)))

    # Create a date range starting from the day after the last known date
    future_dates = pl.date_range(
        start=last_date,
        end=last_date + pl.duration(days=future_days),
        interval="1d",
        closed="right",
        eager=True # immediately converts the result to a Series instead of an expression
    ).dt.date()

    # Combine the future dates with the predicted data
    predicted_df = pl.DataFrame(future_predictions, columns)
    predicted_df.insert_column(0, future_dates)

    return predicted_df


# Convert the TensorFlow model to ONNX format
def save_model_to_onnx(model, x_sample):
    spec = (tf.TensorSpec((None, x_sample.shape[0], x_sample.shape[1]), tf.float32, name="input"),)

    output_path = "model.onnx"
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path, opset=13)


def evaluate_onnx_model(scaler, last_sequence, columns, last_date):
    # Load the ONNX model
    sess = ort.InferenceSession("model.onnx")

    # Prepare the input data.
    # Assuming last_sequence is your input and it needs to be reshaped or processed to match the input requirements of the model.
    input_data = last_sequence.astype(np.float32)  # Ensure dtype is float32
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension if needed

    # Get the name of the input from the model
    input_name = sess.get_inputs()[0].name # input
    output_name = sess.get_outputs()[0].name # dense

    # Prepare the input dictionary
    input_dict = {input_name: input_data}

    # Run inference
    output = sess.run([output_name], input_dict)

    # Output the results
    print("Predicted output:")
    output = scaler.inverse_transform(output[0])

    # Create a date range starting from the day after the last known date
    # future_dates = pl.date_range(
    #     start=last_date,
    #     end=last_date + pl.duration(days=60),
    #     interval="1d",
    #     closed="right",
    #     eager=True # immediately converts the result to a Series instead of an expression
    # ).dt.date()
    future_date = last_date + pl.duration(days=1)

    # Combine the future dates with the predicted data
    predicted_df = pl.DataFrame(output, columns)
    date_series = pl.Series("date", [future_date.dt.date()])
    predicted_df.insert_column(0, date_series)

    # Display the DataFrame
    print('--- predicted_df ---')
    print(predicted_df)


if __name__ == "__main__":
    print("--- preprocess_data ---")
    prices = preprocess_data()

    print(prices)

    scaler = MinMaxScaler(feature_range=(0, 1))

    print("--- training model ---")
    last_data_sequence = train_model(prices, scaler)

    # print("--- last sequence ---")
    # # print(last_data_sequence)
    # print(len(last_data_sequence))

    # print("--- saving model as onnx ---")
    # save_model_to_onnx(model, last_data_sequence)

    # print("--- predictions ---")
    # predictions = predict_future_prices(
    #     model,
    #     last_data_sequence,
    #     scaler,
    #     prices.select(pl.col('date').last()).item(),
    #     prices.columns[1:], # without the date
    #     60
    # )

    # print("--- display predictions ---")
    # print(predictions)
    print("--- display predictions ---")
    evaluate_onnx_model(
        scaler,
        last_data_sequence,
        prices.columns[1:],
        prices.select(pl.col('date').last()).item()
    )