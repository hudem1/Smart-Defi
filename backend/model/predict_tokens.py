from giza.datasets import DatasetsLoader
import polars as pl
from datetime import datetime, timedelta

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf

import tf2onnx
import onnxruntime as ort

from giza.zkcook import mcr

from .common_data_prepocessing import common_preprocess

loader = DatasetsLoader()


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

    model = create_lstm_model((sequence_length, x_train.shape[2]), y_train)

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32)

    # Evaluate the model on the test set
    model.evaluate(x_test, y_test)

    return model, x_test[-1]


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

    output_path = "lstm_model.onnx"
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path, opset=13)


def evaluate_onnx_model(scaler, last_sequence, columns, last_date):
    # Load the ONNX model
    sess = ort.InferenceSession("lstm_model.onnx")

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
    print(f"last_date: {last_date}")
    # future_date = last_date + pl.duration(days=1)
    future_date = last_date + timedelta(days=1)

    # Combine the future dates with the predicted data
    predicted_df = pl.DataFrame(output, columns)
    date_series = pl.Series("date", [future_date])
    predicted_df.insert_column(0, date_series)

    # Display the DataFrame
    print('--- predicted_df ---')
    print(predicted_df)


def reduce_model(model, x_train, y_train, x_test, y_test):
    model, transformer = mcr(model = model,
                         X_train = x_train,
                         y_train = y_train,
                         X_eval = x_test,
                         y_eval = y_test,
                         eval_metric = 'rmse',
                         transform_features = True)




model = None
last_data_sequence = None
scaler = None
last_date = None
columns = None

def initialize_data_for_predictions():
    global model, last_data_sequence, scaler, last_date, columns
    print("--- preprocess_data ---")
    prices, _ = common_preprocess()

    print(prices)

    scaler = MinMaxScaler(feature_range=(0, 1))

    print("--- training model ---")
    # Normalize the data
    scaled_prices = scaler.fit_transform(prices[:, 1:].to_numpy()) # first column is date

    # Prepare the sequences
    sequence_length = 30
    x, y = create_sequences(scaled_prices, sequence_length)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)
    last_data_sequence = x_test[-1]

    last_date = prices.select(pl.col('date').last()).item()

    columns = prices.columns[1:] # without the date

    model = load_model('token_prices.keras')


if __name__ == "__main__":
    print("--- preprocess_data ---")
    prices, _ = common_preprocess()

    print(prices)

    scaler = MinMaxScaler(feature_range=(0, 1))

    print("--- training model ---")
    model, last_data_sequence  = train_model(prices, scaler)

    print("--- last sequence ---")
    print(last_data_sequence)
    print(last_data_sequence.shape)
    print(len(last_data_sequence))

    # model.save("token_prices.keras")

    # print("--- saving model as onnx ---")
    # save_model_to_onnx(model, last_data_sequence)

    print("--- predictions ---")
    predictions = predict_future_prices(
        model,
        last_data_sequence,
        scaler,
        prices.select(pl.col('date').last()).item(),
        prices.columns[1:], # without the date
        30
    )

    print(predictions)

    # print("--- display predictions ---")
    # print(predictions)

    # print("--- display predictions ---")
    # evaluate_onnx_model(
    #     scaler,
    #     last_data_sequence,
    #     prices.columns[1:],
    #     prices.select(pl.col('date').last()).item()
    # )

    # print("--- mcr ---")
    # reduce_model(model, x_train, y_train, x_test, y_test)