import polars as pl
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
# 738818
def generate_dataset():
    start_date = datetime(2024, 1, 1)
    date_list = [start_date + timedelta(days=x) for x in range(100)]
    future_date_list = [date + timedelta(days=30) for date in date_list]

    return pl.DataFrame({
        'current_date': date_list,
        'future_date': future_date_list
    })

def preprocess_data(dates: pl.DataFrame):
    return dates.with_columns([
        pl.col('current_date').map_elements(lambda x: x.toordinal()).alias('current_date_ordinal'),
        pl.col('future_date').map_elements(lambda x: x.toordinal()).alias('future_date_ordinal')
    ])

def train_model(dates: pl.DataFrame):
    # Extract data for sklearn
    X = dates['current_date_ordinal'].to_numpy().reshape(-1, 1)
    y = dates['future_date_ordinal'].to_numpy().reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # print("X_scaled")
    # print(X_scaled)

    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    return model, scaler_X, scaler_y

def predict_future_date(model: LinearRegression, date: str, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler):
    ordinal_date = datetime.strptime(date, '%Y-%m-%d').toordinal()
    # Reshape ordinal_date to a 2D array with a single sample and single feature
    ordinal_date_reshaped = np.array([[ordinal_date]])

    scaled_date = scaler_X.transform(ordinal_date_reshaped)

    predicted_ordinal = model.predict(scaled_date)

    print(f"predicted_ordinal_before_scaling: {predicted_ordinal}")

    predicted_ordinal = scaler_y.inverse_transform(predicted_ordinal)

    print(f"predicted_ordinal_after_scaling: {predicted_ordinal}")

    predicted_date = datetime.fromordinal(int(predicted_ordinal[0][0])) # TODO

    return predicted_date.strftime("%Y-%m-%d")


def save_model_to_onnx(model):
    # Define the initial types for the onnx model
    initial_type = [('float_input', FloatTensorType([None, 1]))]
    # initial_type = [("int_input", Int64TensorType([None, 1]))]

    # Convert the scikit-learn model to onnx
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to a file
    with open("future_date.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


def test_onnx_model(scaler_X: MinMaxScaler, scaler_y: MinMaxScaler):
    # Example input date
    date = '2023-10-25'
    ordinal_date = datetime.strptime(date, '%Y-%m-%d').toordinal()

    # Prepare input in the shape (1, 1) for a single prediction
    input_data = np.array([[ordinal_date]], dtype=np.float32)  # Ensure it's float32
    # input_data = [[738818]]

    print(f"input_data: {input_data}")

    scaled_input = scaler_X.transform(input_data)  # Scale before inference
    print(f"scaled_input: {scaled_input}")

    min_val = scaler_X.data_min_[0]
    max_val = scaler_X.data_max_[0]
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    scaled_input_v2 = (input_data - min_val) / (max_val - min_val)

    print(f"scaled_input: {scaled_input}")
    print(f"scaled_input_v2: {scaled_input_v2}")

    session = ort.InferenceSession("future_date.onnx")

    # Specify the input name (should match the name defined during conversion)
    input_name = session.get_inputs()[0].name
    print(f"input_name: {input_name}")

    # Run inference
    predictions = session.run(None, {input_name: scaled_input_v2})

    print(f"predictions: {predictions}")

    # Get the output (assuming it's a regression model, it will be a single value)
    predicted_output = predictions[0].item()

    predicted_output = scaler_y.inverse_transform([[predicted_output]])

    predicted_date = datetime.fromordinal(int(predicted_output[0][0]))

    min_val_y = scaler_y.data_min_[0]
    max_val_y = scaler_y.data_max_[0]

    # print(f"min_val_x: {min_val}")
    # print(f"max_val_x: {max_val}")
    # print(f"min_val_y: {min_val_y}")
    # print(f"max_val_y: {max_val_y}")

    test = predictions[0].item()
    original_value = test * (max_val_y - min_val_y) + min_val_y
    print(f"original_value: {original_value} && ordinal: {datetime.fromordinal(int(original_value))}")

    predicted_output = predicted_date.strftime("%Y-%m-%d")
    print("Predicted output:", predicted_output)

if __name__ == "__main__":
    dates = generate_dataset()

    dates = preprocess_data(dates)

    model, scaler_X, scaler_y = train_model(dates)

    # Example Usage
    # input_date = '2023-10-25'
    # predicted_future_date = predict_future_date(model, input_date, scaler_X, scaler_y)
    # print(f'Given the date {input_date}, the predicted future date is {predicted_future_date}')

    # save_model_to_onnx(model)
    test_onnx_model(scaler_X, scaler_y)
