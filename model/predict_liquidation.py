import polars as pl
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

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
    y = dates['future_date_ordinal'].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    return model

def predict_future_date(model, date):
    ordinal_date = datetime.strptime(date, '%Y-%m-%d').toordinal()
    predicted_ordinal = model.predict([[ordinal_date]])
    predicted_date = datetime.fromordinal(int(predicted_ordinal[0]))

    return predicted_date.strftime("%Y-%m-%d")


def save_model_to_onnx(model):
    # Define the initial types for the onnx model
    initial_type = [('float_input', FloatTensorType([None, 1]))]

    # Convert the scikit-learn model to onnx
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to a file
    with open("future_date.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


if __name__ == "__main__":
    dates = generate_dataset()

    dates = preprocess_data(dates)

    model = train_model(dates)

    # Example Usage
    input_date = '2023-10-25'
    predicted_future_date = predict_future_date(model, input_date)
    print(f'Given the date {input_date}, the predicted future date is {predicted_future_date}')

    save_model_to_onnx(model)
