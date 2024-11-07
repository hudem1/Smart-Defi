from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import numpy as np
import polars as pl

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model

from model.common_data_prepocessing import common_preprocess
from model.predict_tokens import create_sequences, split_data
from agent.agent import execute_agent


# Global variables to hold the data for model prediction
model = None
last_data_sequence = None
scaler = None
last_date = None
columns = None

app = FastAPI()

class PredictionRequest(BaseModel):
    liquidation_date: str
    token_col: str
    token_debt: str
    collateral_amount: float
    col_value_USD: float
    debt_amount: float
    debt_amount_USD: float


@app.on_event("startup")
async def prepare_data_for_predictions():
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


@app.get("/api/py/helloFastApi")
def hello_fast_api():
    return {"message": "Hello from FastAPI hihihaha"}


@app.get("/predict")
def predict_prices():
    model_id = 111
    version_id = 11

    debt_token = None
    debt_token_amount = None
    collat_token = None
    collat_token_amount = None

    execute_agent(model_id, version_id, debt_token, debt_token_amount, collat_token, collat_token_amount)


@app.post("/predict/temp")
async def predict(data: PredictionRequest):
    global scaler  # Use the scaler from the global scope

    # Convert input data into the format required by your model
    input_data = np.array([
        data.collateral_amount,
        data.col_value_USD,
        data.debt_amount,
        data.debt_amount_USD,
        # Include additional features as needed
    ]).reshape(1, -1)

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
