from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from agent.agent import execute_agent
from model.predict_tokens import initialize_data_for_predictions, predict_future_prices
from fastapi.middleware.cors import CORSMiddleware
from model import predict_tokens

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class PredictionRequest(BaseModel):
    borrowToken: str
    borrowAmount: int
    collateralToken: str
    collateralAmount: int
    # liquidation_date: str
    # token_col: str
    # token_debt: str
    # collateral_amount: float
    # col_value_USD: float
    # debt_amount: float
    # debt_amount_USD: float


@app.on_event("startup")
async def prepare_data_for_predictions():
    initialize_data_for_predictions()


@app.get("/predict_token_prices")
async def predict_token_prices():
    predictions = predict_future_prices(
        predict_tokens.model,
        predict_tokens.last_data_sequence,
        predict_tokens.scaler,
        predict_tokens.last_date,
        predict_tokens.columns,
        90
    )

    predictions = predictions.select(["date", "price_DAI", "price_WETH", "price_WBTC", "price_USDC", "price_USDT"])

    return predictions.to_dicts()


@app.post("/predict_liquidation_date")
async def predict_liquidation(data: PredictionRequest):
    # model_id = 926
    # version_id = 2

    # debt_token = 'USDC'
    # debt_token_amount = 2175 # 2618 * 0.83 / 0.999122 = 2174.84951788 --> 2024-03-05 (29jrs apres)
    # # debt_token_amount = 2180.41 # 2627 * 0.83 = 2180.41 --> 2024-03-04 (28jrs apres)
    # collat_token = 'WETH'
    # collat_token_amount = 1
    print("--- predict prices ---")
    print(data.borrowToken)
    print(data.borrowAmount)
    print(data.collateralToken)
    print(data.collateralAmount)

    predictedLiquidationDate = await execute_agent(data.borrowToken, data.borrowAmount, data.collateralToken, data.collateralAmount)

    return {
        "borrowToken": data.borrowToken,
        "borrowAmount": data.borrowAmount,
        "collateralToken": data.collateralToken,
        "collateralAmount": data.collateralAmount,
        "predictedLiquidationDate": predictedLiquidationDate
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
