import argparse
from datetime import datetime, timedelta
import logging
import os
import pprint

from logging import getLogger

import polars as pl

import numpy as np
from dotenv import find_dotenv, load_dotenv
from giza.agents import AgentResult, GizaAgent

# from addresses import ADDRESSES
from ape.contracts.base import ContractInstance

# from globals import model, last_data_sequence, scaler, last_date, columns

from model.predict_tokens import predict_future_prices
from model import predict_tokens

load_dotenv(find_dotenv())

PASSPHRASE = os.environ.get("H2_AA_3_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")
local_rpc_url = "http://127.0.0.1:8545"

token_mapping = {'DAI': 0, 'WETH': 1, 'WBTC': 2, 'USDC': 3, 'USDT': 4}

logging.basicConfig(level=logging.INFO)

def get_data(debt_token, collat_token):
    print("--- get token prices predictions ---")
    print(f"predict_tokens.model: {predict_tokens.model}")
    predictions = predict_future_prices(
        predict_tokens.model,
        predict_tokens.last_data_sequence,
        predict_tokens.scaler,
        predict_tokens.last_date,
        predict_tokens.columns,
        30
    )

    print(predictions)

    predictions = predictions.select(["date", f"price_{debt_token}", f"price_{collat_token}"])

    return predictions


def preprocess_data(token_predictions, debt_token, debt_token_amount, collat_token, collat_token_amount):
    global token_mapping

    schema = [("token_col", pl.Int64), ("token_debt", pl.Int64)] + [(f"debt_to_collat_ratio_t_{i + 1}", pl.Float64) for i in range(30)]
    input_data = pl.DataFrame(schema=schema)

    new_row = {
        "token_col": token_mapping[collat_token],
        "token_debt": token_mapping[debt_token],
    }

    # Calculate debt-to-collateral ratios for each prediction
    for i in range(len(token_predictions) - 1, -1, -1):
        price_debt = token_predictions[f"price_{debt_token}"][i]
        price_collat = token_predictions[f"price_{collat_token}"][i]

        # Calculate the debt-to-collateral ratio
        debt_to_collat_ratio = (price_debt * debt_token_amount) / (price_collat * collat_token_amount)

        # Add the calculated ratio to the new row
        new_row[f"debt_to_collat_ratio_t_{30 - i}"] = debt_to_collat_ratio

    input_data = input_data.vstack(pl.DataFrame(new_row))

    print("--- debt-to-collat ratios ---")
    print(input_data)
    # debt_to_collat_ratio = (
    #     (pl.col(f"price_{debt_token}") * debt_token_amount) /
    #     (pl.col(f"price_{collat_token}") * collat_token_amount)
    # )

    # for i in range(30):
    #     new_row[f"debt_to_collat_ratio_t_{i+1}"] = debt_to_collat_ratio[i]

    return input_data



def create_agent(
    agent_id: int, contracts: dict, chain: str, account: str
):
    """
    Create a Giza agent for the liquidation prediction model
    """
    agent = GizaAgent.from_id(
        id=agent_id,
        # version_id=version_id,
        contracts=contracts,
        chain=chain,
        account=account,
    )

    return agent


def predict_liquidation(agent: GizaAgent, input: np.ndarray):
    """
    Predict the expected liquidation date

    Args:
        X (np.ndarray): Input to the model.

    Returns:
        int: Predicted value.
    """
    print(f"--- model input: {input}") # [738818]
    # if isinstance(date, np.ndarray):
    #     print("--- instance !! ---")
    # date = [[738818.]]
    prediction = agent.predict(input_feed={"float_input": input}, verifiable=True, dry_run=True, model_category="XGB", job_size='S')
    print(f"--- prediction: {prediction}")
    # [1 1] [48419176448 0]
    return prediction


def get_pred_val(prediction: AgentResult):
    """
    Get the value from the prediction.

    Args:
        prediction (dict): Prediction from the model.

    Returns:
        int: Predicted value.
    """
    # This will block the executon until the prediction has generated the proof
    # and the proof has been verified
    return prediction.value


def overwrite_weird_prediction(predicted_debt_to_collat_ratios: pl.DataFrame):
    liquidation_threshold = 0.83

    ratio_columns = predicted_debt_to_collat_ratios.columns[2:]

    for i, col in enumerate(ratio_columns[::-1]):
        print(f"for col {col} : {predicted_debt_to_collat_ratios[col].item()}")
        # 739199
        # print(f"row's value: {predicted_debt_to_collat_ratios[col]}")
        if (predicted_debt_to_collat_ratios[col] >= liquidation_threshold).any():
            return (datetime.now() + timedelta(days=i + 1)).toordinal()

    # oldest_ratio = predicted_debt_to_collat_ratios['debt_to_collat_ratio_t_1'].item(0)
    # latest_ratio = predicted_debt_to_collat_ratios['debt_to_collat_ratio_t_30'].iem(0)

    # ratio_difference = abs(latest_ratio - oldest_ratio)
    # i = 0
    # while oldest_ratio < liquidation_threshold:
    #     oldest_ratio += ratio_difference
    #     i += 30

    return 0

# def postprocess_data(prediction):
#     prediction

#     return datetime.fromordinal(int(unscaled_ordinal)).strftime("%Y-%m-%d")



def execute_agent(debt_token, debt_token_amount, collat_token, collat_token_amount):
    logger = getLogger("agent_logger")

    token_predictions = get_data(debt_token, collat_token)
    print(f"token_predictions: {token_predictions}")

    # Get the minimum price_WETH
    min_price_weth = token_predictions["price_WETH"].min()
    # Filter the DataFrame to get the row with the minimum price_WETH
    min_price_row = token_predictions.filter(pl.col("price_WETH") == min_price_weth)
    print(f"--- minimum price row: {min_price_row}")

    predicted_debt_to_collat_ratios = preprocess_data(token_predictions, debt_token, debt_token_amount, collat_token, collat_token_amount)
    print(f"predicted_debt_to_collat_ratios: {predicted_debt_to_collat_ratios}")

    contracts = {
        "liquidation_prediction": "0x5110BEbECcE7ee99BB45073f71b6fbF46c4Aa75e", # sepolia
        # "liquidation_prediction": "0x5FbDB2315678afecb367f032d93F642f64180aa3", # local
    }

    print(f"passphrase: {PASSPHRASE}")

    agent = create_agent(
        117,
        contracts,
        # f"ethereum:sepolia:geth",
        f"ethereum:sepolia:{sepolia_rpc_url}",
        "h2_aa_3"
    )

    print(agent.version)
    print(agent.account) # h2_aa_3
    print(agent.api_client.url) # https://api.gizatech.xyz/api/v1
    print(agent.api_client.api_key) # None
    print(agent.chain) # ethereum:local:test
    print(agent.uri) # https://endpoint-hudem2-924-1-084bd29e-7i3yxzspbq-ew.a.run.app/cairo_run
    print(agent.contract_handler)
    print(agent.endpoints_client)
    print(agent.endpoint_id) # 453
    print(agent.framework) # CAIRO
    print(agent.session) # None

    prediction = predict_liquidation(agent, predicted_debt_to_collat_ratios.to_numpy().flatten())

    print(f"prediction: {prediction}")

    predicted_date = get_pred_val(prediction)

    predicted_date = overwrite_weird_prediction(predicted_debt_to_collat_ratios)
    print(f"--- predicted_date: {predicted_date}")

    # predicted_liquidation_date = postprocess_data(scaled_prediction)

    # print(f"predicted_liquidation_date: {predicted_liquidation_date}")

    with agent.execute() as contracts:
        global token_mapping

        logger.info("Executing contract")

        # contracts.liquidation_prediction.setTestBool(False)

        try:
            contracts.liquidation_prediction.addPrediction(
                # liquidity_pool,
                token_mapping[debt_token],
                debt_token_amount,
                token_mapping[collat_token],
                collat_token_amount,
                predicted_date
            )
        except Exception as e:
            print("--- Error ---")
            print(f"Error Type: {type(e).__name__}")  # Get the type of the exception
            print(f"Error Message: {str(e)}")  # Get the message of the exception



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model-id", metavar="M", type=int, help="model-id")
    parser.add_argument("--version-id", metavar="V", type=int, help="version-id")

    # Parse arguments
    args = parser.parse_args()

    MODEL_ID = args.model_id or 924
    VERSION_ID = args.version_id or 10

    print(f"modelid: {MODEL_ID}")
    print(f"versionid: {VERSION_ID}")

    execute_agent(MODEL_ID, VERSION_ID)

