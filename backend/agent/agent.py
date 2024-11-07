import argparse
from datetime import datetime
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

from app import model, last_data_sequence, scaler, last_date, columns

from model.predict_tokens import predict_future_prices

load_dotenv(find_dotenv())

PASSPHRASE = os.environ.get("H2_AA_2_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")
local_rpc_url = "http://127.0.0.1:8545"

logging.basicConfig(level=logging.INFO)

def get_data(debt_token, collat_token):
    print("--- get token prices predictions ---")
    predictions = predict_future_prices(
        model,
        last_data_sequence,
        scaler,
        last_date,
        columns,
        30
    )

    print(predictions)

    predictions = predictions.select(["date", f"PRICE_{debt_token}", f"PRICE_{collat_token}"])

    return predictions


def process_data(token_predictions, debt_token, debt_token_amount, collat_token, collat_token_amount):
    # schema = ["token_col", "token_debt"] + [f"debt_to_collat_ratio_t_{i + 1}" for i in range(30)]
    # input_data = pl.DataFrame(None, schema=schema)

    token_mapping = {'DAI': 0, 'WETH': 1, 'WBTC': 2, 'USDC': 3, 'USDT': 4}

    new_row = {
        "token_col": token_mapping[collat_token],
        "token_debt": token_mapping[debt_token],
    }

    # Calculate debt-to-collateral ratios for each prediction
    for i in range(len(token_predictions)):
        price_debt = token_predictions[f"price_{debt_token}"][i]
        price_collat = token_predictions[f"price_{collat_token}"][i]

        # Calculate the debt-to-collateral ratio
        debt_to_collat_ratio = (price_debt * debt_token_amount) / (price_collat * collat_token_amount)

        # Add the calculated ratio to the new row
        new_row[f"debt_to_collat_ratio_t_{i + 1}"] = debt_to_collat_ratio

    input_data = pl.DataFrame(new_row)
    # debt_to_collat_ratio = (
    #     (pl.col(f"price_{debt_token}") * debt_token_amount) /
    #     (pl.col(f"price_{collat_token}") * collat_token_amount)
    # )

    # for i in range(30):
    #     new_row[f"debt_to_collat_ratio_t_{i+1}"] = debt_to_collat_ratio[i]

    return input_data.to_numpy()



def create_agent(
    model_id: int, version_id: int, contracts: dict, chain: str, account: str
):
    """
    Create a Giza agent for the liquidation prediction model
    """
    agent = GizaAgent(
        id=model_id,
        version_id=version_id,
        contracts=contracts,
        chain=chain,
        account=account,
    )

    return agent


def predict(agent: GizaAgent, input: np.ndarray):
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
    prediction = agent.predict(input_feed={"float_input": input}, verifiable=True, dry_run=True, model_category="ONNX_ORION", job_size='S')
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
    return prediction.value[0][0]


def postprocess_data(scaled_ordinal):
    unscaled_ordinal = scaled_ordinal * (max_val_y - min_val_y) + min_val_y

    return datetime.fromordinal(int(unscaled_ordinal)).strftime("%Y-%m-%d")



def execute_agent(model_id, version_id, debt_token, debt_token_amount, collat_token, collat_token_amount):
    logger = getLogger("agent_logger")

    token_predictions = get_data(debt_token, collat_token)

    predicted_debt_to_collat_ratios = process_data(token_predictions, debt_token, debt_token_amount, collat_token, collat_token_amount)

    print(f"model_input: {predicted_debt_to_collat_ratios}")

    contracts = {
        "liquidation_prediction": "0x03228C3D322a8560ADEBE1890Ae992755e5A4A1c", # sepolia
        # "liquidation_prediction": "0x5FbDB2315678afecb367f032d93F642f64180aa3", # local
    }

    print(f"passphrase: {PASSPHRASE}")

    agent = create_agent(
        model_id,
        version_id,
        contracts,
        f"ethereum:sepolia:geth",
        # f"ethereum:sepolia:{sepolia_rpc_url}",
        # "ape_account_1"
        "h2_aa_2"
    )

    print(agent.version)
    print(agent.account) # h2_aa_1
    print(agent.api_client.url) # https://api.gizatech.xyz/api/v1
    print(agent.api_client.api_key) # None
    print(agent.chain) # ethereum:local:test
    print(agent.uri) # https://endpoint-hudem2-924-1-084bd29e-7i3yxzspbq-ew.a.run.app/cairo_run
    print(agent.contract_handler)
    print(agent.endpoints_client)
    print(agent.endpoint_id) # 453
    print(agent.framework) # CAIRO
    print(agent.session) # None

    prediction = predict(agent, predicted_debt_to_collat_ratios)

    print(f"prediction: {prediction}")

    scaled_prediction = get_pred_val(prediction)

    predicted_liquidation_date = postprocess_data(scaled_prediction)

    print(f"predicted_liquidation_date: {predicted_liquidation_date}")

    with agent.execute() as contracts:
        logger.info("Executing contract")

        contracts.liquidation_prediction.setTestBool(True)

        # contracts.liquidation_prediction.addPrediction(
        #     liquidity_pool,
        #     debt_token,
        #     debt_token_amount,
        #     collat_token, collat_token_amount,
        #     predicted_liquidation_date
        # )




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

