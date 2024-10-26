import argparse
from datetime import datetime
import logging
import os
import pprint

from logging import getLogger

import numpy as np
from dotenv import find_dotenv, load_dotenv
from giza.agents import AgentResult, GizaAgent

# from addresses import ADDRESSES
from ape.contracts.base import ContractInstance

load_dotenv(find_dotenv())

os.environ["DEV_PASSPHRASE"] = os.environ.get("DEV_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")

logging.basicConfig(level=logging.INFO)


def process_data(date):
    ordinal_date = datetime.strptime(date, '%Y-%m-%d').toordinal()

    return [ordinal_date]


def get_data():
    # TODO
    # hardcoding the values for now

    liquidity_pool = ''
    debt_token = ''
    debt_token_amount = 200
    collat_token = ''
    collat_token_amount = 200

    return liquidity_pool, debt_token, debt_token_amount, collat_token, collat_token_amount, '2023-10-25'


def create_agent(
    model_id: int, version_id: int, chain: str, contracts: dict, account: str
):
    """
    Create a Giza agent for the liquidation prediction model
    """
    agent = GizaAgent(
        contracts=contracts,
        id=model_id,
        version_id=version_id,
        chain=chain,
        account=account,
    )

    return agent


def predict(agent: GizaAgent, date: np.ndarray):
    """
    Predict the expected liquidation date

    Args:
        X (np.ndarray): Input to the model.

    Returns:
        int: Predicted value.
    """
    prediction = agent.predict(input_feed={"val": date}, verifiable=True, job_size="S")

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


def execute(model_id, version_id):
    logger = getLogger("agent_logger")

    liquidity_pool, debt_token, debt_token_amount, collat_token, collat_token_amount, date = get_data()
    model_input = process_data(date)

    contracts = {
        "liquidation_prediction": "0x07094ef7ec0875deead70e2c3aa23770ea5b2625",
    }

    agent = create_agent(
        model_id,
        version_id,
        f"ethereum:sepolia:{sepolia_rpc_url}",
        contracts["liquidation_prediction"],
        "ape_account_1"
    )

    prediction = predict(agent, model_input)

    predicted_liquidation_date = get_pred_val(prediction)


    with agent.execute() as contracts:
        logger.info("Executing contract")

        contracts.liquidation_prediction.addPrediction(
            liquidity_pool,
            debt_token,
            debt_token_amount,
            collat_token, collat_token_amount,
            predicted_liquidation_date
        )



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model-id", metavar="M", type=int, help="model-id")
    parser.add_argument("--version-id", metavar="V", type=int, help="version-id")
    parser.add_argument("--tokenA-amount", metavar="A", type=int, help="tokenA-amount")
    parser.add_argument("--tokenB-amount", metavar="B", type=int, help="tokenB-amount")

    # Parse arguments
    args = parser.parse_args()

    MODEL_ID = args.model_id
    VERSION_ID = args.version_id
    tokenA_amount = args.tokenA_amount
    tokenB_amount = args.tokenB_amount

    execute(tokenA_amount, tokenB_amount, MODEL_ID, VERSION_ID)