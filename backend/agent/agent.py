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
from giza.agents.model import GizaModel

from starknet_py.net.client_models import Call
from starknet_py.hash.selector import get_selector_from_name

# from addresses import ADDRESSES
from ape.contracts.base import ContractInstance

from model.predict_tokens import predict_future_prices
from model import predict_tokens

from starknet_py.contract import Contract
from starknet_py.net.account.account import Account
from starknet_py.net.models import StarknetChainId
from starknet_py.net.signer.stark_curve_signer import KeyPair
from starknet_py.net.full_node_client import FullNodeClient

load_dotenv(find_dotenv())

PASSPHRASE = os.environ.get("H2_AA_3_PASSPHRASE")
SEPOLIA_RPC_URL = os.environ.get("SEPOLIA_RPC_URL")
LOCAL_RPC_URL = "http://127.0.0.1:8545"

SN_USER_ADDRESS = os.environ.get("SN_USER_ADDRESS")
SN_PRIVATE_KEY = os.environ.get("SN_SEPOLIA_PRIV_KEY")
SN_CONTRACT_ADDRESS = "0x039219276637883fe0c3177c6d310854bdcf2f502ab782ac73e3dc8664e91e69"

NODE_URL = os.environ.get("SN_SEPOLIA_RPC_URL")
sn_client = FullNodeClient(node_url=NODE_URL)

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
    # agent_id: int, contracts: dict, chain: str, account: str
    model_id: int, version_id: int
):
    """
    Create a Giza agent for the liquidation prediction model
    """
    # agent = GizaAgent.from_id(
    #     id=agent_id,
    #     contracts=contracts,
    #     chain=chain,
    #     account=account,
    # )

    model = GizaModel(
        id=model_id,
        version=version_id
    )

    return model


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
    (result, request_id) = agent.predict(input_feed={"float_input": input}, verifiable=True, dry_run=True, model_category="XGB", job_size='S')
    print(f"--- prediction: {result}")

    # [1 1] [48419176448 0]
    return result


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


async def execute_agent(debt_token, debt_token_amount, collat_token, collat_token_amount):
    logger = getLogger("agent_logger")

    token_predictions = get_data(debt_token, collat_token)
    print(f"token_predictions: {token_predictions}")

    predicted_debt_to_collat_ratios = preprocess_data(token_predictions, debt_token, debt_token_amount, collat_token, collat_token_amount)
    print(f"predicted_debt_to_collat_ratios: {predicted_debt_to_collat_ratios}")

    contracts = {
        "liquidation_prediction": "0x5110BEbECcE7ee99BB45073f71b6fbF46c4Aa75e", # sepolia
    }

    # agent = create_agent(
    #     117,
    #     contracts,
    #     # f"ethereum:sepolia:geth", # caused some limits to tx that can be sent
    #     f"ethereum:sepolia:{SEPOLIA_RPC_URL}",
    #     "h2_aa_3"
    # )

    # print(agent.version)
    # print(agent.account) # h2_aa_3
    # print(agent.api_client.url) # https://api.gizatech.xyz/api/v1
    # print(agent.api_client.api_key) # None
    # print(agent.chain) # ethereum:local:test
    # print(agent.uri) # https://endpoint-hudem2-924-1-084bd29e-7i3yxzspbq-ew.a.run.app/cairo_run
    # print(agent.contract_handler)
    # print(agent.endpoints_client)
    # print(agent.endpoint_id) # 453
    # print(agent.framework) # CAIRO
    # print(agent.session) # None

    agent = create_agent(model_id=926, version_id=2)

    prediction = predict_liquidation(agent, predicted_debt_to_collat_ratios.to_numpy().flatten())

    print(f"prediction: {prediction}")

    # predicted_date = get_pred_val(prediction)

    predicted_date = overwrite_weird_prediction(predicted_debt_to_collat_ratios)
    print(f"--- predicted_date: {predicted_date}")

    sn_account = Account(
        address=SN_USER_ADDRESS,
        client=sn_client,
        key_pair=KeyPair.from_private_key(SN_PRIVATE_KEY),
        chain=StarknetChainId.SEPOLIA,
    )

    sn_contract = await Contract.from_address(provider=sn_account, address=SN_CONTRACT_ADDRESS)

    invocation = await sn_contract.functions["add_prediction"].invoke_v1(
        debt_token={debt_token: None},
        debt_amount=debt_token_amount,
        collat_token={collat_token: None},
        collat_amount=collat_token_amount,
        predicted_liquidation_date=predicted_date,
        max_fee=int(1e14)
    )
    print(f"--- calldata: {invocation.invoke_transaction.calldata} ---")

    await invocation.wait_for_acceptance()
    print(f"--- tx hash: {invocation.hash} ---")

    ### raw contract call ###
    # call = Call(
    #     to_addr=SN_CONTRACT_ADDRESS,
    #     selector=get_selector_from_name("add_prediction"),
    #     calldata=[
    #         token_mapping[debt_token],
    #         get_lower_128_bits(debt_token_amount),
    #         get_higher_128_bits(debt_token_amount),
    #         token_mapping[collat_token],
    #         get_lower_128_bits(collat_token_amount),
    #         get_higher_128_bits(collat_token_amount),
    #         get_lower_128_bits(predicted_date),
    #         get_higher_128_bits(predicted_date),
    #     ],
    # )

    # await sn_account.client.call_contract(call)

    invocation = await sn_contract.functions["set_test_bool"].invoke_v1(
        value=True, max_fee=int(1e14)
    )
    await invocation.wait_for_acceptance()

    ##### With agent interacting with Ethereum smart contract #####
    # with agent.execute() as contracts:
    #     global token_mapping
    #     logger.info("Executing contract")
    # try:
    #     contracts.liquidation_prediction.setTestBool(True)
    #     contracts.liquidation_prediction.addPrediction(
    #         # liquidity_pool,
    #         token_mapping[debt_token],
    #         debt_token_amount,
    #         token_mapping[collat_token],
    #         collat_token_amount,
    #         predicted_date
    #     )
    # except Exception as e:
    #     print("--- Error ---")
    #     print(f"Error Type: {type(e).__name__}")  # Get the type of the exception
    #     print(f"Error Message: {str(e)}")  # Get the message of the exception

    return datetime.fromordinal(predicted_date).strftime("%Y-%m-%d")


def get_lower_128_bits(x: int):
    return x & ((1 << 128) - 1)

def get_higher_128_bits(x: int):
    return (x >> 128) & ((1 << 128) - 1)


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

    execute_agent(MODEL_ID, VERSION_ID)

