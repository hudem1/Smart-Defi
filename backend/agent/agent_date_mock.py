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

PASSPHRASE = os.environ.get("H2_AA_2_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")
local_rpc_url = "http://127.0.0.1:8545"

logging.basicConfig(level=logging.INFO)


min_val_x = 738886.0
max_val_x = 738985.0

min_val_y = 738916.0
max_val_y = 739015.0

def get_data():
    # TODO
    # hardcoding the values for now

    liquidity_pool = ''
    debt_token = ''
    debt_token_amount = 200
    collat_token = ''
    collat_token_amount = 200

    return liquidity_pool, debt_token, debt_token_amount, collat_token, collat_token_amount, '2025-03-16'



def process_data(date):
    ordinal_date = datetime.strptime(date, '%Y-%m-%d').toordinal()

    scaled_input = (ordinal_date - min_val_x) / (max_val_x - min_val_x)

    return np.array([[scaled_input]], dtype=np.float32)
    # return np.array([[ordinal_date]]).astype(np.float32)



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


def predict(agent: GizaAgent, date: np.ndarray):
    """
    Predict the expected liquidation date

    Args:
        X (np.ndarray): Input to the model.

    Returns:
        int: Predicted value.
    """
    print(f"date: {date}") # [738818]
    # if isinstance(date, np.ndarray):
    #     print("--- instance !! ---")
    # date = [[738818.]]
    prediction = agent.predict(input_feed={"float_input": date}, verifiable=True, dry_run=True, model_category="ONNX_ORION", job_size='S')
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



def execute(model_id, version_id):
    logger = getLogger("agent_logger")

    liquidity_pool, debt_token, debt_token_amount, collat_token, collat_token_amount, date = get_data()
    model_input = process_data(date)

    print(f"model_input: {model_input}")

    contracts = {
        "liquidation_prediction": "0x03228C3D322a8560ADEBE1890Ae992755e5A4A1c", # sepolia
        # "liquidation_prediction": "0x5FbDB2315678afecb367f032d93F642f64180aa3", # local
    }

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

    prediction = predict(agent, model_input)

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

    execute(MODEL_ID, VERSION_ID)




# curl -X POST https://endpoint-hudem2-924-1-084bd29e-7i3yxzspbq-ew.a.run.app/cairo_run \
#      -H "Content-Type: application/json" \
#      -d '{
#            "args": "[\"2\", \"2\", \"2\", \"4\", \"1\", \"2\", \"3\", \"4\"]"
#          }'

# https://endpoint-hudem2-924-1-084bd29e-7i3yxzspbq-ew.a.run.app
# 'job_size': 'M', 'args': '1 1 48419176448 0'

# curl -X POST 'https://endpoint-hudem2-924-1-084bd29e-7i3yxzspbq-ew.a.run.app/cairo_run' -H 'Content-Type: application/json' -d '{"job_size": "M", "args": "[1 1] [48419176448 0]"}'
# curl -X POST 'https://endpoint-hudem2-924-10-e9d5a3a9-7i3yxzspbq-ew.a.run.app' -H 'Content-Type: application/json' -d '{"job_size": "S", "args": "[1 1] [45014 1]", "dry_run": "True"}'



# - check curl max u32
# - check doc -> transpile to cairo if float with u64
# - check if can generate an onnx with bigger than floattensortype



#  https://endpoint-hudem2-924-10-e9d5a3a9-7i3yxzspbq-ew.a.run.app

# 0x1399aB056cDF9C506a46cBa9f9674e361cB92cC6
# 0x2b7cf792AA87F723918b8b09DA7A8BFe4839B7fA