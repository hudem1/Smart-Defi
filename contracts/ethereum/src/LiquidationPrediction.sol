// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LiquidationPrediction {
    struct Prediction {
        address liquidityPool;
        address debtToken;
        uint256 debtTokenAmount;
        address collatToken;
        uint256 collatTokenAmount;
        uint256 predictedLiquidationDate;  // stored as a timestamp
    }

    // Mapping from user address to their list of information
    mapping(address => Prediction[]) public predictions;

    bool public testBool;

    // Event to emit when user info is added
    event PredictionAdded(
        address indexed user,
        address indexed liquidityPool,
        address debtToken,
        uint256 debtTokenAmount,
        address collatToken,
        uint256 collatTokenAmount,
        uint256 predictedLiquidationDate
    );

    function addPrediction(
        address _liquidityPool,
        address _debtToken,
        uint256 _debtTokenAmount,
        address _collatToken,
        uint256 _collatTokenAmount,
        uint256 _predictedLiquidationDate
    ) public {
        // Add the prediction to the user's list in the mapping
        predictions[msg.sender].push(Prediction(
            _liquidityPool,
            _debtToken,
            _debtTokenAmount,
            _collatToken,
            _collatTokenAmount,
            _predictedLiquidationDate
        ));

        emit PredictionAdded(
            msg.sender,
            _liquidityPool,
            _debtToken,
            _debtTokenAmount,
            _collatToken,
            _collatTokenAmount,
            _predictedLiquidationDate
        );
    }

    function getPredictions(address _user) public view returns (Prediction[] memory) {
        return predictions[_user];
    }

    function setTestBool() public {
        testBool = true;
    }
}


// forge create --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 LiquidationPrediction
// cast call 0x5FbDB2315678afecb367f032d93F642f64180aa3 "testBool()"
// cast send --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 0x5FbDB2315678afecb367f032d93F642f64180aa3 "setTestBool()"

// https://endpoint-hudem2-924-1-084bd29e-7i3yxzspbq-ew.a.run.app
