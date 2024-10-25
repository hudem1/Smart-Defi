// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {Script, console} from "forge-std/Script.sol";
import {LiquidationPrediction} from "../src/LiquidationPrediction.sol";

contract LiquidationPredictionScript is Script {
    LiquidationPrediction public predTracker;

    function setUp() public {}

    function run() public {
        vm.startBroadcast();

        predTracker = new LiquidationPrediction();

        vm.stopBroadcast();
    }
}
