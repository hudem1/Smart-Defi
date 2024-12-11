use starknet::{ContractAddress};
use LiquidationPrediction::{Prediction, Token};

#[starknet::interface]
pub trait ILiquidationPrediction<TContractState> {
    fn add_prediction(
        ref self: TContractState,
        // liquidity_pool: ContractAddress,
        debt_token: Token,
        debt_amount: u256,
        collat_token: Token,
        collat_amount: u256,
        predicted_liquidation_date: u256
    );
    fn get_predictions(self: @TContractState, user: ContractAddress) -> Array<Prediction>;
    fn get_test_bool(self: @TContractState) -> bool;
    fn set_test_bool(ref self: TContractState, value: bool);
}

#[starknet::contract]
mod LiquidationPrediction {
    use starknet::storage::VecTrait;
    use starknet::{ContractAddress};
    use starknet::storage::{Map, Vec, MutableVecTrait, StoragePathEntry, StoragePointerWriteAccess};
    use core::starknet::get_caller_address;

    // Pb with serialization/deseralization
    #[derive(Drop, Serde, starknet::Store)]
    pub enum Token {
        DAI,
        WETH,
        WBTC,
        USDC,
        USDT
    }

    // impl SerdeTokenImpl of Serde<Token> {
    //     fn serialize(self: @Token, ref output: Array<felt252>) {
    //         match self {
    //             Token::DAI => output.append(0),
    //             Token::WETH => output.append(1),
    //             Token::WBTC => output.append(2),
    //             Token::USDC => output.append(3),
    //             Token::USDT => output.append(4),
    //         };
    //     }

    //     fn deserialize(ref serialized: Span<felt252>) -> Option<Token> {
    //         let token_value = *serialized.pop_front().unwrap();

    //         if token_value == 0 {
    //             Option::Some(Token::DAI)
    //         } else if token_value == 1 {
    //             Option::Some(Token::WETH)
    //         } else if token_value == 2 {
    //             Option::Some(Token::WBTC)
    //         } else if token_value == 3 {
    //             Option::Some(Token::USDC)
    //         } else {
    //             Option::Some(Token::USDT)
    //         }
    //     }
    // }

    #[derive(Drop, Serde, starknet::Store)]
    pub struct Prediction {
        // liquidity_pool: ContractAddress,
        debt_token: Token, // DAI: 0, WETH: 1, WBTC: 2, USDC: 3, USDT: 4
        debt_amount: u256,
        collat_token: Token, // DAI: 0, WETH: 1, WBTC: 2, USDC: 3, USDT: 4
        collat_amount: u256,
        predicted_liquidation_date: u256,  // stored as a timestamp
    }


    #[storage]
    struct Storage {
        test_bool: bool,
        predictions: Map<ContractAddress, Vec<Prediction>>,
    }

    #[abi(embed_v0)]
    impl LiquidationPredictionImpl of super::ILiquidationPrediction<ContractState> {
        fn add_prediction(
            ref self: ContractState,
            // liquidity_pool: ContractAddress,
            debt_token: Token,
            debt_amount: u256,
            collat_token: Token,
            collat_amount: u256,
            predicted_liquidation_date: u256
        ) {
            // assert!(debt_token < 5, "Debt token has only 5 choices: DAI, WETH, WBTC, USDC, USDT");
            // assert!(collat_token < 5, "Collat token has only 5 choices: DAI, WETH, WBTC, USDC, USDT");

            let caller = get_caller_address();
            let prediction = Prediction {
                // liquidity_pool,
                debt_token,
                debt_amount,
                collat_token,
                collat_amount,
                predicted_liquidation_date
            };

            self.predictions.entry(caller).append().write(prediction);
        }

        fn get_predictions(self: @ContractState, user: ContractAddress) -> Array<Prediction> {
            let user_predictions = self.predictions.entry(user);

            let mut result = ArrayTrait::<Prediction>::new();

            for i in 0..user_predictions.len() {
                let pred = user_predictions.at(i).read();
                result.append(pred);
            };

            result
        }

        fn get_test_bool(self: @ContractState) -> bool {
            self.test_bool.read()
        }

        fn set_test_bool(ref self: ContractState, value: bool) {
            self.test_bool.write(value);
        }
    }
}
