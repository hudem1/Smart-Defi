"use client";

import { Dispatch, SetStateAction, useState } from 'react';
import { fetchPrediction } from '../api/api';
import { Prediction } from './displayPredictions';

const tokens = ['WETH', 'WBTC', 'USDC', 'USDT', 'DAI'] as const;
type Token = typeof tokens[number];

export interface FormData {
  borrowToken: Token;
  borrowAmount: number;
  collateralToken: Token;
  collateralAmount: number;
}

interface PredictLiquidationFormPros {
    setPredictions: Dispatch<SetStateAction<Prediction[]>>;
    setIsLoading: Dispatch<SetStateAction<{ status: boolean; message: string }>>;
}

const BorrowForm = (props: PredictLiquidationFormPros) => {
  const [formData, setFormData] = useState<FormData>({
    borrowToken: 'WETH',
    borrowAmount: 0,
    collateralToken: 'WETH',
    collateralAmount: 0,
  });

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    const { name, value } = e.target;
    console.log(value)
    setFormData((prev) => ({
      ...prev,
      [name]: name.includes('Amount') ? parseFloat(value || '0') : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const predictionPromise = fetchPrediction(formData);

    props.setIsLoading({ status: true, message: 'Predicting token prices...' });
    await new Promise((resolve) => setTimeout(resolve, 4000));

    props.setIsLoading({ status: true, message: 'Predicting liquidation date...' });
    await new Promise((resolve) => setTimeout(resolve, 11000));

    // await new Promise((resolve) => setTimeout(resolve, 10000));
    // props.setIsLoading({ status: true, message: 'Generating model proof...' });

    // await new Promise((resolve) => setTimeout(resolve, 10000));
    // props.setIsLoading({ status: true, message: 'Verifying model proof...' });

    props.setIsLoading({ status: true, message: 'Updating smart contract...' });
    await new Promise((resolve) => setTimeout(resolve, 10000));

    try {
        const prediction = await predictionPromise;

        props.setPredictions((preds: Prediction[]) => [
            ...preds,
            prediction,
        ])
    } finally {
        props.setIsLoading({ status: false, message: '' });
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex items-center space-x-4">
            <div className="w-1/2">
                <label htmlFor="borrowToken" className="block text-sm font-medium text-gray-700">
                    Token to Borrow
                </label>
                <select
                    id="borrowToken"
                    name="borrowToken"
                    value={formData.borrowToken}
                    onChange={handleChange}
                    className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md text-black"
                >
                    {tokens.map((token) => (
                        <option key={token} value={token}>
                            {token}
                        </option>
                    ))}
                </select>
            </div>

            <div className="w-1/2">
                <label htmlFor="borrowAmount" className="block text-sm font-medium text-gray-700">
                    Amount to Borrow
                </label>
                <input
                    type="text"
                    id="borrowAmount"
                    name="borrowAmount"
                    value={formData.borrowAmount}
                    onChange={handleChange}
                    pattern='\d*'
                    inputMode='numeric'
                    className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md text-black"
                />
            </div>
        </div>

        <div className="flex items-center space-x-4">
            <div className="w-1/2">
                <label htmlFor="collateralToken" className="block text-sm font-medium text-gray-700">
                    Token to Use as Collateral
                </label>
                <select
                    id="collateralToken"
                    name="collateralToken"
                    value={formData.collateralToken}
                    onChange={handleChange}
                    className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md text-black"
                >
                    {tokens.map((token) => (
                        <option key={token} value={token}>
                        {token}
                        </option>
                    ))}
                </select>
            </div>

            <div className="w-1/2">
                <label htmlFor="collateralAmount" className="block text-sm font-medium text-gray-700">
                    Amount of Collateral
                </label>
                <input
                    type="text"
                    id="collateralAmount"
                    name="collateralAmount"
                    value={formData.collateralAmount}
                    min={1}
                    onChange={handleChange}
                    pattern='\d*'
                    inputMode='numeric'
                    className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md text-black"
                />
            </div>
        </div>

    <div className="flex justify-center mt-8">
        <button
            type="submit"
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
            Predict Liquidation Date
        </button>
    </div>
    </form>
  );
};

export default BorrowForm;
