"use client";

import { useEffect, useState } from "react";
import Predict, { Prediction } from "./predict_liquidation/displayPredictions";
import BorrowForm from "./predict_liquidation/predictForm";
import TokenPriceGraph from "./token_prices/TokenPriceGraph";

export default function Home() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [isLoading, setIsLoading] = useState<{ status: boolean; message: string }>({ status: false, message: '' });

  return (
    <div className="relative">
      {isLoading.status && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="loader mb-4"></div>
          <p className="text-white text-lg">{isLoading.message}</p>
        </div>
      )}
      <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
        <main className="flex flex-col sm:flex-row gap-8 row-start-2 items-center sm:items-start w-full">

          <div className="w-1/2">
            <TokenPriceGraph />
          </div>
          <div className="w-1/2">
            <div className="mb-8">
              <BorrowForm setPredictions={setPredictions} setIsLoading={setIsLoading} />
            </div>
            <Predict predictions={predictions} />
          </div>
        </main>
      </div>
    </div>
  );
}
