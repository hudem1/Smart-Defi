"use client";

export interface Prediction {
    borrowToken: string;
    borrowAmount: number;
    collateralToken: string;
    collateralAmount: number;
    predictedLiquidationDate: string;
}

interface PredictProps {
    predictions: Prediction[];
}

const Predict = ({ predictions }: PredictProps) => {
    return (
      <div className="p-4 bg-gray-50 rounded-md text-black w-full">
        <h2 className="text-xl font-bold mb-4">Prediction List</h2>
        {predictions.length > 0 ? (
          <ul className="space-y-4">
            {predictions.map((prediction, index) => (
              <li key={index} className="p-4 bg-white shadow-md rounded-md">
                <p><strong>Borrow Token:</strong> {prediction.borrowToken}</p>
                <p><strong>Borrow Amount:</strong> {prediction.borrowAmount}</p>
                <p><strong>Collateral Token:</strong> {prediction.collateralToken}</p>
                <p><strong>Collateral Amount:</strong> {prediction.collateralAmount}</p>
                <p><strong>Predicted Liquidation Date:</strong> {prediction.predictedLiquidationDate || 'N/A'}</p>
              </li>
            ))}
          </ul>
        ) : (
          <p>No predictions available.</p>
        )}
      </div>
    );
};

export default Predict;
