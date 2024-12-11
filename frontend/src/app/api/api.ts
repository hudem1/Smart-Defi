import { Prediction } from '../predict_liquidation/displayPredictions';
import { FormData } from '../predict_liquidation/predictForm';
import { TokensPredictedPrice } from '../token_prices/TokenPriceGraph';

const BACKEND_URL = "http://127.0.0.1:8000"

export const fetchLiquidationDatePrediction = async (formData: FormData): Promise<Prediction> => {
    try {
      const response = await fetch(`${BACKEND_URL}/predict_liquidation_date`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      return response.json();
    } catch (error) {
      console.error("API call error:", error);
      throw error;
    }
};

export const fetchTokensPrediction = async (): Promise<TokensPredictedPrice[]> => {
  try {
    const response = await fetch(`${BACKEND_URL}/predict_token_prices`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    console.error("API call error:", error);
    throw error;
  }
};
