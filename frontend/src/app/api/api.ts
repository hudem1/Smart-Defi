import { Prediction } from '../predict_liquidation/displayPredictions';
import { FormData } from '../predict_liquidation/predictForm';

const BACKEND_URL = "http://127.0.0.1:8000"

export const fetchPrediction = async (formData: FormData): Promise<Prediction> => {
    try {
      const response = await fetch(`${BACKEND_URL}/predict_liquidation`, {
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

