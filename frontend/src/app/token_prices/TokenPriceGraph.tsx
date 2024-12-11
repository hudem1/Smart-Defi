import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchTokensPrediction } from '../api/api';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export interface TokensPredictedPrice {
    date: string;
    price_DAI: number;
    price_WETH: number;
    price_WBTC: number;
    price_USDC: number;
    price_USDT: number;
}

const TokenPriceGraph = () => {
    const [tokenPrices, setTokenPrices] = useState<TokensPredictedPrice[]>([]);

    useEffect(() => {
        const fetchData = async () => {
            const tokens = await fetchTokensPrediction();
            setTokenPrices(tokens);
        };

        fetchData();
    }, []);

    const createOptions = useCallback((graphTitle: string) => ({
        responsive: true,
        interaction: {
            mode: 'index' as const,
            intersect: false,
        },
        stacked: false,
        plugins: {
            title: {
                display: true,
                text: graphTitle,
            },
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Date',
                },
            },
            y: {
                type: 'linear' as const,
                display: true,
                position: 'left' as const,
                title: {
                    display: true,
                    text: 'Price in USD ($)',
                },
            },
        },
    }), []);

    const stablecoinsOptions = createOptions('Stablecoins: DAI, USDC, USDT');
    const wethOptions = createOptions('WETH');
    const wbtcOptions = createOptions('WBTC');

    const stablecoinsData = useMemo(() => ({
        labels: tokenPrices.map(pred => pred.date),
        datasets: [
            {
                label: 'DAI',
                data: tokenPrices.map(pred => pred.price_DAI),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                yAxisID: 'y',
            },
            {
                label: 'USDC',
                data: tokenPrices.map(pred => pred.price_USDC),
                borderColor: 'rgb(53, 162, 235)',
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
                yAxisID: 'y',
            },
            {
                label: 'USDT',
                data: tokenPrices.map(pred => pred.price_USDT),
                borderColor: 'rgb(53, 235, 53)',
                backgroundColor: 'rgba(53, 235, 53, 0.5)',
                yAxisID: 'y',
            },
        ],
    }), [tokenPrices]);


    const wethData = useMemo(() => ( {
        labels: tokenPrices.map(pred => pred.date),
        datasets: [
            {
                label: 'WETH',
                data: tokenPrices.map(pred => pred.price_WETH),
                borderColor: 'rgb(255, 0, 255)',
                backgroundColor: 'rgba(255, 0, 255, 0.5)',
                yAxisID: 'y',
            },
        ],
    }), [tokenPrices]);


    const wbtcData = useMemo(() => ( {
        labels: tokenPrices.map(pred => pred.date),
        datasets: [
            {
                label: 'WBTC',
                data: tokenPrices.map(pred => pred.price_WBTC),
                borderColor: 'rgb(255, 165, 0)',
                backgroundColor: 'rgba(255, 165, 0, 0.5)',
                yAxisID: 'y',
            },
        ],
    }), [tokenPrices]);

  return (
    <>
        <div className="mb-4 p-6 bg-white shadow-md rounded">
            <Line data={stablecoinsData} options={stablecoinsOptions} />
        </div>
        <div className="mb-4 p-6 bg-white shadow-md rounded">
            <Line data={wethData} options={wethOptions} />
        </div>
        <div className="p-6 bg-white shadow-md rounded">
            <Line data={wbtcData} options={wbtcOptions} />
        </div>
    </>
  );
};

export default TokenPriceGraph;
