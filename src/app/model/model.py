from giza.datasets import DatasetsLoader
import certifi
import os

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
from sklearn.metrics import mean_squared_error

from giza.zkcook import serialize_model

# Instantiate the DatasetsLoader object
loader = DatasetsLoader()

os.environ['SSL_CERT_FILE'] = certifi.where()

df = loader.load('aave-daily-deposits-borrowsv3')

df.head()

# Convert date to datetime and sort the data by date
df = df.with_columns(pl.col("date").str.strptime(pl.Date, fmt="%Y-%m-%d"))
df = df.sort("date")

# Create lag features for deposits_volume and borrows_volume
lags = 7
for lag in range(1, lags + 1):
    df = df.with_columns([
        pl.col("deposits_volume").shift(lag).alias(f"deposits_volume_lag_{lag}"),
        pl.col("borrows_volume").shift(lag).alias(f"borrows_volume_lag_{lag}")
    ])

# Drop missing values caused by lagging
df = df.drop_nulls()

# Encode categorical variables
df = df.with_columns([
    pl.col("symbol").cast(pl.Categorical).cast(pl.Int32).alias("symbol_encoded"),
    pl.col("contract_address").cast(pl.Categorical).cast(pl.Int32).alias("contract_address_encoded")
])

# Split the data into features and target variables
X = df.drop(["date", "deposits_volume", "borrows_volume"])
y = df["deposits_volume", "borrows_volume"]

# Convert Polars DataFrame to NumPy arrays for compatibility with scikit-learn
X_np = X.to_numpy()
y_np = y.to_numpy()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, shuffle=False)

# Initialize and train the XGBoost model with MultiOutputRegressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
multi_output_model = MultiOutputRegressor(xgb_model)

# Train the model
multi_output_model.fit(X_train, y_train)

# Predict on the test set
y_pred = multi_output_model.predict(X_test)

# Calculate Mean Squared Error for both outputs
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for deposits and borrows: {mse}")


serialize_model(multi_output_model, "xgb_lending_borrowing.json")