# ==========================================================
# Neural Network Regression Project
# Traffic Volume Prediction
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------------------------------------
# Load the dataset
# ----------------------------------------------------------

data = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# ----------------------------------------------------------
# Select 500 random rows
# ----------------------------------------------------------

# Random sampling helps keep a representative subset
# of the original dataset

data = data.sample(
    n=500,
    random_state=42
)

# ----------------------------------------------------------
# Date preprocessing
# ----------------------------------------------------------

# Convert the date column into datetime format
data["date_time"] = pd.to_datetime(data["date_time"])

# Extract useful time-related features
data["hour"] = data["date_time"].dt.hour
data["day"] = data["date_time"].dt.day
data["month"] = data["date_time"].dt.month

# ----------------------------------------------------------
# Remove unnecessary columns
# ----------------------------------------------------------

data = data.drop(
    columns=[
        "holiday",
        "weather_description",
        "date_time"
    ]
)

# ----------------------------------------------------------
# Convert categorical variables into numeric values
# ----------------------------------------------------------

data = pd.get_dummies(
    data,
    columns=["weather_main"],
    drop_first=True
)

# ----------------------------------------------------------
# Define input and output variables
# ----------------------------------------------------------

X = data.drop("traffic_volume", axis=1)

y = data["traffic_volume"]

# ----------------------------------------------------------
# Split the dataset
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------------
# Feature scaling
# ----------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# Build the neural network
# ----------------------------------------------------------

model = MLPRegressor(

    hidden_layer_sizes=(64, 32),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    max_iter=500,

    random_state=42
)

# ----------------------------------------------------------
# Train the model
# ----------------------------------------------------------

model.fit(X_train, y_train)

# ----------------------------------------------------------
# Make predictions
# ----------------------------------------------------------

predictions = model.predict(X_test)

# ----------------------------------------------------------
# Model evaluation
# ----------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

# ----------------------------------------------------------
# Display results
# ----------------------------------------------------------

print("\nModel Performance")

print("MAE :", mae)

print("RMSE :", rmse)

print("R2 Score :", r2)
