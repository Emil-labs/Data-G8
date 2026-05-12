"""
=============================================================================
TRAFFIC VOLUME PREDICTION & ANALYSIS PIPELINE
=============================================================================

This script provides a complete Machine Learning and Deep Learning pipeline
to analyze and predict Interstate Traffic Volume based on meteorological 
and temporal features.

PIPELINE CHAPTERS:
1. IMPORTS & CONFIGURATION
2. DATA LOADING & CLEANING
3. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
4. FEATURE ENGINEERING
5. DATA SPLITTING (CHRONOLOGICAL) & PREPROCESSING
6. MODEL DEFINITIONS (Classic ML & PyTorch CNN)
7. TIME SERIES CROSS-VALIDATION
8. FINAL EVALUATION ON HOLD-OUT SET (PRODUCTION SIMULATION)
=============================================================================
"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-Learn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# PyTorch (Deep Learning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuration
plt.style.use('ggplot')
os.makedirs('plots', exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# 2. DATA LOADING & CLEANING
# =============================================================================
print("--> CHAPTER 2: Loading and Cleaning Data...")

# Load dataset
file_path = r"c:\Users\Hamam Ayoub\Downloads\Metro_Interstate_Traffic_Volume.csv"
df = pd.read_csv(file_path)

# Convert string dates to datetime objects
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')

# Clean anomalous data (e.g., Temperature of 0 Kelvin is a sensor error)
df = df[df['temp'] > 0].copy()

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
# =============================================================================
print("--> CHAPTER 3: Generating EDA Plots (Traffic vs Weather)...")

# Extract basic time features for EDA
df['hour'] = df['date_time'].dt.hour
df['time_of_day'] = pd.cut(df['hour'], bins=[-1, 6, 12, 18, 24], 
                           labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'])

# Plot 3.1: Baseline - Traffic vs Hour of Day
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='traffic_volume', data=df, palette='viridis')
plt.title("Baseline: Traffic Volume Distribution by Hour of the Day")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Traffic Volume")
plt.savefig('plots/eda_traffic_by_hour.png')
plt.close()

# Plot 3.2: Traffic vs Weather Conditions (Rush Hours Only)
# To avoid the confounding effect of time, we isolate rush hours (7-9 AM and 16-18 PM)
rush_hour_mask = df['hour'].isin([7, 8, 9, 16, 17, 18])
df_rush = df[rush_hour_mask]

plt.figure(figsize=(14, 7))
sns.boxplot(x='weather_main', y='traffic_volume', data=df_rush, palette='Set2')
plt.title("Traffic Volume vs Weather Conditions (Rush Hours Only: 7-9h & 16-18h)")
plt.xlabel("Main Weather Category")
plt.ylabel("Traffic Volume")
plt.xticks(rotation=45)
plt.savefig('plots/eda_traffic_vs_weather_rushhour.png')
plt.close()

# Plot 3.3: Traffic vs Temperature (Colored by Time of Day)
plt.figure(figsize=(10, 8))
# Sample data to avoid overplotting
df_sample = df.sample(n=min(10000, len(df)), random_state=42)
sns.scatterplot(x='temp', y='traffic_volume', hue='time_of_day', data=df_sample, alpha=0.5, palette='coolwarm')
plt.title("Traffic Volume vs Temperature (Kelvin)")
plt.xlabel("Temperature (K)")
plt.ylabel("Traffic Volume")
plt.legend(title='Time of Day')
plt.savefig('plots/eda_traffic_vs_temperature.png')
plt.close()

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("--> CHAPTER 4: Feature Engineering...")

# Create temporal features
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month
df['is_holiday'] = (df['holiday'] != 'None').astype(int)

# Cyclical Encoding for Hour (to map 23:00 close to 01:00)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# =============================================================================
# 5. DATA SPLITTING (CHRONOLOGICAL) & PREPROCESSING
# =============================================================================
print("--> CHAPTER 5: Chronological Splitting and Preprocessing...")

# CRITICAL: Sort chronologically to prevent Data Leakage!
df = df.sort_values('date_time').reset_index(drop=True)

# Define Hold-out set (Future data representing production phase)
split_date = pd.to_datetime('2018-09-25')
future_mask = df['date_time'] >= split_date

df_train = df[~future_mask].copy()
df_future = df[future_mask].copy()

print(f"    [+] Training/Validation Data (before {split_date.date()}): {len(df_train)} rows.")
print(f"    [+] Hold-out Future Data (after {split_date.date()}): {len(df_future)} rows.")

# Feature Selection
categorical_features = ['weather_main']
numerical_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'day_of_week', 'month', 'hour_sin', 'hour_cos', 'is_holiday']
all_features = numerical_features + categorical_features

X_train_raw = df_train[all_features]
y_train_raw = df_train['traffic_volume'].values
X_future_raw = df_future[all_features]
y_future_raw = df_future['traffic_volume'].values

# Preprocessing Pipeline (Scaling & Encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

X_train_proc = preprocessor.fit_transform(X_train_raw)
X_future_proc = preprocessor.transform(X_future_raw)

# =============================================================================
# 6. MODEL DEFINITIONS (Classic ML & PyTorch CNN)
# =============================================================================
print("--> CHAPTER 6: Defining Models...")

# 6.1 Custom Gradient Descent Implementation
class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # Add bias term
        self.theta = np.zeros(X_b.shape[1])
        m = len(y)
        for i in range(self.iterations):
            predictions = X_b.dot(self.theta)
            errors = predictions - y
            gradients = (1/m) * X_b.T.dot(errors)
            self.theta -= self.learning_rate * gradients
            
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

# 6.2 1D Convolutional Neural Network (Sliding Window for Time Series)
class TS_CNN1D(nn.Module):
    def __init__(self, num_features, seq_length=24, out_channels=16):
        super(TS_CNN1D, self).__init__()
        # 1D Convolution to extract patterns over time (Sequence)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.flattened_dim = out_channels * (seq_length // 2)
        self.fc1 = nn.Linear(self.flattened_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, self.flattened_dim)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Helper function to create sequences for CNN
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

SEQUENCE_LENGTH = 24 # Predict using the last 24 hours

# Dictionary of Classic ML Models
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Descent': GradientDescentLinearRegression(learning_rate=0.1, iterations=500),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    'KNN': KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)
}

# =============================================================================
# 7. TIME SERIES CROSS-VALIDATION
# =============================================================================
print("--> CHAPTER 7: TimeSeriesSplit Cross Validation (10 Folds)...")

tscv = TimeSeriesSplit(n_splits=10)

metrics_history = {name: {'RMSE': [], 'R2': []} for name in models.keys()}
metrics_history['CNN 1D (Sequence)'] = {'RMSE': [], 'R2': []}

fold = 1
for train_index, test_index in tscv.split(X_train_proc):
    
    # Split data sequentially
    X_tr, X_te = X_train_proc[train_index], X_train_proc[test_index]
    y_tr, y_te = y_train_raw[train_index], y_train_raw[test_index]
    
    # 7.1 Train and Evaluate Classic Models
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        metrics_history[name]['RMSE'].append(np.sqrt(mean_squared_error(y_te, preds)))
        metrics_history[name]['R2'].append(r2_score(y_te, preds))
        
    # 7.2 Train and Evaluate CNN
    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, SEQUENCE_LENGTH)
    X_te_seq, y_te_seq = create_sequences(X_te, y_te, SEQUENCE_LENGTH)
    
    if len(X_te_seq) > 0:
        # Convert NumPy arrays to PyTorch Tensors
        X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32).transpose(1, 2)
        y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32).view(-1, 1)
        X_te_t = torch.tensor(X_te_seq, dtype=torch.float32).transpose(1, 2)
        
        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=128, shuffle=True)
        
        cnn = TS_CNN1D(num_features=X_train_proc.shape[1], seq_length=SEQUENCE_LENGTH)
        optimizer = optim.Adam(cnn.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        cnn.train()
        for epoch in range(10): # 10 Epochs per fold
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                loss = criterion(cnn(X_b), y_b)
                loss.backward()
                optimizer.step()
                
        cnn.eval()
        with torch.no_grad():
            preds_cnn = cnn(X_te_t).numpy().flatten()
            metrics_history['CNN 1D (Sequence)']['RMSE'].append(np.sqrt(mean_squared_error(y_te_seq, preds_cnn)))
            metrics_history['CNN 1D (Sequence)']['R2'].append(r2_score(y_te_seq, preds_cnn))
    else:
        metrics_history['CNN 1D (Sequence)']['RMSE'].append(np.nan)
        metrics_history['CNN 1D (Sequence)']['R2'].append(np.nan)
        
    print(f"    [+] Fold {fold}/10 Completed.")
    fold += 1

# =============================================================================
# 8. FINAL EVALUATION ON HOLD-OUT SET (PRODUCTION SIMULATION)
# =============================================================================
print("--> CHAPTER 8: Final Evaluation on Hold-out Set (Future Real)...")

future_metrics = {}
future_preds = {}

# 8.1 Final Training (Classic Models)
for name, model in models.items():
    # Train on ALL available training data
    model.fit(X_train_proc, y_train_raw)
    preds = model.predict(X_future_proc)
    
    # Calculate Metrics
    r2 = r2_score(y_future_raw, preds)
    rmse = np.sqrt(mean_squared_error(y_future_raw, preds))
    
    future_metrics[name] = {'R2': r2, 'RMSE': rmse}
    future_preds[name] = preds

# 8.2 Final Training (CNN)
X_full_seq, y_full_seq = create_sequences(X_train_proc, y_train_raw, SEQUENCE_LENGTH)
X_fut_seq, y_fut_seq = create_sequences(X_future_proc, y_future_raw, SEQUENCE_LENGTH)

X_full_t = torch.tensor(X_full_seq, dtype=torch.float32).transpose(1, 2)
y_full_t = torch.tensor(y_full_seq, dtype=torch.float32).view(-1, 1)
X_fut_t = torch.tensor(X_fut_seq, dtype=torch.float32).transpose(1, 2)

full_loader = DataLoader(TensorDataset(X_full_t, y_full_t), batch_size=128, shuffle=True)
final_cnn = TS_CNN1D(num_features=X_train_proc.shape[1], seq_length=SEQUENCE_LENGTH)
optimizer = optim.Adam(final_cnn.parameters(), lr=0.01)
criterion = nn.MSELoss()

final_cnn.train()
for epoch in range(15): # More epochs for final training
    for X_b, y_b in full_loader:
        optimizer.zero_grad()
        loss = criterion(final_cnn(X_b), y_b)
        loss.backward()
        optimizer.step()

final_cnn.eval()
with torch.no_grad():
    cnn_fut_preds = final_cnn(X_fut_t).numpy().flatten()
    r2 = r2_score(y_fut_seq, cnn_fut_preds)
    rmse = np.sqrt(mean_squared_error(y_fut_seq, cnn_fut_preds))
    
    future_metrics['CNN 1D (Sequence)'] = {'R2': r2, 'RMSE': rmse}
    future_preds['CNN 1D (Sequence)'] = cnn_fut_preds

# 8.3 Print Final Results
print("\n" + "="*40)
print("FINAL METRICS ON HOLD-OUT SET (FUTURE)")
print("="*40)
for name, res in future_metrics.items():
    print(f"{name.ljust(20)} -> R2: {res['R2']:.4f} | RMSE: {res['RMSE']:.2f}")

# 8.4 Generate Final Time Series Plots
dates_classic = df_future['date_time'].values
dates_cnn = df_future['date_time'].values[SEQUENCE_LENGTH:]

plt.figure(figsize=(18, 12))
for i, (name, preds) in enumerate(future_preds.items(), 1):
    plt.subplot(3, 2, i)
    if 'CNN' in name:
        plt.plot(dates_cnn, y_fut_seq, label='True Volume', color='black', linewidth=1.5)
        plt.plot(dates_cnn, preds, label='Predicted', color='orange', linestyle='--', alpha=0.8)
    else:
        plt.plot(dates_classic, y_future_raw, label='True Volume', color='black', linewidth=1.5)
        plt.plot(dates_classic, preds, label='Predicted', color='orange', linestyle='--', alpha=0.8)
    
    plt.title(f"{name} (Hold-out Set Predictions)")
    plt.xlabel("Date & Time")
    plt.ylabel("Traffic Volume")
    plt.xticks(rotation=45)
    plt.legend()

plt.tight_layout()
plt.savefig('plots/future_predictions_timeseries.png')
plt.close()

print("\n[SUCCESS] Pipeline execution finished. All plots saved to the 'plots' directory.")
