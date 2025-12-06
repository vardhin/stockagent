import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import warnings
import torch
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
STOCK_SYMBOL = "TATASTEEL.NS"
OPTIMAL_THRESHOLD = 0.30 
VAL_HORIZON = 5  # CHANGED: If you want Swing Trades, validate on 5-day holds, not 1-day.

# Define the indicators you want the model to actually see
EXOG_VARS = [
    'RSI', 'MACD', 'MACD_Hist', 'Vol_Ratio', 
    'BB_Width', 'Momentum_5', 'Log_Returns', 'Volatility_5'
]

OPTIMAL_CONFIG = {
    'windows': [100, 150, 200],
    'learning_rate': 1e-4, 
    'dropout': 0.10,        # Slightly reduced dropout for stability
    'batch_size': 32,
    'max_steps': 1500,     
    'scaler': 'standard',  
    'n_blocks': [3, 2, 1],
    
    # 👇 THE FIX: Make all units 512 to prevent shape mismatch
    'mlp_units': [[512, 512], [512, 512], [512, 512]], 
    
    'n_pool_kernel_size': [2, 2, 1],
    'n_freq_downsample': [8, 4, 1],
}

print("="*70)
print(f"🚀 MULTIVARIATE N-HITS SWING MODEL: {STOCK_SYMBOL}")
print(f"Features: {EXOG_VARS}")
print("="*70)

# 1. DATA PREPARATION
def prepare_data(symbol, period="10y"): # Use specific period to avoid ancient history
    print(f"📥 Downloading {symbol}...")
    data = yf.download(symbol, period=period, interval="1d", progress=False)
    
    # Handle MultiIndex columns if yfinance returns them
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data.columns = data.columns.get_level_values(0)
        except:
            pass # Sometimes structure varies
            
    data.reset_index(inplace=True)
    
    # --- Feature Engineering ---
    # Price Transforms
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility_5'] = data['Log_Returns'].rolling(5).std()
    
    # Volume
    data['Vol_MA20'] = data['Volume'].rolling(20).mean()
    data['Vol_Ratio'] = data['Volume'] / data['Vol_MA20']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Width (Volatility measure)
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Width'] = ((bb_mid + 2*bb_std) - (bb_mid - 2*bb_std)) / bb_mid
    
    # Momentum
    data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    
    # Cleanup
    data.dropna(inplace=True)
    return data

df = prepare_data(STOCK_SYMBOL)

# --- CRITICAL FIX: Add Exogenous Vars to ai_df ---
ai_df = pd.DataFrame({'unique_id': 'TATA', 'ds': df['Date'], 'y': df['Close']})
for col in EXOG_VARS:
    ai_df[col] = df[col].values # Copy indicators to input DF

print(f"✅ Prepared {len(ai_df)} days of data with {len(EXOG_VARS)} technical features.")

# 2. VALIDATION (Corrected for Swing Trading)
print("\n🔄 Validating logic (Simulating Production)...")
models_validate = []

for i, window in enumerate(OPTIMAL_CONFIG['windows']):
    model = NHITS(
        h=VAL_HORIZON,               # Validate on 5-day horizon (Swing)
        input_size=window,
        hist_exog_list=EXOG_VARS,    # <--- TELLING MODEL TO USE INDICATORS
        max_steps=1000,              # Faster for validation
        scaler_type=OPTIMAL_CONFIG['scaler'],
        n_blocks=OPTIMAL_CONFIG['n_blocks'],
        mlp_units=OPTIMAL_CONFIG['mlp_units'],
        n_pool_kernel_size=OPTIMAL_CONFIG['n_pool_kernel_size'],
        n_freq_downsample=OPTIMAL_CONFIG['n_freq_downsample'],
        random_seed=42 + i,
        alias=f"NHITS_{window}"
    )
    models_validate.append(model)

nf = NeuralForecast(models=models_validate, freq='D')

# Backtest on last 120 days (approx 6 months of trading)
cv_df = nf.cross_validation(df=ai_df, n_windows=20, step_size=5) 
# step_size=5 means we predict Mon-Fri, then move to next Mon. 
# This simulates "Weekly Swing" trading.

# Calculate Performance
cv_df.reset_index(inplace=True)

# We take the mean of the 3 models
model_cols = [c for c in cv_df.columns if 'NHITS' in c]
cv_df['Ensemble_Pred'] = cv_df[model_cols].mean(axis=1)

# Calculate 5-day Returns (Forecast vs Actual)
# Since cv_df is in 'long' format (stacked), we can just compare y vs pred
cv_df['Error_Pct'] = (cv_df['Ensemble_Pred'] - cv_df['y']) / cv_df['y']
mae = cv_df['Error_Pct'].abs().mean()

print(f"\n📊 Validation (MAE): {mae*100:.2f}% avg price error")

# 3. TRAINING FINAL PRODUCTION MODEL
print("\n" + "="*70)
print("🎓 Training FINAL PRODUCTION Ensemble...")
print("="*70)

final_models = []
for i, window in enumerate(OPTIMAL_CONFIG['windows']):
    print(f"   • Training Model {i+1}/3 (Window {window})...")
    model = NHITS(
        h=5, # Predict next 5 days
        input_size=window,
        hist_exog_list=EXOG_VARS, # <--- CRITICAL
        max_steps=2000,
        scaler_type=OPTIMAL_CONFIG['scaler'],
        learning_rate=OPTIMAL_CONFIG['learning_rate'],
        n_blocks=OPTIMAL_CONFIG['n_blocks'],
        mlp_units=OPTIMAL_CONFIG['mlp_units'],
        n_pool_kernel_size=OPTIMAL_CONFIG['n_pool_kernel_size'],
        n_freq_downsample=OPTIMAL_CONFIG['n_freq_downsample'],
        random_seed=42 + i
    )
    final_models.append(model)

nf_production = NeuralForecast(models=final_models, freq='D')
nf_production.fit(df=ai_df)

# 4. SAVE & PREDICT
save_dir = "./tata_swing_models"
os.makedirs(save_dir, exist_ok=True)
nf_production.save(path=save_dir, overwrite=True)

# Generate Forecast for NEXT 5 DAYS
future_exog = ai_df.tail(200) # N-HiTS needs history to predict future
forecast = nf_production.predict(future_exog)
forecast['Ensemble'] = forecast[[c for c in forecast.columns if 'NHITS' in c]].mean(axis=1)

print("\n🔮 PREDICTION FOR NEXT 5 DAYS:")
print(forecast[['ds', 'Ensemble']])
print(f"\n✅ Models saved to {save_dir}")