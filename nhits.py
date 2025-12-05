import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
STOCK_SYMBOL = "TATAMOTORS.NS"
TEST_DAYS = 60           # Backtest over the last 60 trading days
HORIZON = 1              # We are testing "Next Day" prediction accuracy

print(f"--- STARTING BACKTEST FOR {STOCK_SYMBOL} ---")

# 1. GET DATA
data = yf.download(STOCK_SYMBOL, period="2y", interval="1d")

# Flatten multi-level columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.reset_index(inplace=True)

# Format for NeuralForecast
ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

# 2. SETUP THE MODEL
# We use N-HiTS again, but configured for backtesting
models = [NHITS(h=HORIZON,
                input_size=30,
                max_steps=200,      # Higher steps = better accuracy but slower
                scaler_type='robust')]

nf = NeuralForecast(models=models, freq='D')

# 3. RUN BACKTEST (The Heavy Lifting)
# This effectively loops through the last 60 days, retraining/predicting each time
print(f"Simulating the last {TEST_DAYS} days... (This takes a moment)")
cv_df = nf.cross_validation(df=ai_df, n_windows=TEST_DAYS, step_size=1)

# 4. CALCULATE PERFORMANCE METRICS
# Merge with previous day's data to calculate % change
cv_df.reset_index(inplace=True)
cv_df['prev_y'] = cv_df['y'].shift(1) # We need yesterday's price to calculate % change
cv_df = cv_df.dropna() # Drop the first row which has no 'previous'

# Calculate Actual % Change vs Predicted % Change
cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
cv_df['pred_pct']   = ((cv_df['NHITS'] - cv_df['prev_y']) / cv_df['prev_y']) * 100

# Directional Accuracy: Did AI get the sign right? (Up vs Down)
# We check if (Actual > 0 and Pred > 0) OR (Actual < 0 and Pred < 0)
cv_df['correct_direction'] = np.sign(cv_df['actual_pct']) == np.sign(cv_df['pred_pct'])

# Error Margin: How far off was the percentage?
cv_df['error_margin'] = abs(cv_df['actual_pct'] - cv_df['pred_pct'])

# 5. GENERATE REPORT CARD
accuracy = cv_df['correct_direction'].mean() * 100
avg_error = cv_df['error_margin'].mean()

print("\n" + "="*50)
print(f"BACKTEST RESULTS: {STOCK_SYMBOL} (Last {TEST_DAYS} Days)")
print("="*50)

print(f"1. DIRECTIONAL ACCURACY: {accuracy:.2f}%")
print(f"   (How often did it correctly say 'Up' or 'Down'?)")
print(f"   Note: > 55% is good. > 60% is excellent.")

print(f"\n2. AVERAGE MAGNITUDE ERROR: {avg_error:.2f}%")
print(f"   (On average, the AI is off by this much percent)")

print(f"\n3. SAMPLE PREDICTIONS (Last 5 Days):")
print(cv_df[['ds', 'actual_pct', 'pred_pct', 'correct_direction']].tail(5).to_string(index=False))

# 6. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(cv_df['ds'], cv_df['y'], label='Actual Price', color='black')
plt.plot(cv_df['ds'], cv_df['NHITS'], label='AI Prediction', color='cyan', linestyle='--')
plt.title(f"{STOCK_SYMBOL}: AI Backtest Performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()