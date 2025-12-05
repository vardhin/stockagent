import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
import warnings
import torch

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# --- CONFIGURATION ---
STOCK_SYMBOL = "TATASTEEL.NS"
TEST_DAYS = 60           # Backtest over the last 60 trading days
HORIZON = 1              # We are testing "Next Day" prediction accuracy

print(f"--- STARTING BACKTEST FOR {STOCK_SYMBOL} ---")

# 1. GET DATA - Use more historical data for better learning
data = yf.download(STOCK_SYMBOL, period="5y", interval="1d")  # Increased from 2y to 5y

# Flatten multi-level columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.reset_index(inplace=True)

# Add technical features to help the model
data['Returns'] = data['Close'].pct_change()
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility'] = data['Returns'].rolling(window=20).std()

# Drop NaN values from rolling calculations
data = data.dropna()

# Format for NeuralForecast
ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

# 2. SETUP THE MODEL
# Optimized configuration for stability and accuracy
models = [NHITS(h=HORIZON,
                input_size=90,           # Increased from 60 for more context
                max_steps=1000,          # More training iterations
                scaler_type='robust',
                learning_rate=5e-4,      # Reduced learning rate for stability
                batch_size=64,           # Larger batch size for stability
                windows_batch_size=256,  # Reduced for memory efficiency
                dropout_prob_theta=0.3,  # Reduced dropout
                random_seed=42           # For reproducibility
)]

nf = NeuralForecast(models=models, freq='D')

# 3. RUN BACKTEST (The Heavy Lifting)
print(f"Simulating the last {TEST_DAYS} days... (This takes a moment)")
cv_df = nf.cross_validation(df=ai_df, n_windows=TEST_DAYS, step_size=1)

# 4. CALCULATE PERFORMANCE METRICS
cv_df.reset_index(inplace=True)
cv_df['prev_y'] = cv_df['y'].shift(1)
cv_df = cv_df.dropna()

# Calculate percentage changes
cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
cv_df['pred_pct']   = ((cv_df['NHITS'] - cv_df['prev_y']) / cv_df['prev_y']) * 100

# Directional accuracy
cv_df['correct_direction'] = np.sign(cv_df['actual_pct']) == np.sign(cv_df['pred_pct'])

# Error margin
cv_df['error_margin'] = abs(cv_df['actual_pct'] - cv_df['pred_pct'])

# 5. GENERATE REPORT CARD
accuracy = cv_df['correct_direction'].mean() * 100
avg_error = cv_df['error_margin'].mean()
median_error = cv_df['error_margin'].median()

print("\n" + "="*50)
print(f"BACKTEST RESULTS: {STOCK_SYMBOL} (Last {TEST_DAYS} Days)")
print("="*50)

print(f"1. DIRECTIONAL ACCURACY: {accuracy:.2f}%")
print(f"   (How often did it correctly say 'Up' or 'Down'?)")
print(f"   Note: > 55% is good. > 60% is excellent.")

print(f"\n2. ERROR METRICS:")
print(f"   - Average magnitude error: {avg_error:.2f}%")
print(f"   - Median magnitude error: {median_error:.2f}%")
print(f"   (On average, the AI is off by this much percent)")

print(f"\n3. SAMPLE PREDICTIONS (Last 5 Days):")
print(cv_df[['ds', 'actual_pct', 'pred_pct', 'correct_direction']].tail(5).to_string(index=False))

print(f"\n4. DETAILED METRICS:")
correct_up = ((cv_df['actual_pct'] > 0) & (cv_df['pred_pct'] > 0)).sum()
correct_down = ((cv_df['actual_pct'] < 0) & (cv_df['pred_pct'] < 0)).sum()
false_positive = ((cv_df['actual_pct'] < 0) & (cv_df['pred_pct'] > 0)).sum()
false_negative = ((cv_df['actual_pct'] > 0) & (cv_df['pred_pct'] < 0)).sum()

print(f"   - Correctly predicted UP: {correct_up}")
print(f"   - Correctly predicted DOWN: {correct_down}")
print(f"   - False positives (predicted UP, was DOWN): {false_positive}")
print(f"   - False negatives (predicted DOWN, was UP): {false_negative}")

# Calculate precision and recall
if (correct_up + false_positive) > 0:
    precision = correct_up / (correct_up + false_positive) * 100
    print(f"   - Precision (UP predictions): {precision:.2f}%")
if (correct_up + false_negative) > 0:
    recall = correct_up / (correct_up + false_negative) * 100
    print(f"   - Recall (UP predictions): {recall:.2f}%")

print(f"\n5. PREDICTION BIAS:")
print(f"   - Average predicted change: {cv_df['pred_pct'].mean():.2f}%")
print(f"   - Average actual change: {cv_df['actual_pct'].mean():.2f}%")
bias_diff = abs(cv_df['pred_pct'].mean() - cv_df['actual_pct'].mean())
print(f"   - Bias difference: {bias_diff:.2f}%")

# 6. VISUALIZATION - Enhanced
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

# Price comparison
ax1.plot(cv_df['ds'], cv_df['y'], label='Actual Price', color='black', linewidth=2)
ax1.plot(cv_df['ds'], cv_df['NHITS'], label='AI Prediction', color='cyan', linestyle='--', linewidth=2)
ax1.set_title(f"{STOCK_SYMBOL}: AI Backtest Performance", fontsize=14, fontweight='bold')
ax1.set_ylabel('Price', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Percentage change comparison
ax2.plot(cv_df['ds'], cv_df['actual_pct'], label='Actual % Change', color='black', linewidth=2)
ax2.plot(cv_df['ds'], cv_df['pred_pct'], label='Predicted % Change', color='cyan', linestyle='--', linewidth=2)
ax2.axhline(y=0, color='red', linestyle='-', alpha=0.3)
ax2.set_title("Daily % Change: Actual vs Predicted", fontsize=12)
ax2.set_ylabel("% Change", fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Error distribution
ax3.hist(cv_df['error_margin'], bins=30, color='orange', alpha=0.7, edgecolor='black')
ax3.axvline(cv_df['error_margin'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_error:.2f}%')
ax3.axvline(cv_df['error_margin'].median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}%')
ax3.set_title("Prediction Error Distribution", fontsize=12)
ax3.set_xlabel("Error Margin (%)", fontsize=12)
ax3.set_ylabel("Frequency", fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_backtest_results.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as '{STOCK_SYMBOL}_backtest_results.png'")
plt.show()