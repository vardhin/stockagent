import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
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
TEST_DAYS = 60
HORIZON = 1

print(f"--- STARTING ENHANCED BACKTEST FOR {STOCK_SYMBOL} ---")

# 1. GET DATA with more features
data = yf.download(STOCK_SYMBOL, period="5y", interval="1d")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.reset_index(inplace=True)

# Enhanced technical features
data['Returns'] = data['Close'].pct_change()
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['EMA_12'] = data['Close'].ewm(span=12).mean()
data['EMA_26'] = data['Close'].ewm(span=26).mean()
data['Volatility'] = data['Returns'].rolling(window=20).std()
data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

# RSI calculation
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = calculate_rsi(data['Close'])

# MACD
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9).mean()

# Bollinger Bands
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Std'] = data['Close'].rolling(window=20).std()
data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

# Price momentum
data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
data['Momentum_10'] = data['Close'] - data['Close'].shift(10)

# Drop NaN
data = data.dropna()

# Format for NeuralForecast
ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

# 2. ENSEMBLE OF MODELS with different configurations
models = [
    NHITS(h=HORIZON, input_size=60, max_steps=800, scaler_type='robust',
          learning_rate=1e-3, batch_size=32, windows_batch_size=128,
          dropout_prob_theta=0.2, random_seed=42, alias='NHITS_1'),
    
    NHITS(h=HORIZON, input_size=90, max_steps=1000, scaler_type='robust',
          learning_rate=5e-4, batch_size=64, windows_batch_size=256,
          dropout_prob_theta=0.3, random_seed=43, alias='NHITS_2'),
    
    NHITS(h=HORIZON, input_size=120, max_steps=1200, scaler_type='standard',
          learning_rate=3e-4, batch_size=32, windows_batch_size=128,
          dropout_prob_theta=0.4, random_seed=44, alias='NHITS_3'),
]

nf = NeuralForecast(models=models, freq='D')

# 3. RUN BACKTEST
print(f"Training ensemble of {len(models)} models on {TEST_DAYS} windows...")
cv_df = nf.cross_validation(df=ai_df, n_windows=TEST_DAYS, step_size=1)

# 4. ENSEMBLE PREDICTIONS (Average of all models)
cv_df.reset_index(inplace=True)
model_cols = [col for col in cv_df.columns if col.startswith('NHITS_')]
cv_df['NHITS_Ensemble'] = cv_df[model_cols].mean(axis=1)

# Calculate metrics for ensemble
cv_df['prev_y'] = cv_df['y'].shift(1)
cv_df = cv_df.dropna()

cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
cv_df['pred_pct'] = ((cv_df['NHITS_Ensemble'] - cv_df['prev_y']) / cv_df['prev_y']) * 100

# Directional accuracy with confidence threshold
threshold = 0.3  # Only predict when confidence is high
cv_df['pred_direction'] = np.where(abs(cv_df['pred_pct']) > threshold,
                                    np.sign(cv_df['pred_pct']), 0)
cv_df['actual_direction'] = np.sign(cv_df['actual_pct'])

# Calculate accuracy only for predictions made
mask = cv_df['pred_direction'] != 0
if mask.sum() > 0:
    confident_accuracy = (cv_df[mask]['pred_direction'] == cv_df[mask]['actual_direction']).mean() * 100
    coverage = mask.mean() * 100
else:
    confident_accuracy = 0
    coverage = 0

# Original metrics
cv_df['correct_direction'] = np.sign(cv_df['actual_pct']) == np.sign(cv_df['pred_pct'])
cv_df['error_margin'] = abs(cv_df['actual_pct'] - cv_df['pred_pct'])

accuracy = cv_df['correct_direction'].mean() * 100
avg_error = cv_df['error_margin'].mean()
median_error = cv_df['error_margin'].median()

# 5. DETAILED REPORT
print("\n" + "="*60)
print(f"ENHANCED BACKTEST RESULTS: {STOCK_SYMBOL} (Last {TEST_DAYS} Days)")
print("="*60)

print(f"\n📊 ENSEMBLE PERFORMANCE:")
print(f"   - Models in ensemble: {len(models)}")
print(f"   - Directional accuracy (all predictions): {accuracy:.2f}%")
print(f"   - Confident predictions accuracy (>{threshold}%): {confident_accuracy:.2f}%")
print(f"   - Coverage (% of days predicted): {coverage:.2f}%")

print(f"\n📈 ERROR METRICS:")
print(f"   - Average magnitude error: {avg_error:.2f}%")
print(f"   - Median magnitude error: {median_error:.2f}%")

print(f"\n🎯 INDIVIDUAL MODEL PERFORMANCE:")
for model_col in model_cols:
    model_pred_pct = ((cv_df[model_col] - cv_df['prev_y']) / cv_df['prev_y']) * 100
    model_accuracy = (np.sign(cv_df['actual_pct']) == np.sign(model_pred_pct)).mean() * 100
    print(f"   - {model_col}: {model_accuracy:.2f}%")

print(f"\n📅 SAMPLE PREDICTIONS (Last 5 Days):")
display_df = cv_df[['ds', 'actual_pct', 'pred_pct', 'correct_direction']].tail(5).copy()
display_df['confidence'] = abs(display_df['pred_pct'])
print(display_df.to_string(index=False))

correct_up = ((cv_df['actual_pct'] > 0) & (cv_df['pred_pct'] > 0)).sum()
correct_down = ((cv_df['actual_pct'] < 0) & (cv_df['pred_pct'] < 0)).sum()
false_positive = ((cv_df['actual_pct'] < 0) & (cv_df['pred_pct'] > 0)).sum()
false_negative = ((cv_df['actual_pct'] > 0) & (cv_df['pred_pct'] < 0)).sum()

print(f"\n📊 CONFUSION MATRIX:")
print(f"   - Correctly predicted UP: {correct_up}")
print(f"   - Correctly predicted DOWN: {correct_down}")
print(f"   - False positives (predicted UP, was DOWN): {false_positive}")
print(f"   - False negatives (predicted DOWN, was UP): {false_negative}")

if (correct_up + false_positive) > 0:
    precision = correct_up / (correct_up + false_positive) * 100
    print(f"   - Precision (UP predictions): {precision:.2f}%")
if (correct_up + false_negative) > 0:
    recall = correct_up / (correct_up + false_negative) * 100
    print(f"   - Recall (UP predictions): {recall:.2f}%")

print(f"\n⚖️ PREDICTION BIAS:")
print(f"   - Average predicted change: {cv_df['pred_pct'].mean():.2f}%")
print(f"   - Average actual change: {cv_df['actual_pct'].mean():.2f}%")

# 6. ENHANCED VISUALIZATION
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Price comparison
ax1.plot(cv_df['ds'], cv_df['y'], label='Actual', color='black', linewidth=2)
ax1.plot(cv_df['ds'], cv_df['NHITS_Ensemble'], label='Ensemble Prediction', 
         color='cyan', linestyle='--', linewidth=2)
ax1.set_title(f"{STOCK_SYMBOL}: Ensemble Prediction", fontsize=14, fontweight='bold')
ax1.set_ylabel('Price', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Percentage change
correct_mask = cv_df['correct_direction']
ax2.scatter(cv_df[correct_mask]['ds'], cv_df[correct_mask]['actual_pct'], 
           color='green', alpha=0.6, label='Correct', s=50)
ax2.scatter(cv_df[~correct_mask]['ds'], cv_df[~correct_mask]['actual_pct'], 
           color='red', alpha=0.6, label='Wrong', s=50)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_title("Prediction Accuracy by Day", fontsize=12)
ax2.set_ylabel("Actual % Change", fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Error distribution
ax3.hist(cv_df['error_margin'], bins=30, color='orange', alpha=0.7, edgecolor='black')
ax3.axvline(avg_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_error:.2f}%')
ax3.axvline(median_error, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}%')
ax3.set_title("Error Distribution", fontsize=12)
ax3.set_xlabel("Error Margin (%)", fontsize=12)
ax3.set_ylabel("Frequency", fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Rolling accuracy
window = 10
cv_df['rolling_accuracy'] = cv_df['correct_direction'].rolling(window=window).mean() * 100
ax4.plot(cv_df['ds'], cv_df['rolling_accuracy'], color='purple', linewidth=2)
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
ax4.axhline(y=accuracy, color='green', linestyle='--', alpha=0.5, label=f'Overall: {accuracy:.2f}%')
ax4.set_title(f"Rolling {window}-Day Accuracy", fontsize=12)
ax4.set_ylabel("Accuracy (%)", fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_enhanced_backtest.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved as '{STOCK_SYMBOL}_enhanced_backtest.png'")
plt.show()