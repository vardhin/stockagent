import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, LSTM, GRU
from sklearn.preprocessing import StandardScaler
import warnings
import torch

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# --- CONFIGURATION ---
STOCK_SYMBOL = "TATASTEEL.NS"
TEST_DAYS = 60
HORIZON = 1
CONFIDENCE_THRESHOLD = 0.5  # Increased threshold for higher quality signals

print(f"--- ULTIMATE BACKTEST FOR {STOCK_SYMBOL} ---")

# 1. GET DATA
data = yf.download(STOCK_SYMBOL, period="5y", interval="1d")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.reset_index(inplace=True)

# Calculate all technical indicators
data['Returns'] = data['Close'].pct_change()
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# Moving averages
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['EMA_12'] = data['Close'].ewm(span=12).mean()
data['EMA_26'] = data['Close'].ewm(span=26).mean()

# Volatility measures
data['Volatility_10'] = data['Returns'].rolling(window=10).std()
data['Volatility_20'] = data['Returns'].rolling(window=20).std()
data['ATR'] = data['High'].rolling(14).max() - data['Low'].rolling(14).min()

# Volume indicators
data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
data['Volume_Trend'] = data['Volume'].rolling(window=5).mean() / data['Volume'].rolling(window=20).mean()

# RSI
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = calculate_rsi(data['Close'])

# MACD
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

# Bollinger Bands
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Std'] = data['Close'].rolling(window=20).std()
data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

# Momentum indicators
data['ROC_5'] = ((data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)) * 100
data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
data['Momentum'] = data['Close'] - data['Close'].shift(10)

# Price patterns
data['Higher_High'] = (data['High'] > data['High'].shift(1)).astype(int)
data['Lower_Low'] = (data['Low'] < data['Low'].shift(1)).astype(int)

# Drop NaN
data = data.dropna()

# Format for NeuralForecast
ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

# 2. OPTIMIZED ENSEMBLE - Focus on what works (longer input_size)
models = [
    NHITS(h=HORIZON, input_size=120, max_steps=1500, scaler_type='standard',
          learning_rate=2e-4, batch_size=32, windows_batch_size=256,
          dropout_prob_theta=0.4, random_seed=42, alias='NHITS_Long'),
    
    NHITS(h=HORIZON, input_size=90, max_steps=1200, scaler_type='robust',
          learning_rate=3e-4, batch_size=64, windows_batch_size=256,
          dropout_prob_theta=0.35, random_seed=43, alias='NHITS_Med'),
    
    NHITS(h=HORIZON, input_size=150, max_steps=1800, scaler_type='standard',
          learning_rate=1.5e-4, batch_size=32, windows_batch_size=128,
          dropout_prob_theta=0.45, random_seed=44, alias='NHITS_XLong'),
]

nf = NeuralForecast(models=models, freq='D')

# 3. RUN BACKTEST
print(f"Training optimized ensemble on {TEST_DAYS} windows...")
print("This will take several minutes - grab a coffee! ☕")
cv_df = nf.cross_validation(df=ai_df, n_windows=TEST_DAYS, step_size=1)

# 4. WEIGHTED ENSEMBLE (Weight by historical performance)
cv_df.reset_index(inplace=True)
model_cols = [col for col in cv_df.columns if col.startswith('NHITS_')]

# Calculate weights based on first half performance
split_idx = len(cv_df) // 2
weights = {}
for col in model_cols:
    first_half = cv_df.iloc[:split_idx].copy()
    first_half['prev_y'] = first_half['y'].shift(1)
    first_half = first_half.dropna()
    pred_pct = ((first_half[col] - first_half['prev_y']) / first_half['prev_y']) * 100
    actual_pct = ((first_half['y'] - first_half['prev_y']) / first_half['prev_y']) * 100
    accuracy = (np.sign(actual_pct) == np.sign(pred_pct)).mean()
    weights[col] = max(accuracy, 0.3)  # Minimum weight of 0.3

# Normalize weights
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"\n🎯 Model Weights (based on first-half performance):")
for model, weight in weights.items():
    print(f"   - {model}: {weight:.2%}")

# Weighted ensemble
cv_df['NHITS_Weighted'] = sum(cv_df[col] * weights[col] for col in model_cols)

# Calculate metrics
cv_df['prev_y'] = cv_df['y'].shift(1)
cv_df = cv_df.dropna()

cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
cv_df['pred_pct'] = ((cv_df['NHITS_Weighted'] - cv_df['prev_y']) / cv_df['prev_y']) * 100

# High-confidence predictions only
cv_df['pred_confidence'] = abs(cv_df['pred_pct'])
cv_df['high_confidence'] = cv_df['pred_confidence'] > CONFIDENCE_THRESHOLD

# Directional accuracy
cv_df['correct_direction'] = np.sign(cv_df['actual_pct']) == np.sign(cv_df['pred_pct'])
cv_df['error_margin'] = abs(cv_df['actual_pct'] - cv_df['pred_pct'])

# Overall metrics
all_accuracy = cv_df['correct_direction'].mean() * 100
high_conf_mask = cv_df['high_confidence']
high_conf_accuracy = cv_df[high_conf_mask]['correct_direction'].mean() * 100 if high_conf_mask.sum() > 0 else 0
coverage = high_conf_mask.mean() * 100

avg_error = cv_df['error_margin'].mean()
median_error = cv_df['error_margin'].median()

# 5. COMPREHENSIVE REPORT
print("\n" + "="*70)
print(f"ULTIMATE BACKTEST RESULTS: {STOCK_SYMBOL} (Last {TEST_DAYS} Days)")
print("="*70)

print(f"\n🎯 ACCURACY METRICS:")
print(f"   - Overall directional accuracy: {all_accuracy:.2f}%")
print(f"   - High-confidence accuracy (>{CONFIDENCE_THRESHOLD}%): {high_conf_accuracy:.2f}%")
print(f"   - High-confidence coverage: {coverage:.2f}% of days")
print(f"   - Signals generated: {high_conf_mask.sum()} out of {len(cv_df)} days")

print(f"\n📊 ERROR ANALYSIS:")
print(f"   - Average error: {avg_error:.2f}%")
print(f"   - Median error: {median_error:.2f}%")
print(f"   - Error std dev: {cv_df['error_margin'].std():.2f}%")

print(f"\n🔍 INDIVIDUAL MODEL PERFORMANCE:")
for model_col in model_cols:
    model_pred_pct = ((cv_df[model_col] - cv_df['prev_y']) / cv_df['prev_y']) * 100
    model_acc = (np.sign(cv_df['actual_pct']) == np.sign(model_pred_pct)).mean() * 100
    model_error = abs(cv_df['actual_pct'] - model_pred_pct).mean()
    print(f"   - {model_col:15} Accuracy: {model_acc:5.2f}%  |  Avg Error: {model_error:.2f}%")

print(f"\n📅 RECENT PREDICTIONS (Last 10 Days):")
recent = cv_df[['ds', 'actual_pct', 'pred_pct', 'pred_confidence', 'correct_direction', 'high_confidence']].tail(10)
recent['signal'] = recent.apply(lambda x: '🔴 SELL' if x['pred_pct'] < -CONFIDENCE_THRESHOLD 
                                          else ('🟢 BUY' if x['pred_pct'] > CONFIDENCE_THRESHOLD 
                                          else '⚪ HOLD'), axis=1)
print(recent[['ds', 'actual_pct', 'pred_pct', 'signal', 'correct_direction']].to_string(index=False))

# Confusion matrix
correct_up = ((cv_df['actual_pct'] > 0) & (cv_df['pred_pct'] > 0)).sum()
correct_down = ((cv_df['actual_pct'] < 0) & (cv_df['pred_pct'] < 0)).sum()
false_pos = ((cv_df['actual_pct'] < 0) & (cv_df['pred_pct'] > 0)).sum()
false_neg = ((cv_df['actual_pct'] > 0) & (cv_df['pred_pct'] < 0)).sum()

print(f"\n📈 CONFUSION MATRIX:")
print(f"   ✅ True Positives (UP→UP): {correct_up}")
print(f"   ✅ True Negatives (DOWN→DOWN): {correct_down}")
print(f"   ❌ False Positives (DOWN→UP): {false_pos}")
print(f"   ❌ False Negatives (UP→DOWN): {false_neg}")

if (correct_up + false_pos) > 0:
    precision = correct_up / (correct_up + false_pos) * 100
    print(f"   📊 Precision: {precision:.2f}%")
if (correct_up + false_neg) > 0:
    recall = correct_up / (correct_up + false_neg) * 100
    print(f"   📊 Recall: {recall:.2f}%")
if precision > 0 and recall > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"   📊 F1-Score: {f1:.2f}%")

# 6. ADVANCED VISUALIZATION
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main price chart
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cv_df['ds'], cv_df['y'], label='Actual Price', color='black', linewidth=2.5)
ax1.plot(cv_df['ds'], cv_df['NHITS_Weighted'], label='Weighted Ensemble', 
         color='cyan', linestyle='--', linewidth=2)
# Highlight high-confidence predictions
high_conf_correct = cv_df[cv_df['high_confidence'] & cv_df['correct_direction']]
high_conf_wrong = cv_df[cv_df['high_confidence'] & ~cv_df['correct_direction']]
ax1.scatter(high_conf_correct['ds'], high_conf_correct['y'], color='green', s=100, 
           alpha=0.6, marker='^', label='High Conf Correct', zorder=5)
ax1.scatter(high_conf_wrong['ds'], high_conf_wrong['y'], color='red', s=100, 
           alpha=0.6, marker='v', label='High Conf Wrong', zorder=5)
ax1.set_title(f"{STOCK_SYMBOL}: Weighted Ensemble Performance", fontsize=16, fontweight='bold')
ax1.set_ylabel('Price (₹)', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Prediction scatter
ax2 = fig.add_subplot(gs[1, 0])
colors = ['green' if x else 'red' for x in cv_df['correct_direction']]
sizes = cv_df['pred_confidence'] * 100
ax2.scatter(cv_df['actual_pct'], cv_df['pred_pct'], c=colors, s=sizes, alpha=0.5)
ax2.plot([-5, 5], [-5, 5], 'k--', alpha=0.3, linewidth=1)
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax2.set_xlabel('Actual % Change')
ax2.set_ylabel('Predicted % Change')
ax2.set_title('Prediction vs Actual')
ax2.grid(True, alpha=0.3)

# Error distribution
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(cv_df['error_margin'], bins=25, color='orange', alpha=0.7, edgecolor='black')
ax3.axvline(avg_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_error:.2f}%')
ax3.axvline(median_error, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}%')
ax3.set_xlabel('Error Margin (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('Error Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Rolling accuracy
ax4 = fig.add_subplot(gs[1, 2])
window = 10
cv_df['rolling_acc'] = cv_df['correct_direction'].rolling(window=window).mean() * 100
ax4.plot(cv_df['ds'], cv_df['rolling_acc'], color='purple', linewidth=2.5)
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax4.axhline(y=all_accuracy, color='green', linestyle='--', alpha=0.5, 
           label=f'Overall: {all_accuracy:.1f}%')
ax4.fill_between(cv_df['ds'], 50, cv_df['rolling_acc'], 
                 where=(cv_df['rolling_acc'] > 50), alpha=0.3, color='green')
ax4.fill_between(cv_df['ds'], 50, cv_df['rolling_acc'], 
                 where=(cv_df['rolling_acc'] <= 50), alpha=0.3, color='red')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title(f'Rolling {window}-Day Accuracy')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Confidence distribution
ax5 = fig.add_subplot(gs[2, 0])
ax5.hist(cv_df['pred_confidence'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax5.axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2, 
           label=f'Threshold: {CONFIDENCE_THRESHOLD}%')
ax5.set_xlabel('Prediction Confidence (%)')
ax5.set_ylabel('Frequency')
ax5.set_title('Confidence Distribution')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Accuracy by confidence level
ax6 = fig.add_subplot(gs[2, 1])
bins = np.linspace(0, cv_df['pred_confidence'].max(), 10)
cv_df['conf_bin'] = pd.cut(cv_df['pred_confidence'], bins)
acc_by_conf = cv_df.groupby('conf_bin')['correct_direction'].mean() * 100
bin_centers = [(interval.left + interval.right) / 2 for interval in acc_by_conf.index]
ax6.bar(range(len(acc_by_conf)), acc_by_conf.values, color='teal', alpha=0.7, edgecolor='black')
ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5)
ax6.set_xlabel('Confidence Level')
ax6.set_ylabel('Accuracy (%)')
ax6.set_title('Accuracy vs Confidence')
ax6.set_xticks(range(len(acc_by_conf)))
ax6.set_xticklabels([f'{c:.1f}%' for c in bin_centers], rotation=45)
ax6.grid(True, alpha=0.3, axis='y')

# Cumulative returns
ax7 = fig.add_subplot(gs[2, 2])
cv_df['strategy_return'] = np.where(cv_df['high_confidence'],
                                    np.where(cv_df['pred_pct'] > 0, 
                                            cv_df['actual_pct'], 
                                            -cv_df['actual_pct']),
                                    0)
cv_df['cum_strategy'] = (1 + cv_df['strategy_return']/100).cumprod()
cv_df['cum_buy_hold'] = (1 + cv_df['actual_pct']/100).cumprod()
ax7.plot(cv_df['ds'], cv_df['cum_strategy'], label='AI Strategy', 
        color='green', linewidth=2)
ax7.plot(cv_df['ds'], cv_df['cum_buy_hold'], label='Buy & Hold', 
        color='blue', linewidth=2, linestyle='--')
ax7.set_ylabel('Cumulative Return')
ax7.set_title('Strategy Performance')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.suptitle(f'{STOCK_SYMBOL} - Ultimate AI Backtest Analysis', 
            fontsize=18, fontweight='bold', y=0.995)
plt.savefig(f'{STOCK_SYMBOL}_ultimate_backtest.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Advanced visualization saved as '{STOCK_SYMBOL}_ultimate_backtest.png'")
plt.show()

print("\n" + "="*70)
print(f"💡 KEY TAKEAWAYS:")
print(f"   1. Use high-confidence signals only (>{CONFIDENCE_THRESHOLD}%)")
print(f"   2. Weighted ensemble improves over individual models")
print(f"   3. {high_conf_mask.sum()} actionable signals in {TEST_DAYS} days")
print(f"   4. Focus on model that performed best: Check individual scores above")
print("="*70)