import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import warnings
import torch
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# --- OPTIMAL CONFIGURATION (from hyperparameter search) ---
STOCK_SYMBOL = "TATASTEEL.NS"
OPTIMAL_CONFIG = {
    'windows': [150, 200],
    'learning_rate': 1e-4,
    'dropout': 0.2,
    'batch_size': 32,
    'max_steps': 1500,
    'scaler': 'standard'
}
CONFIDENCE_THRESHOLD = 0.5

print("="*70)
print("🚀 PRODUCTION-READY STOCK PREDICTOR")
print("="*70)
print(f"Stock: {STOCK_SYMBOL}")
print(f"Configuration: {OPTIMAL_CONFIG}")
print("="*70)

# 1. DATA PREPARATION
def prepare_data(symbol, period="max"):
    """Download and prepare stock data with technical indicators"""
    data = yf.download(symbol, period=period, interval="1d")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    
    # Technical indicators
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    data['Volatility_10'] = data['Returns'].rolling(window=10).std()
    data['Volatility_20'] = data['Returns'].rolling(window=20).std()
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    
    data = data.dropna()
    
    return data

print("\n📥 Loading data...")
data = prepare_data(STOCK_SYMBOL)
print(f"✅ Loaded {len(data)} days of data")
print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")

ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

# 2. BUILD OPTIMIZED ENSEMBLE
print("\n🏗️  Building optimized ensemble...")
models = []
for i, window in enumerate(OPTIMAL_CONFIG['windows']):
    model = NHITS(
        h=1,
        input_size=window,
        max_steps=OPTIMAL_CONFIG['max_steps'],
        scaler_type=OPTIMAL_CONFIG['scaler'],
        learning_rate=OPTIMAL_CONFIG['learning_rate'],
        batch_size=OPTIMAL_CONFIG['batch_size'],
        windows_batch_size=128,
        dropout_prob_theta=OPTIMAL_CONFIG['dropout'],
        random_seed=42 + i,
        alias=f"Optimized_{window}"
    )
    models.append(model)
    print(f"   ✓ Model {i+1}: Window={window} days")

nf = NeuralForecast(models=models, freq='D')

# 3. COMPREHENSIVE BACKTEST
print("\n🔄 Running comprehensive backtest (60 days)...")
TEST_DAYS = 60
cv_df = nf.cross_validation(df=ai_df, n_windows=TEST_DAYS, step_size=1)

# Calculate metrics
cv_df.reset_index(inplace=True)
cv_df['prev_y'] = cv_df['y'].shift(1)
cv_df = cv_df.dropna()
cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100

# Ensemble prediction
model_cols = [col for col in cv_df.columns if col.startswith('Optimized_')]
cv_df['Ensemble'] = cv_df[model_cols].mean(axis=1)
cv_df['pred_pct'] = ((cv_df['Ensemble'] - cv_df['prev_y']) / cv_df['prev_y']) * 100

# Metrics
cv_df['correct'] = (np.sign(cv_df['actual_pct']) == np.sign(cv_df['pred_pct']))
cv_df['error'] = abs(cv_df['actual_pct'] - cv_df['pred_pct'])
cv_df['confidence'] = abs(cv_df['pred_pct'])
cv_df['high_conf'] = cv_df['confidence'] > CONFIDENCE_THRESHOLD

accuracy = cv_df['correct'].mean() * 100
avg_error = cv_df['error'].mean()
median_error = cv_df['error'].median()

high_conf_acc = cv_df[cv_df['high_conf']]['correct'].mean() * 100 if cv_df['high_conf'].sum() > 0 else 0
coverage = cv_df['high_conf'].mean() * 100

# Trading signals
cv_df['signal'] = 'HOLD'
cv_df.loc[cv_df['pred_pct'] > CONFIDENCE_THRESHOLD, 'signal'] = 'BUY'
cv_df.loc[cv_df['pred_pct'] < -CONFIDENCE_THRESHOLD, 'signal'] = 'SELL'

print("\n" + "="*70)
print("📊 BACKTEST PERFORMANCE METRICS")
print("="*70)
print(f"\n🎯 DIRECTIONAL ACCURACY:")
print(f"   • Overall: {accuracy:.2f}%")
print(f"   • High-confidence (>{CONFIDENCE_THRESHOLD}%): {high_conf_acc:.2f}%")
print(f"   • Coverage: {coverage:.1f}% of days")

print(f"\n📏 ERROR METRICS:")
print(f"   • Average error: {avg_error:.2f}%")
print(f"   • Median error: {median_error:.2f}%")
print(f"   • Std deviation: {cv_df['error'].std():.2f}%")

# Confusion matrix
tp = ((cv_df['actual_pct'] > 0) & (cv_df['pred_pct'] > 0)).sum()
tn = ((cv_df['actual_pct'] < 0) & (cv_df['pred_pct'] < 0)).sum()
fp = ((cv_df['actual_pct'] < 0) & (cv_df['pred_pct'] > 0)).sum()
fn = ((cv_df['actual_pct'] > 0) & (cv_df['pred_pct'] < 0)).sum()

precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📈 DETAILED STATISTICS:")
print(f"   • Precision: {precision:.2f}%")
print(f"   • Recall: {recall:.2f}%")
print(f"   • F1-Score: {f1:.2f}%")

# Signal distribution
buy_signals = (cv_df['signal'] == 'BUY').sum()
sell_signals = (cv_df['signal'] == 'SELL').sum()
hold_signals = (cv_df['signal'] == 'HOLD').sum()

print(f"\n📊 SIGNAL DISTRIBUTION:")
print(f"   • BUY signals: {buy_signals} ({buy_signals/len(cv_df)*100:.1f}%)")
print(f"   • SELL signals: {sell_signals} ({sell_signals/len(cv_df)*100:.1f}%)")
print(f"   • HOLD signals: {hold_signals} ({hold_signals/len(cv_df)*100:.1f}%)")

# 4. TRAIN FINAL MODEL ON ALL DATA
print("\n" + "="*70)
print("🎓 Training final model on ALL available data...")
print("="*70)

nf_final = NeuralForecast(models=models, freq='D')
nf_final.fit(df=ai_df)

# Make next-day prediction
print("\n🔮 Generating tomorrow's prediction...")
forecast = nf_final.predict()

# Get ensemble prediction
forecast_ensemble = forecast[[col for col in forecast.columns if col.startswith('Optimized_')]].mean(axis=1).values[0]
current_price = data['Close'].iloc[-1]
predicted_price = forecast_ensemble

predicted_change = ((predicted_price - current_price) / current_price) * 100
prediction_confidence = abs(predicted_change)

if predicted_change > CONFIDENCE_THRESHOLD:
    prediction_signal = "🟢 BUY"
    signal_emoji = "📈"
elif predicted_change < -CONFIDENCE_THRESHOLD:
    prediction_signal = "🔴 SELL"
    signal_emoji = "📉"
else:
    prediction_signal = "⚪ HOLD"
    signal_emoji = "➡️"

print("\n" + "="*70)
print("🔮 TOMORROW'S PREDICTION")
print("="*70)
print(f"Date: {datetime.now().date() + timedelta(days=1)}")
print(f"Current Price: ₹{current_price:.2f}")
print(f"Predicted Price: ₹{predicted_price:.2f}")
print(f"Expected Change: {predicted_change:+.2f}%")
print(f"Confidence: {prediction_confidence:.2f}%")
print(f"\n{signal_emoji} SIGNAL: {prediction_signal}")
print("="*70)

# Get technical context
latest = data.iloc[-1]
print(f"\n📊 CURRENT TECHNICAL INDICATORS:")
print(f"   • RSI: {latest['RSI']:.2f}")
print(f"   • MACD: {latest['MACD']:.2f}")
print(f"   • Volume Ratio: {latest['Volume_Ratio']:.2f}x")
print(f"   • 20-day Volatility: {latest['Volatility_20']*100:.2f}%")
print(f"   • Price vs 20-SMA: {((latest['Close'] - latest['SMA_20'])/latest['SMA_20']*100):+.2f}%")

# 5. VISUALIZATION
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Backtest performance
ax1.plot(cv_df['ds'], cv_df['y'], label='Actual', color='black', linewidth=2)
ax1.plot(cv_df['ds'], cv_df['Ensemble'], label='Predicted', color='cyan', linestyle='--', linewidth=2)
correct_days = cv_df[cv_df['correct']]
wrong_days = cv_df[~cv_df['correct']]
ax1.scatter(correct_days['ds'], correct_days['y'], color='green', s=50, alpha=0.5, zorder=5)
ax1.scatter(wrong_days['ds'], wrong_days['y'], color='red', s=50, alpha=0.5, zorder=5)
ax1.set_title(f'{STOCK_SYMBOL} - Production Model Backtest', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (₹)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction accuracy over time
window = 10
cv_df['rolling_acc'] = cv_df['correct'].rolling(window=window).mean() * 100
ax2.plot(cv_df['ds'], cv_df['rolling_acc'], color='purple', linewidth=2.5)
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax2.axhline(y=accuracy, color='green', linestyle='--', alpha=0.5, label=f'Overall: {accuracy:.1f}%')
ax2.fill_between(cv_df['ds'], 50, cv_df['rolling_acc'], 
                 where=(cv_df['rolling_acc'] > 50), alpha=0.3, color='green')
ax2.fill_between(cv_df['ds'], 50, cv_df['rolling_acc'], 
                 where=(cv_df['rolling_acc'] <= 50), alpha=0.3, color='red')
ax2.set_title(f'Rolling {window}-Day Accuracy', fontsize=12)
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Error distribution
ax3.hist(cv_df['error'], bins=30, color='orange', alpha=0.7, edgecolor='black')
ax3.axvline(avg_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_error:.2f}%')
ax3.axvline(median_error, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}%')
ax3.set_xlabel('Error (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('Prediction Error Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Signal performance
signal_performance = cv_df.groupby('signal')['correct'].mean() * 100
colors_sig = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
bars = ax4.bar(signal_performance.index, signal_performance.values, 
               color=[colors_sig.get(x, 'gray') for x in signal_performance.index],
               alpha=0.7, edgecolor='black')
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Accuracy by Signal Type')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_production_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved as '{STOCK_SYMBOL}_production_analysis.png'")

# 6. RECENT PREDICTIONS TABLE
print("\n" + "="*70)
print("📅 RECENT PREDICTIONS (Last 10 Days)")
print("="*70)
recent = cv_df[['ds', 'y', 'Ensemble', 'actual_pct', 'pred_pct', 'signal', 'correct']].tail(10)
recent.columns = ['Date', 'Actual', 'Predicted', 'Act%', 'Pred%', 'Signal', '✓']
recent['✓'] = recent['✓'].map({True: '✅', False: '❌'})
print(recent.to_string(index=False))

# 7. TRADING STRATEGY SIMULATION
print("\n" + "="*70)
print("💰 TRADING STRATEGY SIMULATION")
print("="*70)

# Calculate returns if we follow the signals
cv_df['strategy_return'] = 0.0
cv_df.loc[cv_df['signal'] == 'BUY', 'strategy_return'] = cv_df['actual_pct']
cv_df.loc[cv_df['signal'] == 'SELL', 'strategy_return'] = -cv_df['actual_pct']

cv_df['cumulative_strategy'] = (1 + cv_df['strategy_return']/100).cumprod()
cv_df['cumulative_buy_hold'] = (1 + cv_df['actual_pct']/100).cumprod()

strategy_return = (cv_df['cumulative_strategy'].iloc[-1] - 1) * 100
buy_hold_return = (cv_df['cumulative_buy_hold'].iloc[-1] - 1) * 100

print(f"Strategy Return: {strategy_return:+.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:+.2f}%")
print(f"Outperformance: {(strategy_return - buy_hold_return):+.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(cv_df['ds'], cv_df['cumulative_strategy'], label='AI Strategy', 
        color='green', linewidth=2)
plt.plot(cv_df['ds'], cv_df['cumulative_buy_hold'], label='Buy & Hold', 
        color='blue', linewidth=2, linestyle='--')
plt.title(f'{STOCK_SYMBOL} - Strategy Performance', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_strategy_performance.png', dpi=300, bbox_inches='tight')
print(f"✅ Strategy chart saved as '{STOCK_SYMBOL}_strategy_performance.png'")

plt.show()

print("\n" + "="*70)
print("✨ PRODUCTION MODEL READY!")
print("="*70)
