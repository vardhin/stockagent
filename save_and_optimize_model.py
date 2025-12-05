import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import warnings
import torch
from datetime import datetime, timedelta
import pickle
import json

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

STOCK_SYMBOL = "TATASTEEL.NS"
OPTIMAL_CONFIG = {
    'windows': [100, 150, 200],
    'learning_rate': 1e-4,
    'dropout': 0.15,
    'batch_size': 32,
    'max_steps': 2500,
    'scaler': 'standard',
    'n_blocks': [4, 3, 2],
    'mlp_units': [[768, 512], [512, 512], [512, 256]],
    'n_pool_kernel_size': [2, 2, 1],
    'n_freq_downsample': [8, 4, 1],
}

TARGET_SIGNALS_PER_60_DAYS = 12  # ~1 signal every 5 days

print("="*70)
print("🎯 MODEL TRAINING & THRESHOLD FINE-TUNING")
print("="*70)
print(f"Stock: {STOCK_SYMBOL}")
print(f"Target: 1 signal every 5-6 days (~12 signals/60 days)")
print("="*70)

# 1. DATA PREPARATION
def prepare_data(symbol, period="max"):
    data = yf.download(symbol, period=period, interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    
    # Indicators (same as before)
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    data['Volatility_5'] = data['Returns'].rolling(window=5).std()
    data['Volatility_10'] = data['Returns'].rolling(window=10).std()
    data['Volatility_20'] = data['Returns'].rolling(window=20).std()
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
    
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
    
    data = data.dropna()
    return data

print("\n📥 Loading data...")
data = prepare_data(STOCK_SYMBOL)
print(f"✅ Loaded {len(data)} days of data")

ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

# 2. BUILD AND TRAIN MODELS
print("\n🏗️  Building ensemble models...")
models = []
for i, window in enumerate(OPTIMAL_CONFIG['windows']):
    model = NHITS(
        h=1,  # 1-day for threshold optimization
        input_size=window,
        max_steps=OPTIMAL_CONFIG['max_steps'],
        scaler_type=OPTIMAL_CONFIG['scaler'],
        learning_rate=OPTIMAL_CONFIG['learning_rate'],
        batch_size=OPTIMAL_CONFIG['batch_size'],
        windows_batch_size=64,
        dropout_prob_theta=OPTIMAL_CONFIG['dropout'],
        n_blocks=OPTIMAL_CONFIG['n_blocks'],
        mlp_units=OPTIMAL_CONFIG['mlp_units'],
        n_pool_kernel_size=OPTIMAL_CONFIG['n_pool_kernel_size'],
        n_freq_downsample=OPTIMAL_CONFIG['n_freq_downsample'],
        random_seed=42 + i,
        alias=f"Model_{window}"
    )
    models.append(model)
    print(f"   ✓ Model {i+1}: {window}-day window")

nf = NeuralForecast(models=models, freq='D')

print("\n🔄 Running cross-validation...")
TEST_DAYS = 60
cv_df = nf.cross_validation(df=ai_df, n_windows=TEST_DAYS, step_size=1)

cv_df.reset_index(inplace=True)
cv_df['prev_y'] = cv_df['y'].shift(1)
cv_df = cv_df.dropna()
cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100

model_cols = [col for col in cv_df.columns if col.startswith('Model_')]
cv_df['Ensemble'] = cv_df[model_cols].mean(axis=1)
cv_df['pred_pct'] = ((cv_df['Ensemble'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
cv_df['confidence'] = abs(cv_df['pred_pct'])

# 3. FIND THRESHOLD FOR TARGET SIGNAL FREQUENCY
print("\n🎯 Finding threshold for ~12 signals per 60 days (1 every 5 days)...")
print("-"*70)

threshold_results = []
test_thresholds = np.arange(0.05, 1.0, 0.05)  # Fine-grained search

for threshold in test_thresholds:
    predictions = np.sign(cv_df['pred_pct'])
    predictions[cv_df['confidence'] < threshold] = 0
    
    actuals = np.sign(cv_df['actual_pct'])
    
    mask = predictions != 0
    if mask.sum() == 0:
        continue
    
    correct = (predictions[mask] == actuals[mask])
    
    tp = ((actuals > 0) & (predictions > 0)).sum()
    tn = ((actuals < 0) & (predictions < 0)).sum()
    fp = ((actuals < 0) & (predictions > 0)).sum()
    fn = ((actuals > 0) & (predictions < 0)).sum()
    
    accuracy = correct.mean() * 100 if mask.sum() > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    coverage = mask.mean() * 100
    signals = mask.sum()
    
    threshold_results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'coverage': coverage,
        'signals': signals,
        'frequency_days': TEST_DAYS / signals if signals > 0 else 999
    })
    
    # Print all results for analysis
    if 8 <= signals <= 15 and f1 >= 75:  # Focus on 8-15 signals (good range)
        print(f"Threshold {threshold:.2f}%: Signals={signals:2d} (every {TEST_DAYS/signals:.1f}d) | F1={f1:.1f}% | P={precision:.1f}% | R={recall:.1f}%")

results_df = pd.DataFrame(threshold_results)

# Find threshold closest to target with good F1
valid_results = results_df[(results_df['f1_score'] >= 75)]
if len(valid_results) > 0:
    # Find closest to 12 signals
    valid_results['signal_diff'] = abs(valid_results['signals'] - TARGET_SIGNALS_PER_60_DAYS)
    best_idx = valid_results['signal_diff'].idxmin()
    best_threshold = valid_results.loc[best_idx, 'threshold']
    best_signals = int(valid_results.loc[best_idx, 'signals'])
    best_f1 = valid_results.loc[best_idx, 'f1_score']
    best_precision = valid_results.loc[best_idx, 'precision']
    best_recall = valid_results.loc[best_idx, 'recall']
    best_freq = valid_results.loc[best_idx, 'frequency_days']
else:
    # Fallback to best F1
    best_idx = results_df['f1_score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_signals = int(results_df.loc[best_idx, 'signals'])
    best_f1 = results_df.loc[best_idx, 'f1_score']
    best_precision = results_df.loc[best_idx, 'precision']
    best_recall = results_df.loc[best_idx, 'recall']
    best_freq = results_df.loc[best_idx, 'frequency_days']

print("\n" + "="*70)
print("🏆 OPTIMAL THRESHOLD FOUND")
print("="*70)
print(f"\n✅ Threshold: {best_threshold:.2f}%")
print(f"✅ Signals: {best_signals} per 60 days (1 every {best_freq:.1f} days)")
print(f"✅ F1-Score: {best_f1:.2f}%")
print(f"✅ Precision: {best_precision:.2f}%")
print(f"✅ Recall: {best_recall:.2f}%")

# 4. TRAIN FINAL PRODUCTION MODEL
print("\n" + "="*70)
print("🎓 Training PRODUCTION model (for saving)...")
print("="*70)

models_production = []
for i, window in enumerate(OPTIMAL_CONFIG['windows']):
    model = NHITS(
        h=5,  # 5-day forecast
        input_size=window,
        max_steps=OPTIMAL_CONFIG['max_steps'],
        scaler_type=OPTIMAL_CONFIG['scaler'],
        learning_rate=OPTIMAL_CONFIG['learning_rate'],
        batch_size=OPTIMAL_CONFIG['batch_size'],
        windows_batch_size=64,
        dropout_prob_theta=OPTIMAL_CONFIG['dropout'],
        n_blocks=OPTIMAL_CONFIG['n_blocks'],
        mlp_units=OPTIMAL_CONFIG['mlp_units'],
        n_pool_kernel_size=OPTIMAL_CONFIG['n_pool_kernel_size'],
        n_freq_downsample=OPTIMAL_CONFIG['n_freq_downsample'],
        random_seed=42 + i,
        alias=f"Production_{window}"
    )
    models_production.append(model)

nf_production = NeuralForecast(models=models_production, freq='D')
nf_production.fit(df=ai_df)

# 5. SAVE EVERYTHING
print("\n💾 Saving trained models and configuration...")

# Save NeuralForecast models
nf_production.save(path='./models/', model_index=None, overwrite=True)
print("✅ Models saved to './models/' directory")

# Save data for backtesting
backtest_data = cv_df[['ds', 'y', 'Ensemble', 'pred_pct', 'actual_pct']].copy()
backtest_data.to_csv(f'{STOCK_SYMBOL}_backtest_data.csv', index=False)
print(f"✅ Backtest data saved to '{STOCK_SYMBOL}_backtest_data.csv'")

# Save full configuration
config_complete = {
    'stock_symbol': STOCK_SYMBOL,
    'model_config': OPTIMAL_CONFIG,
    'optimal_threshold': float(best_threshold),
    'performance_metrics': {
        'f1_score': float(best_f1),
        'precision': float(best_precision),
        'recall': float(best_recall),
        'signals_per_60_days': int(best_signals),
        'signal_frequency_days': float(best_freq)
    },
    'current_price': float(data['Close'].iloc[-1]),
    'last_training_date': str(data['Date'].iloc[-1]),
    'data_points': len(data),
    'test_period_days': TEST_DAYS
}

with open(f'{STOCK_SYMBOL}_model_config.json', 'w') as f:
    json.dump(config_complete, f, indent=2)
print(f"✅ Configuration saved to '{STOCK_SYMBOL}_model_config.json'")

# Save threshold analysis
results_df.to_csv(f'{STOCK_SYMBOL}_threshold_analysis.csv', index=False)
print(f"✅ Threshold analysis saved to '{STOCK_SYMBOL}_threshold_analysis.csv'")

# 6. VISUALIZATION
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Signals vs Threshold
ax1.scatter(results_df['threshold'], results_df['signals'], 
           c=results_df['f1_score'], cmap='RdYlGn', s=100, 
           edgecolors='black', alpha=0.7)
ax1.axhline(y=TARGET_SIGNALS_PER_60_DAYS, color='blue', linestyle='--', 
           linewidth=2, label=f'Target ({TARGET_SIGNALS_PER_60_DAYS} signals)')
ax1.axvline(x=best_threshold, color='red', linestyle='--', 
           linewidth=2, label=f'Optimal ({best_threshold:.2f}%)')
ax1.set_xlabel('Threshold (%)')
ax1.set_ylabel('Number of Signals')
ax1.set_title('Signal Frequency vs Threshold (color=F1-score)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(ax1.collections[0], ax=ax1, label='F1-Score (%)')

# Plot 2: F1-Score sweet spot
valid_plot = results_df[results_df['f1_score'] >= 70]
ax2.plot(valid_plot['signals'], valid_plot['f1_score'], 
        marker='o', linewidth=2.5, color='green')
ax2.scatter([best_signals], [best_f1], s=400, marker='*', 
           color='red', edgecolors='black', linewidths=2, 
           label='Optimal', zorder=5)
ax2.axvline(x=TARGET_SIGNALS_PER_60_DAYS, color='blue', 
           linestyle='--', alpha=0.5)
ax2.set_xlabel('Number of Signals')
ax2.set_ylabel('F1-Score (%)')
ax2.set_title('F1-Score vs Signal Count', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Performance metrics
metrics = ['Precision', 'Recall', 'F1-Score']
values = [best_precision, best_recall, best_f1]
colors = ['blue', 'red', 'green']
bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score (%)')
ax3.set_title(f'Performance Metrics (Threshold: {best_threshold:.2f}%)', fontweight='bold')
ax3.set_ylim([0, 110])
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Trading frequency visualization
ax4.text(0.5, 0.7, f'🎯 Trading Strategy', ha='center', fontsize=20, fontweight='bold')
ax4.text(0.5, 0.55, f'Signal every {best_freq:.1f} days', ha='center', fontsize=16)
ax4.text(0.5, 0.45, f'{best_signals} trades per 60 days', ha='center', fontsize=14)
ax4.text(0.5, 0.35, f'~{best_signals*2} trades per quarter', ha='center', fontsize=14)
ax4.text(0.5, 0.2, f'Win Rate: {best_precision:.1f}%', ha='center', fontsize=16, color='green')
ax4.text(0.5, 0.1, f'F1-Score: {best_f1:.1f}%', ha='center', fontsize=16, color='blue')
ax4.axis('off')

plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_model_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Summary visualization saved to '{STOCK_SYMBOL}_model_summary.png'")

plt.show()

print("\n" + "="*70)
print("🎉 MODEL TRAINING & SAVING COMPLETE!")
print("="*70)
print(f"\n📦 Saved Files:")
print(f"   1. ./models/ - Trained neural network models")
print(f"   2. {STOCK_SYMBOL}_model_config.json - Configuration")
print(f"   3. {STOCK_SYMBOL}_backtest_data.csv - Historical predictions")
print(f"   4. {STOCK_SYMBOL}_threshold_analysis.csv - Threshold sweep results")
print(f"   5. {STOCK_SYMBOL}_model_summary.png - Performance visualization")

print(f"\n📊 Model Summary:")
print(f"   • Stock: {STOCK_SYMBOL}")
print(f"   • Threshold: {best_threshold:.2f}%")
print(f"   • Signal Frequency: Every {best_freq:.1f} days")
print(f"   • Expected Signals: {best_signals} per 60 days")
print(f"   • F1-Score: {best_f1:.2f}%")
print(f"   • Win Rate: {best_precision:.1f}%")

print("\n" + "="*70)
print("🚀 READY FOR BACKTESTING!")
print("="*70)
print("Next step: Run the backtesting bot with:")
print("  • Starting Capital: ₹1,00,000")
print("  • Test Period: 60 days")
print("  • Multiple Trading Strategies")
print("="*70)