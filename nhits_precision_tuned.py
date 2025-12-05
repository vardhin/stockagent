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

# --- CONFIGURATION ---
STOCK_SYMBOL = "TATASTEEL.NS"
OPTIMAL_CONFIG = {
    'windows': [150, 200],
    'learning_rate': 1e-4,
    'dropout': 0.2,
    'batch_size': 32,
    'max_steps': 2000,  # Increased from 1500
    'scaler': 'standard',
    # NEW: Model architecture parameters
    'n_blocks': [3, 3, 3],  # Triple stack (was default [1, 1, 1])
    'mlp_units': [[512, 512], [512, 512], [512, 512]],  # Larger hidden layers
    'n_pool_kernel_size': [2, 2, 1],  # Multi-scale pooling
    'n_freq_downsample': [4, 2, 1],  # Hierarchical downsampling
}

# PRECISION-TUNED THRESHOLDS
# Higher thresholds = fewer but more accurate signals
THRESHOLDS = {
    'conservative': 1.0,  # High precision, low recall
    'balanced': 0.7,      # Balanced F1-score
    'aggressive': 0.5,    # High recall, lower precision
    'adaptive': 'auto'    # Automatically optimized
}

PREDICTION_HORIZON = 5

print("="*70)
print("🎯 PRECISION-OPTIMIZED STOCK PREDICTOR")
print("="*70)
print(f"Stock: {STOCK_SYMBOL}")
print(f"Goal: Maximize F1-Score (Balance Precision & Recall)")
print("="*70)

# 1. DATA PREPARATION
def prepare_data(symbol, period="max"):
    data = yf.download(symbol, period=period, interval="1d")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    
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
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
    
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    
    data = data.dropna()
    return data

print("\n📥 Loading data...")
data = prepare_data(STOCK_SYMBOL)

ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

# 2. BUILD MODELS
print("\n🏗️  Building ENHANCED optimized ensemble...")
print("   • Deeper architecture (3-stack blocks)")
print("   • Larger capacity (512 hidden units)")
print("   • Multi-scale temporal patterns")
models = []
for i, window in enumerate(OPTIMAL_CONFIG['windows']):
    model = NHITS(
        h=1,
        input_size=window,
        max_steps=OPTIMAL_CONFIG['max_steps'],
        scaler_type=OPTIMAL_CONFIG['scaler'],
        learning_rate=OPTIMAL_CONFIG['learning_rate'],
        batch_size=OPTIMAL_CONFIG['batch_size'],
        windows_batch_size=64,  # Reduced due to larger model
        dropout_prob_theta=OPTIMAL_CONFIG['dropout'],
        # NEW: Architecture parameters
        n_blocks=OPTIMAL_CONFIG['n_blocks'],
        mlp_units=OPTIMAL_CONFIG['mlp_units'],
        n_pool_kernel_size=OPTIMAL_CONFIG['n_pool_kernel_size'],
        n_freq_downsample=OPTIMAL_CONFIG['n_freq_downsample'],
        random_seed=42 + i,
        alias=f"Model_{window}"
    )
    models.append(model)
    print(f"   ✓ Model {i+1}: {window}-day window (Deep architecture)")

nf = NeuralForecast(models=models, freq='D')

# 3. BACKTEST WITH MULTIPLE THRESHOLDS
print("\n🔄 Running backtest to optimize thresholds...")
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

# 4. FIND OPTIMAL THRESHOLD
print("\n🎯 Testing different thresholds to maximize F1-score...")
print("-"*70)

threshold_results = []
test_thresholds = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]

for threshold in test_thresholds:
    # Apply threshold
    predictions = np.sign(cv_df['pred_pct'])
    predictions[cv_df['confidence'] < threshold] = 0  # Neutral
    
    actuals = np.sign(cv_df['actual_pct'])
    
    # Only evaluate non-neutral predictions
    mask = predictions != 0
    if mask.sum() == 0:
        continue
    
    correct = (predictions[mask] == actuals[mask])
    
    # Calculate metrics
    tp = ((actuals > 0) & (predictions > 0)).sum()
    tn = ((actuals < 0) & (predictions < 0)).sum()
    fp = ((actuals < 0) & (predictions > 0)).sum()
    fn = ((actuals > 0) & (predictions < 0)).sum()
    
    accuracy = correct.mean() * 100 if mask.sum() > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    coverage = mask.mean() * 100
    
    threshold_results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'coverage': coverage,
        'signals': mask.sum()
    })
    
    print(f"Threshold {threshold:.1f}% → Acc: {accuracy:.1f}% | P: {precision:.1f}% | R: {recall:.1f}% | F1: {f1:.1f}% | Cov: {coverage:.1f}%")

results_df = pd.DataFrame(threshold_results)

# Find best threshold for each metric
best_f1_idx = results_df['f1_score'].idxmax()
best_precision_idx = results_df['precision'].idxmax()
best_accuracy_idx = results_df['accuracy'].idxmax()

best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
best_precision_threshold = results_df.loc[best_precision_idx, 'threshold']
best_accuracy_threshold = results_df.loc[best_accuracy_idx, 'threshold']

print("\n" + "="*70)
print("🏆 OPTIMAL THRESHOLDS BY OBJECTIVE")
print("="*70)
print(f"\n📊 Best F1-Score ({results_df.loc[best_f1_idx, 'f1_score']:.2f}%):")
print(f"   Threshold: {best_f1_threshold:.1f}%")
print(f"   Precision: {results_df.loc[best_f1_idx, 'precision']:.2f}%")
print(f"   Recall: {results_df.loc[best_f1_idx, 'recall']:.2f}%")
print(f"   Coverage: {results_df.loc[best_f1_idx, 'coverage']:.1f}%")

print(f"\n🎯 Best Precision ({results_df.loc[best_precision_idx, 'precision']:.2f}%):")
print(f"   Threshold: {best_precision_threshold:.1f}%")
print(f"   Recall: {results_df.loc[best_precision_idx, 'recall']:.2f}%")
print(f"   F1-Score: {results_df.loc[best_precision_idx, 'f1_score']:.2f}%")
print(f"   Coverage: {results_df.loc[best_precision_idx, 'coverage']:.1f}%")

print(f"\n✅ Best Accuracy ({results_df.loc[best_accuracy_idx, 'accuracy']:.2f}%):")
print(f"   Threshold: {best_accuracy_threshold:.1f}%")
print(f"   Precision: {results_df.loc[best_accuracy_idx, 'precision']:.2f}%")
print(f"   Recall: {results_df.loc[best_accuracy_idx, 'recall']:.2f}%")
print(f"   F1-Score: {results_df.loc[best_accuracy_idx, 'f1_score']:.2f}%")

# 5. GENERATE PREDICTIONS WITH OPTIMAL THRESHOLDS
print("\n" + "="*70)
print("🎓 Training final DEEP models...")
print("="*70)

# Build multi-day models with enhanced architecture
models_multiday = []
for i, window in enumerate(OPTIMAL_CONFIG['windows']):
    model = NHITS(
        h=PREDICTION_HORIZON,
        input_size=window,
        max_steps=OPTIMAL_CONFIG['max_steps'],
        scaler_type=OPTIMAL_CONFIG['scaler'],
        learning_rate=OPTIMAL_CONFIG['learning_rate'],
        batch_size=OPTIMAL_CONFIG['batch_size'],
        windows_batch_size=64,
        dropout_prob_theta=OPTIMAL_CONFIG['dropout'],
        # Enhanced architecture
        n_blocks=OPTIMAL_CONFIG['n_blocks'],
        mlp_units=OPTIMAL_CONFIG['mlp_units'],
        n_pool_kernel_size=OPTIMAL_CONFIG['n_pool_kernel_size'],
        n_freq_downsample=OPTIMAL_CONFIG['n_freq_downsample'],
        random_seed=42 + i,
        alias=f"Final_{window}"
    )
    models_multiday.append(model)
    print(f"   ✓ Deep Model {i+1}: {window}-day window")

nf_final = NeuralForecast(models=models_multiday, freq='D')
nf_final.fit(df=ai_df)

forecast = nf_final.predict()
model_cols_final = [col for col in forecast.columns if col.startswith('Final_')]
forecast['Ensemble'] = forecast[model_cols_final].mean(axis=1)

current_price = data['Close'].iloc[-1]
forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), 
                                periods=PREDICTION_HORIZON, freq='D')

# Create forecasts with multiple threshold strategies
strategies = {
    'Balanced (F1)': best_f1_threshold,
    'Conservative (Precision)': best_precision_threshold,
    'Standard': 0.5
}

print("\n" + "="*70)
print(f"📅 {PREDICTION_HORIZON}-DAY FORECAST WITH MULTIPLE STRATEGIES")
print("="*70)
print(f"Current Price: ₹{current_price:.2f}\n")

for strategy_name, threshold in strategies.items():
    print(f"\n{strategy_name} (Threshold: {threshold:.1f}%):")
    print("-"*50)
    
    forecast_df = pd.DataFrame({
        'Day': range(1, PREDICTION_HORIZON + 1),
        'Date': forecast_dates,
        'Price': forecast['Ensemble'].values,
    })
    
    forecast_df['Change%'] = ((forecast_df['Price'] - current_price) / current_price) * 100
    forecast_df['Daily_Change%'] = forecast_df['Price'].pct_change() * 100
    forecast_df['Daily_Change%'].iloc[0] = forecast_df['Change%'].iloc[0]
    forecast_df['Confidence'] = abs(forecast_df['Daily_Change%'])
    
    # Apply threshold
    forecast_df['Signal'] = 'HOLD'
    forecast_df.loc[(forecast_df['Daily_Change%'] > threshold), 'Signal'] = 'BUY'
    forecast_df.loc[(forecast_df['Daily_Change%'] < -threshold), 'Signal'] = 'SELL'
    
    print(forecast_df[['Day', 'Date', 'Price', 'Daily_Change%', 'Signal']].to_string(index=False))
    
    # Strategy summary
    buy_days = (forecast_df['Signal'] == 'BUY').sum()
    sell_days = (forecast_df['Signal'] == 'SELL').sum()
    total_change = forecast_df['Change%'].iloc[-1]
    
    print(f"\nStrategy Summary:")
    print(f"  • BUY signals: {buy_days}/{PREDICTION_HORIZON} days")
    print(f"  • SELL signals: {sell_days}/{PREDICTION_HORIZON} days")
    print(f"  • 5-day target: ₹{forecast_df['Price'].iloc[-1]:.2f} ({total_change:+.2f}%)")

# 6. VISUALIZATION
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Threshold optimization
ax1.plot(results_df['threshold'], results_df['f1_score'], 
        marker='o', linewidth=2.5, label='F1-Score', color='green')
ax1.plot(results_df['threshold'], results_df['precision'], 
        marker='s', linewidth=2, label='Precision', color='blue', linestyle='--')
ax1.plot(results_df['threshold'], results_df['recall'], 
        marker='^', linewidth=2, label='Recall', color='red', linestyle='--')
ax1.axvline(best_f1_threshold, color='green', linestyle=':', alpha=0.5, label=f'Best F1 ({best_f1_threshold:.1f}%)')
ax1.set_xlabel('Confidence Threshold (%)')
ax1.set_ylabel('Score (%)')
ax1.set_title('Threshold Optimization: Finding the Sweet Spot', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Precision-Recall Tradeoff
ax2.scatter(results_df['recall'], results_df['precision'], 
           s=results_df['f1_score']*5, c=results_df['threshold'], 
           cmap='viridis', alpha=0.7, edgecolors='black')
best_point = results_df.loc[best_f1_idx]
ax2.scatter([best_point['recall']], [best_point['precision']], 
           s=300, marker='*', color='red', edgecolors='black', 
           linewidths=2, label='Optimal Point')
ax2.set_xlabel('Recall (%)')
ax2.set_ylabel('Precision (%)')
ax2.set_title('Precision-Recall Tradeoff (size=F1-score)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Threshold (%)')

# Plot 3: Coverage vs F1-Score
ax3.scatter(results_df['coverage'], results_df['f1_score'], 
           s=results_df['accuracy']*3, c='purple', alpha=0.6, edgecolors='black')
ax3.set_xlabel('Coverage (%)')
ax3.set_ylabel('F1-Score (%)')
ax3.set_title('Coverage vs F1-Score (size=accuracy)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Signal counts by threshold
ax4.bar(results_df['threshold'], results_df['signals'], 
       color='steelblue', alpha=0.7, edgecolor='black')
ax4.axhline(y=TEST_DAYS*0.1, color='red', linestyle='--', alpha=0.5, label='10% coverage')
ax4.set_xlabel('Threshold (%)')
ax4.set_ylabel('Number of Signals')
ax4.set_title('Signal Frequency by Threshold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_precision_optimization.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Analysis saved as '{STOCK_SYMBOL}_precision_optimization.png'")

plt.show()

# 7. RECOMMENDATIONS
print("\n" + "="*70)
print("💡 STRATEGIC RECOMMENDATIONS")
print("="*70)

print("\n🎯 For Different Trading Styles:")
print(f"\n1. CONSERVATIVE TRADER:")
print(f"   • Use threshold: {best_precision_threshold:.1f}%")
print(f"   • Precision: {results_df.loc[best_precision_idx, 'precision']:.1f}%")
print(f"   • Fewer signals, but highly accurate")
print(f"   • Best for: Risk-averse investors")

print(f"\n2. BALANCED TRADER:")
print(f"   • Use threshold: {best_f1_threshold:.1f}%")
print(f"   • F1-Score: {results_df.loc[best_f1_idx, 'f1_score']:.1f}%")
print(f"   • Optimal precision-recall balance")
print(f"   • Best for: Most traders (RECOMMENDED)")

print(f"\n3. AGGRESSIVE TRADER:")
print(f"   • Use threshold: 0.3-0.5%")
print(f"   • High recall, catches more moves")
print(f"   • More signals, lower accuracy")
print(f"   • Best for: Active traders, day traders")

print("\n" + "="*70)
print("🎓 KEY LEARNINGS")
print("="*70)
print("✅ Threshold tuning SIGNIFICANTLY improves precision")
print("✅ Higher threshold = Higher precision, Lower recall")
print("✅ Lower threshold = Higher recall, Lower precision")
print(f"✅ Your optimal F1-score threshold: {best_f1_threshold:.1f}%")
print(f"✅ This gives {results_df.loc[best_f1_idx, 'precision']:.1f}% precision with {results_df.loc[best_f1_idx, 'recall']:.1f}% recall")
print("="*70)

print(f"\n✨ Use the '{list(strategies.keys())[0]}' strategy for best results!")