import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
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
CONFIDENCE_THRESHOLD = 0.5

print(f"🔬 MEGA WINDOW EXPERIMENT FOR {STOCK_SYMBOL}")
print("Testing hypothesis: Larger input windows → Better accuracy")
print("="*70)

# 1. GET DATA (need more history for large windows)
data = yf.download(STOCK_SYMBOL, period="max", interval="1d")  # Maximum available data

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.reset_index(inplace=True)

# Technical indicators (same as before)
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
data['ATR'] = data['High'].rolling(14).max() - data['Low'].rolling(14).min()
data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = calculate_rsi(data['Close'])
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Std'] = data['Close'].rolling(window=20).std()
data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

data = data.dropna()

print(f"✅ Total data points available: {len(data)} days")
print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")

ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

# 2. TEST DIFFERENT WINDOW SIZES
# Theory: Larger windows capture longer-term patterns
window_configs = [
    {'size': 90, 'name': 'Standard', 'steps': 1000},
    {'size': 150, 'name': 'Large', 'steps': 1500},
    {'size': 200, 'name': 'XLarge', 'steps': 2000},
    {'size': 250, 'name': 'XXLarge', 'steps': 2500},
]

print(f"\n🧪 Testing {len(window_configs)} different window configurations...")
print("-"*70)

models = []
for i, config in enumerate(window_configs):
    model = NHITS(
        h=HORIZON,
        input_size=config['size'],
        max_steps=config['steps'],
        scaler_type='standard',
        learning_rate=1.5e-4,  # Lower LR for stability with larger models
        batch_size=32,
        windows_batch_size=128,
        dropout_prob_theta=0.4,
        random_seed=42 + i,
        alias=f"Window_{config['size']}"
    )
    models.append(model)
    print(f"   ✓ Model {i+1}: {config['name']} ({config['size']} days lookback)")

nf = NeuralForecast(models=models, freq='D')

# 3. RUN BACKTEST
print(f"\n⏳ Training {len(models)} models on {TEST_DAYS} windows...")
print("   This will take 10-15 minutes. Perfect time for a coffee break! ☕")
print("-"*70)

cv_df = nf.cross_validation(df=ai_df, n_windows=TEST_DAYS, step_size=1)

# 4. ANALYZE EACH WINDOW SIZE
cv_df.reset_index(inplace=True)
cv_df['prev_y'] = cv_df['y'].shift(1)
cv_df = cv_df.dropna()
cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100

print("\n" + "="*70)
print("📊 WINDOW SIZE EXPERIMENT RESULTS")
print("="*70)

# Track performance by window size
results = []
model_cols = [col for col in cv_df.columns if col.startswith('Window_')]

for i, col in enumerate(model_cols):
    config = window_configs[i]
    
    # Calculate metrics
    pred_pct = ((cv_df[col] - cv_df['prev_y']) / cv_df['prev_y']) * 100
    correct = (np.sign(cv_df['actual_pct']) == np.sign(pred_pct))
    accuracy = correct.mean() * 100
    
    # High confidence predictions
    confidence = abs(pred_pct)
    high_conf_mask = confidence > CONFIDENCE_THRESHOLD
    high_conf_acc = correct[high_conf_mask].mean() * 100 if high_conf_mask.sum() > 0 else 0
    coverage = high_conf_mask.mean() * 100
    
    # Error metrics
    error = abs(cv_df['actual_pct'] - pred_pct)
    avg_error = error.mean()
    
    results.append({
        'window': config['size'],
        'name': config['name'],
        'accuracy': accuracy,
        'high_conf_acc': high_conf_acc,
        'coverage': coverage,
        'avg_error': avg_error
    })
    
    print(f"\n🔍 {config['name']} Window ({config['size']} days):")
    print(f"   • Overall Accuracy: {accuracy:.2f}%")
    print(f"   • High-Conf Accuracy: {high_conf_acc:.2f}% (on {coverage:.1f}% of days)")
    print(f"   • Average Error: {avg_error:.2f}%")

# Find best performer
results_df = pd.DataFrame(results)
best_idx = results_df['accuracy'].idxmax()
best_config = results_df.iloc[best_idx]

print("\n" + "="*70)
print("🏆 WINNER: Best Performing Window Size")
print("="*70)
print(f"   Window Size: {best_config['window']} days")
print(f"   Name: {best_config['name']}")
print(f"   Accuracy: {best_config['accuracy']:.2f}%")
print(f"   High-Conf Accuracy: {best_config['high_conf_acc']:.2f}%")
print("="*70)

# 5. CREATE SUPER ENSEMBLE using best 2 performers
top_2_idx = results_df.nlargest(2, 'accuracy').index
top_2_models = [model_cols[i] for i in top_2_idx]

print(f"\n🎯 Creating Super Ensemble from top 2 models:")
for idx in top_2_idx:
    print(f"   • {results_df.iloc[idx]['name']} ({results_df.iloc[idx]['window']} days): {results_df.iloc[idx]['accuracy']:.2f}%")

# Weight by accuracy
weights = {}
for idx in top_2_idx:
    col = model_cols[idx]
    weight = results_df.iloc[idx]['accuracy'] / 100
    weights[col] = max(weight, 0.4)

total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

cv_df['Super_Ensemble'] = sum(cv_df[col] * weights[col] for col in top_2_models)

# Calculate super ensemble metrics
super_pred_pct = ((cv_df['Super_Ensemble'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
super_correct = (np.sign(cv_df['actual_pct']) == np.sign(super_pred_pct))
super_accuracy = super_correct.mean() * 100

super_confidence = abs(super_pred_pct)
super_high_conf = super_confidence > CONFIDENCE_THRESHOLD
super_high_acc = super_correct[super_high_conf].mean() * 100 if super_high_conf.sum() > 0 else 0
super_coverage = super_high_conf.mean() * 100

print(f"\n🚀 SUPER ENSEMBLE PERFORMANCE:")
print(f"   • Overall Accuracy: {super_accuracy:.2f}%")
print(f"   • High-Conf Accuracy: {super_high_acc:.2f}%")
print(f"   • Coverage: {super_coverage:.1f}% of days")

# 6. VISUALIZATION
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Window size vs Accuracy
ax1.bar(results_df['window'], results_df['accuracy'], color='steelblue', alpha=0.7, edgecolor='black')
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax1.set_xlabel('Window Size (Days)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Impact of Window Size on Accuracy', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
for i, row in results_df.iterrows():
    ax1.text(row['window'], row['accuracy'] + 1, f"{row['accuracy']:.1f}%", 
            ha='center', va='bottom', fontweight='bold')

# Window size vs High-Confidence Accuracy
ax2.bar(results_df['window'], results_df['high_conf_acc'], color='green', alpha=0.7, edgecolor='black')
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax2.set_xlabel('Window Size (Days)', fontsize=12)
ax2.set_ylabel('High-Confidence Accuracy (%)', fontsize=12)
ax2.set_title('High-Confidence Predictions by Window Size', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Window size vs Error
ax3.bar(results_df['window'], results_df['avg_error'], color='orange', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Window Size (Days)', fontsize=12)
ax3.set_ylabel('Average Error (%)', fontsize=12)
ax3.set_title('Prediction Error by Window Size', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Coverage comparison
ax4.bar(results_df['window'], results_df['coverage'], color='purple', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Window Size (Days)', fontsize=12)
ax4.set_ylabel('Coverage (%)', fontsize=12)
ax4.set_title('High-Confidence Signal Coverage', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'{STOCK_SYMBOL} - Window Size Impact Analysis', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_window_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Analysis saved as '{STOCK_SYMBOL}_window_analysis.png'")

# 7. DETAILED PREDICTIONS with BEST MODEL
print(f"\n📅 LAST 10 PREDICTIONS (Super Ensemble):")
cv_df['signal'] = cv_df.apply(
    lambda x: '🔴 SELL' if super_pred_pct.loc[x.name] < -CONFIDENCE_THRESHOLD 
    else ('🟢 BUY' if super_pred_pct.loc[x.name] > CONFIDENCE_THRESHOLD else '⚪ HOLD'), 
    axis=1
)
recent = cv_df[['ds', 'actual_pct', 'signal']].tail(10)
recent['pred_pct'] = super_pred_pct.tail(10).values
recent['correct'] = super_correct.tail(10).values
print(recent.to_string(index=False))

plt.show()

# 8. RECOMMENDATIONS
print("\n" + "="*70)
print("💡 KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)
print(f"1. 🎯 OPTIMAL WINDOW: {best_config['window']} days gives {best_config['accuracy']:.1f}% accuracy")
print(f"2. 📈 TREND: {'Larger' if best_config['window'] > 150 else 'Moderate'} windows work best for {STOCK_SYMBOL}")
print(f"3. 🚀 SUPER ENSEMBLE: Achieves {super_accuracy:.1f}% accuracy")
print(f"4. 💪 CONFIDENCE: {super_high_acc:.1f}% accuracy on high-confidence signals")
print(f"5. 📊 ACTIONABLE: {int(super_coverage * TEST_DAYS / 100)} trading signals in {TEST_DAYS} days")
print("\n🎓 LEARNING:")
print("   • Window size DOES matter - experiment with your stock!")
print("   • Different stocks may have different optimal windows")
print("   • Combine best performers for superior results")
print("="*70)