import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import warnings
import torch
from itertools import product

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# --- CONFIGURATION ---
STOCK_SYMBOL = "TATASTEEL.NS"
TEST_DAYS = 60
HORIZON = 1
CONFIDENCE_THRESHOLD = 0.5

# Fixed optimal window sizes from previous experiment
OPTIMAL_WINDOWS = [150, 200]

print(f"🔬 HYPERPARAMETER OPTIMIZATION FOR {STOCK_SYMBOL}")
print(f"Fixed window sizes: {OPTIMAL_WINDOWS}")
print("Testing: Learning Rate, Dropout, Batch Size, Max Steps")
print("="*70)

# 1. GET DATA
data = yf.download(STOCK_SYMBOL, period="max", interval="1d")

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

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = calculate_rsi(data['Close'])
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Std'] = data['Close'].rolling(window=20).std()
data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)

data = data.dropna()

ai_df = pd.DataFrame({
    'unique_id': '1',
    'ds': data['Date'],
    'y': data['Close']
})

print(f"✅ Data loaded: {len(data)} days")

# 2. HYPERPARAMETER EXPERIMENTS
experiments = [
    # Experiment 1: Learning Rate Impact
    {
        'name': 'Learning Rate',
        'configs': [
            {'lr': 1e-3, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 5e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 5e-5, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
        ]
    },
    # Experiment 2: Dropout Impact
    {
        'name': 'Dropout Rate',
        'configs': [
            {'lr': 1e-4, 'dropout': 0.2, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.3, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.5, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
        ]
    },
    # Experiment 3: Batch Size Impact
    {
        'name': 'Batch Size',
        'configs': [
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 16, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 64, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 128, 'steps': 1500, 'scaler': 'standard'},
        ]
    },
    # Experiment 4: Training Steps Impact
    {
        'name': 'Training Steps',
        'configs': [
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1000, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 2000, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 2500, 'scaler': 'standard'},
        ]
    },
    # Experiment 5: Scaler Type Impact
    {
        'name': 'Scaler Type',
        'configs': [
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'standard'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'robust'},
            {'lr': 1e-4, 'dropout': 0.4, 'batch': 32, 'steps': 1500, 'scaler': 'minmax'},
        ]
    },
]

# 3. RUN EXPERIMENTS
all_results = []

for exp_idx, experiment in enumerate(experiments):
    print(f"\n{'='*70}")
    print(f"🧪 EXPERIMENT {exp_idx + 1}: {experiment['name']}")
    print(f"{'='*70}")
    
    for config_idx, config in enumerate(experiment['configs']):
        print(f"\n   Testing configuration {config_idx + 1}/{len(experiment['configs'])}: {config}")
        
        # Create models for both window sizes
        models = []
        for window in OPTIMAL_WINDOWS:
            model = NHITS(
                h=HORIZON,
                input_size=window,
                max_steps=config['steps'],
                scaler_type=config['scaler'],
                learning_rate=config['lr'],
                batch_size=config['batch'],
                windows_batch_size=128,
                dropout_prob_theta=config['dropout'],
                random_seed=42,
                alias=f"W{window}"
            )
            models.append(model)
        
        # Train and evaluate
        nf = NeuralForecast(models=models, freq='D')
        cv_df = nf.cross_validation(df=ai_df, n_windows=TEST_DAYS, step_size=1)
        
        # Calculate metrics
        cv_df.reset_index(inplace=True)
        cv_df['prev_y'] = cv_df['y'].shift(1)
        cv_df = cv_df.dropna()
        cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
        
        # Ensemble prediction
        model_cols = [col for col in cv_df.columns if col.startswith('W')]
        cv_df['Ensemble'] = cv_df[model_cols].mean(axis=1)
        
        pred_pct = ((cv_df['Ensemble'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
        correct = (np.sign(cv_df['actual_pct']) == np.sign(pred_pct))
        accuracy = correct.mean() * 100
        
        # Error metrics
        error = abs(cv_df['actual_pct'] - pred_pct)
        avg_error = error.mean()
        median_error = error.median()
        
        # High confidence metrics
        confidence = abs(pred_pct)
        high_conf_mask = confidence > CONFIDENCE_THRESHOLD
        high_conf_acc = correct[high_conf_mask].mean() * 100 if high_conf_mask.sum() > 0 else 0
        coverage = high_conf_mask.mean() * 100
        
        # Store results
        result = {
            'experiment': experiment['name'],
            'config': str(config),
            'accuracy': accuracy,
            'avg_error': avg_error,
            'median_error': median_error,
            'high_conf_acc': high_conf_acc,
            'coverage': coverage,
            **config
        }
        all_results.append(result)
        
        print(f"      ✓ Accuracy: {accuracy:.2f}% | Error: {avg_error:.2f}% | High-Conf: {high_conf_acc:.2f}%")

# 4. ANALYZE RESULTS
results_df = pd.DataFrame(all_results)

print("\n" + "="*70)
print("📊 COMPREHENSIVE RESULTS SUMMARY")
print("="*70)

for exp_name in results_df['experiment'].unique():
    exp_results = results_df[results_df['experiment'] == exp_name]
    best_idx = exp_results['accuracy'].idxmax()
    best = exp_results.loc[best_idx]
    
    print(f"\n🎯 {exp_name}:")
    print(f"   Best Configuration: LR={best['lr']}, Dropout={best['dropout']}, Batch={best['batch']}, Steps={best['steps']}, Scaler={best['scaler']}")
    print(f"   ├─ Accuracy: {best['accuracy']:.2f}%")
    print(f"   ├─ Avg Error: {best['avg_error']:.2f}%")
    print(f"   ├─ Median Error: {best['median_error']:.2f}%")
    print(f"   ├─ High-Conf Accuracy: {best['high_conf_acc']:.2f}%")
    print(f"   └─ Coverage: {best['coverage']:.1f}%")

# Find overall best configuration
best_overall_idx = results_df['accuracy'].idxmax()
best_overall = results_df.loc[best_overall_idx]

print("\n" + "="*70)
print("🏆 OVERALL BEST CONFIGURATION")
print("="*70)
print(f"Experiment: {best_overall['experiment']}")
print(f"Configuration:")
print(f"   • Learning Rate: {best_overall['lr']}")
print(f"   • Dropout: {best_overall['dropout']}")
print(f"   • Batch Size: {best_overall['batch']}")
print(f"   • Training Steps: {best_overall['steps']}")
print(f"   • Scaler: {best_overall['scaler']}")
print(f"\nPerformance:")
print(f"   • Directional Accuracy: {best_overall['accuracy']:.2f}%")
print(f"   • Average Error: {best_overall['avg_error']:.2f}%")
print(f"   • Median Error: {best_overall['median_error']:.2f}%")
print(f"   • High-Confidence Accuracy: {best_overall['high_conf_acc']:.2f}%")
print(f"   • Coverage: {best_overall['coverage']:.1f}%")
print("="*70)

# 5. VISUALIZATION
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle(f'{STOCK_SYMBOL} - Hyperparameter Impact Analysis', fontsize=16, fontweight='bold')

exp_names = results_df['experiment'].unique()
colors = ['steelblue', 'green', 'orange', 'purple', 'red']

# Plot 1: Accuracy by Experiment
ax = axes[0, 0]
for i, exp_name in enumerate(exp_names):
    exp_data = results_df[results_df['experiment'] == exp_name]
    ax.plot(range(len(exp_data)), exp_data['accuracy'], 
           marker='o', label=exp_name, linewidth=2, color=colors[i])
ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Random')
ax.set_xlabel('Configuration Index')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Across Experiments')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Error by Experiment
ax = axes[0, 1]
for i, exp_name in enumerate(exp_names):
    exp_data = results_df[results_df['experiment'] == exp_name]
    ax.plot(range(len(exp_data)), exp_data['avg_error'], 
           marker='s', label=exp_name, linewidth=2, color=colors[i])
ax.set_xlabel('Configuration Index')
ax.set_ylabel('Average Error (%)')
ax.set_title('Prediction Error Across Experiments')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Learning Rate Impact
ax = axes[1, 0]
lr_data = results_df[results_df['experiment'] == 'Learning Rate']
x_pos = range(len(lr_data))
width = 0.35
ax.bar([p - width/2 for p in x_pos], lr_data['accuracy'], width, 
       label='Accuracy', color='steelblue', alpha=0.7)
ax.bar([p + width/2 for p in x_pos], lr_data['avg_error'] * 10, width, 
       label='Error×10', color='orange', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{lr:.0e}" for lr in lr_data['lr']], rotation=45)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Percentage')
ax.set_title('Learning Rate Impact')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Dropout Impact
ax = axes[1, 1]
drop_data = results_df[results_df['experiment'] == 'Dropout Rate']
x_pos = range(len(drop_data))
ax.bar([p - width/2 for p in x_pos], drop_data['accuracy'], width, 
       label='Accuracy', color='green', alpha=0.7)
ax.bar([p + width/2 for p in x_pos], drop_data['avg_error'] * 10, width, 
       label='Error×10', color='red', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{d:.1f}" for d in drop_data['dropout']])
ax.set_xlabel('Dropout Rate')
ax.set_ylabel('Percentage')
ax.set_title('Dropout Impact')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 5: Accuracy vs Error Scatter
ax = axes[2, 0]
scatter = ax.scatter(results_df['accuracy'], results_df['avg_error'], 
                    c=range(len(results_df)), cmap='viridis', s=100, alpha=0.6)
ax.set_xlabel('Accuracy (%)')
ax.set_ylabel('Average Error (%)')
ax.set_title('Accuracy vs Error Trade-off')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Config Index')

# Plot 6: High-Confidence Performance
ax = axes[2, 1]
ax.scatter(results_df['high_conf_acc'], results_df['coverage'], 
          s=results_df['accuracy']*3, alpha=0.6, c='purple')
ax.set_xlabel('High-Confidence Accuracy (%)')
ax.set_ylabel('Coverage (%)')
ax.set_title('High-Confidence Performance (size=overall accuracy)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_hyperparam_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved as '{STOCK_SYMBOL}_hyperparam_analysis.png'")
plt.show()

# 6. SUMMARY TABLE
print("\n" + "="*70)
print("📋 DETAILED RESULTS TABLE")
print("="*70)
summary_cols = ['experiment', 'lr', 'dropout', 'batch', 'steps', 'scaler', 
                'accuracy', 'avg_error', 'median_error', 'high_conf_acc', 'coverage']
print(results_df[summary_cols].to_string(index=False))

# 7. RECOMMENDATIONS
print("\n" + "="*70)
print("💡 RECOMMENDATIONS FOR PRODUCTION")
print("="*70)
print(f"1. 🎯 Use Window Sizes: {OPTIMAL_WINDOWS} days")
print(f"2. 📚 Learning Rate: {best_overall['lr']} (from {best_overall['experiment']})")
print(f"3. 🎲 Dropout: {best_overall['dropout']}")
print(f"4. 📦 Batch Size: {best_overall['batch']}")
print(f"5. 🔄 Training Steps: {best_overall['steps']}")
print(f"6. 📊 Scaler: {best_overall['scaler']}")
print(f"\n🎉 Expected Performance:")
print(f"   • Directional Accuracy: ~{best_overall['accuracy']:.1f}%")
print(f"   • Average Error: ~{best_overall['avg_error']:.2f}%")
print(f"   • High-Confidence Accuracy: ~{best_overall['high_conf_acc']:.1f}%")
print("="*70)