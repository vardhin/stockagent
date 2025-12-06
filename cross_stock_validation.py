import pandas as pd
import numpy as np
import json
import yfinance as yf
from neuralforecast import NeuralForecast
from datetime import datetime, timedelta
import warnings
import torch
import os

warnings.filterwarnings('ignore')

print("="*90)
print("🔬 CROSS-STOCK MODEL VALIDATION (Pre-trained NHITS)")
print("="*90)

# Load original model configuration
with open('TATASTEEL.NS_production_config.json', 'r') as f:
    config = json.load(f)

THRESHOLD = config['threshold']
INITIAL_CAPITAL = 100000

print(f"\n📋 Testing Configuration:")
print(f"   Threshold: {THRESHOLD}%")
print(f"   Capital: ₹{INITIAL_CAPITAL:,.0f}")
print(f"   Using Pre-trained NHITS Models from ./models/")

# Test stocks from steel sector
test_stocks = {
    'TATASTEEL.NS': 'Tata Steel (Original)',
    'JSWSTEEL.NS': 'JSW Steel',
    'HINDALCO.NS': 'Hindalco',
    'SAIL.NS': 'SAIL',
    'JINDALSTEL.NS': 'Jindal Steel',
    'COALINDIA.NS': 'Coal India',
}

def prepare_nhits_format(symbol, start_date='2024-01-01', end_date='2025-12-05'):
    """Download and prepare data in NeuralForecast format"""
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Handle MultiIndex columns (yfinance sometimes returns this)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Reset index to get Date as a column
    data = data.reset_index()
    
    if len(data) < 200:  # Need at least 200 days for model
        return None
    
    df = pd.DataFrame({
        'unique_id': '1',
        'ds': pd.to_datetime(data['Date']),
        'y': data['Close'].values
    })
    
    return df

def load_pretrained_models_and_predict(df):
    """Load pre-trained NHITS models and generate predictions"""
    
    # Check if models exist
    if not os.path.exists('./models/'):
        raise FileNotFoundError("Models directory not found. Please train models first using save_production_model.py")
    
    # Load the saved NeuralForecast ensemble
    nf = NeuralForecast.load(path='./models/')
    
    # Calculate how many predictions we can make
    # We need at least 200 days for the longest window model
    min_history = 200
    max_predictions = max(0, len(df) - min_history)
    
    if max_predictions == 0:
        raise ValueError(f"Time series too short. Need at least {min_history} days, got {len(df)}")
    
    # Limit to reasonable number of predictions (max 60 or whatever is available)
    n_predictions = min(60, max_predictions)
    
    predictions = []
    
    for i in range(n_predictions):
        # Get the data up to current point
        current_df = df.iloc[:min_history + i].copy()
        
        # Generate prediction for next day
        pred = nf.predict(df=current_df)
        
        # Get the actual value for this day
        actual_idx = min_history + i
        actual_value = df.iloc[actual_idx]['y']
        actual_date = df.iloc[actual_idx]['ds']
        
        # Get prediction columns dynamically (model names might be different)
        pred_cols = [col for col in pred.columns if col not in ['ds', 'unique_id']]
        
        # Store prediction with actual value
        pred_row = {
            'ds': actual_date,
            'y': actual_value,
        }
        
        # Add all prediction columns
        for col in pred_cols:
            pred_row[col] = pred[col].iloc[0]
        
        predictions.append(pred_row)
    
    # Convert to DataFrame
    cv_df = pd.DataFrame(predictions)
    
    return cv_df

def generate_signals_from_predictions(cv_df, threshold):
    """Generate BUY/SELL signals from predictions"""
    cv_df = cv_df.copy()
    
    # Calculate previous day's price
    cv_df['prev_y'] = cv_df['y'].shift(1)
    cv_df = cv_df.dropna()
    
    # Calculate actual percentage change
    cv_df['actual_pct'] = ((cv_df['y'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
    
    # Ensemble prediction (average of all model predictions, excluding metadata columns)
    pred_cols = [col for col in cv_df.columns if col not in ['ds', 'y', 'prev_y', 'actual_pct', 'unique_id']]
    cv_df['Ensemble'] = cv_df[pred_cols].mean(axis=1)
    cv_df['pred_pct'] = ((cv_df['Ensemble'] - cv_df['prev_y']) / cv_df['prev_y']) * 100
    cv_df['confidence'] = abs(cv_df['pred_pct'])
    
    # Generate signals
    cv_df['signal'] = 'HOLD'
    cv_df.loc[cv_df['pred_pct'] > threshold, 'signal'] = 'BUY'
    cv_df.loc[cv_df['pred_pct'] < -threshold, 'signal'] = 'SELL'
    
    return cv_df

def backtest_simple_strategy(signals_df, capital):
    """Simple signal following strategy (same as backtest bot)"""
    cash = capital
    shares = 0
    trades = []
    portfolio_values = []
    position_open = False
    entry_price = 0
    
    for idx, row in signals_df.iterrows():
        price = row['y']
        signal = row['signal']
        date = row['ds']
        
        # BUY signal
        if signal == 'BUY' and not position_open and cash > price:
            shares_to_buy = int(cash // price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                shares = shares_to_buy
                cash -= cost
                entry_price = price
                position_open = True
                trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'cost': cost
                })
        
        # SELL signal
        elif signal == 'SELL' and position_open:
            revenue = shares * price
            profit = revenue - (shares * entry_price)
            trades.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'shares': shares,
                'revenue': revenue,
                'profit': profit
            })
            cash += revenue
            shares = 0
            position_open = False
        
        # Track portfolio value
        portfolio_value = cash + (shares * price)
        portfolio_values.append(portfolio_value)
    
    # Close position at end if open
    if position_open:
        final_price = signals_df.iloc[-1]['y']
        revenue = shares * final_price
        profit = revenue - (shares * entry_price)
        trades.append({
            'date': signals_df.iloc[-1]['ds'],
            'action': 'SELL',
            'price': final_price,
            'shares': shares,
            'revenue': revenue,
            'profit': profit
        })
        cash += revenue
        shares = 0
    
    final_value = cash
    
    return {
        'final_value': final_value,
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def calculate_metrics(result, df, initial_capital):
    """Calculate performance metrics"""
    final_value = result['final_value']
    trades = result['trades']
    
    # Calculate returns
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
    total_return_inr = final_value - initial_capital
    
    # Calculate buy & hold return
    start_price = float(df['y'].iloc[0])
    end_price = float(df['y'].iloc[-1])
    bh_return = ((end_price - start_price) / start_price) * 100
    
    # Count trades and win rate
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    total_trades = len(buy_trades)
    winning_trades = 0
    losing_trades = 0
    
    for i in range(min(len(buy_trades), len(sell_trades))):
        profit = sell_trades[i].get('profit', 0)
        if profit > 0:
            winning_trades += 1
        elif profit < 0:
            losing_trades += 1
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate max drawdown
    portfolio_values = np.array(result['portfolio_values'])
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max * 100
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    # Sharpe ratio
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    # Outperformance
    outperformance = total_return_pct - bh_return
    
    return {
        'total_return_pct': total_return_pct,
        'total_return_inr': total_return_inr,
        'bh_return': bh_return,
        'outperformance': outperformance,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'final_value': final_value
    }

# Run validation on all stocks
print("\n" + "="*90)
print("🔄 TESTING PRE-TRAINED MODELS ON DIFFERENT STOCKS")
print("="*90)
print("\n⚠️  NOTE: Using TATASTEEL-trained models on other stocks")
print("   This tests if the model architecture generalizes, not if it's optimal for each stock.\n")

results = []

for symbol, name in test_stocks.items():
    print(f"\n📊 Testing: {name} ({symbol})")
    print("-" * 90)
    
    try:
        # Download data - need at least 200 days for models + 60 days for predictions
        print(f"   ⏳ Downloading data...")
        df = prepare_nhits_format(symbol, '2024-01-01', '2025-12-05')
        
        if df is None or len(df) < 200:
            print(f"   ❌ Insufficient data (need 200+ days)")
            continue
        
        print(f"   ✓ Downloaded {len(df)} days of data")
        
        # Load pre-trained models and predict
        print(f"   ⏳ Loading pre-trained NHITS models and generating predictions...")
        cv_df = load_pretrained_models_and_predict(df)
        
        # Generate signals
        print(f"   ⏳ Generating signals...")
        signals = generate_signals_from_predictions(cv_df, THRESHOLD)
        buy_signals = (signals['signal'] == 'BUY').sum()
        sell_signals = (signals['signal'] == 'SELL').sum()
        print(f"   ✓ Generated signals: {buy_signals} BUY, {sell_signals} SELL")
        
        # Backtest
        print(f"   ⏳ Running backtest...")
        backtest_result = backtest_simple_strategy(signals, INITIAL_CAPITAL)
        
        # Calculate metrics
        metrics = calculate_metrics(backtest_result, signals, INITIAL_CAPITAL)
        
        results.append({
            'Stock': symbol,
            'Name': name,
            **metrics
        })
        
        print(f"   ✅ Strategy Return: {metrics['total_return_pct']:+.2f}%")
        print(f"   ✅ Buy & Hold: {metrics['bh_return']:+.2f}%")
        print(f"   ✅ Outperformance: {metrics['outperformance']:+.2f}%")
        print(f"   ✅ Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   ✅ Trades: {metrics['total_trades']}")
        print(f"   ✅ Sharpe: {metrics['sharpe_ratio']:.2f}")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Analysis
print("\n" + "="*90)
print("📊 CROSS-STOCK VALIDATION RESULTS")
print("="*90)

if len(results) == 0:
    print("\n❌ No successful validations")
else:
    results_df = pd.DataFrame(results)
    
    # Display results table
    print("\n" + results_df[['Name', 'total_return_pct', 'bh_return', 'outperformance', 
                            'win_rate', 'total_trades', 'sharpe_ratio']].to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*90)
    print("📈 PERFORMANCE SUMMARY")
    print("="*90)
    
    avg_strategy_return = results_df['total_return_pct'].mean()
    avg_bh_return = results_df['bh_return'].mean()
    avg_outperformance = results_df['outperformance'].mean()
    avg_win_rate = results_df['win_rate'].mean()
    avg_sharpe = results_df['sharpe_ratio'].mean()
    
    print(f"\n💰 Average Strategy Return: {avg_strategy_return:+.2f}%")
    print(f"💰 Average Buy & Hold: {avg_bh_return:+.2f}%")
    print(f"💰 Average Outperformance: {avg_outperformance:+.2f}%")
    print(f"📊 Average Win Rate: {avg_win_rate:.1f}%")
    print(f"📊 Average Sharpe Ratio: {avg_sharpe:.2f}")
    
    # Consistency check
    positive_outperformance = (results_df['outperformance'] > 0).sum()
    total_stocks = len(results_df)
    consistency = (positive_outperformance / total_stocks) * 100
    
    print(f"\n🎯 Model Generalization:")
    print(f"   • Outperformed B&H: {positive_outperformance}/{total_stocks} stocks ({consistency:.1f}%)")
    
    if consistency >= 70:
        print(f"   ✅ HIGH GENERALIZATION - Model works well across stocks")
    elif consistency >= 50:
        print(f"   ⚠️  MODERATE GENERALIZATION - Model is somewhat stock-specific")
    else:
        print(f"   ❌ LOW GENERALIZATION - Model is highly TATASTEEL-specific")
    
    # Compare with original TATASTEEL
    tatasteel_result = results_df[results_df['Stock'] == 'TATASTEEL.NS']
    if not tatasteel_result.empty:
        tata_return = tatasteel_result.iloc[0]['total_return_pct']
        tata_outperformance = tatasteel_result.iloc[0]['outperformance']
        
        other_stocks = results_df[results_df['Stock'] != 'TATASTEEL.NS']
        avg_other_return = other_stocks['total_return_pct'].mean()
        avg_other_outperf = other_stocks['outperformance'].mean()
        
        print(f"\n📊 TATASTEEL vs Others:")
        print(f"   • TATASTEEL Return: {tata_return:+.2f}%")
        print(f"   • TATASTEEL Outperformance: {tata_outperformance:+.2f}%")
        print(f"   • Avg Other Stocks Return: {avg_other_return:+.2f}%")
        print(f"   • Avg Other Stocks Outperformance: {avg_other_outperf:+.2f}%")
        
        if tata_outperformance > avg_outperformance * 1.5:
            print(f"\n   ⚠️  WARNING: TATASTEEL outperformance is {tata_outperformance/avg_outperformance:.1f}x higher than average")
            print(f"   → Model is likely overfitted to TATASTEEL-specific patterns")
            print(f"   → Consider training separate models for each stock")
        else:
            print(f"\n   ✅ TATASTEEL performance is consistent with other stocks")
            print(f"   → Model architecture generalizes well")
    
    # Risk analysis
    print(f"\n⚠️  Risk Metrics:")
    print(f"   • Avg Max Drawdown: {results_df['max_drawdown'].mean():.2f}%")
    print(f"   • Worst Drawdown: {results_df['max_drawdown'].max():.2f}%")
    print(f"   • Best Sharpe: {results_df['sharpe_ratio'].max():.2f}")
    print(f"   • Worst Sharpe: {results_df['sharpe_ratio'].min():.2f}")
    
    # Final recommendation
    print("\n" + "="*90)
    print("🎯 DEPLOYMENT RECOMMENDATION")
    print("="*90)
    
    deploy_ready = (
        consistency >= 50 and
        avg_outperformance > 0 and
        avg_win_rate > 50
    )
    
    if deploy_ready:
        print("\n✅ MODEL CAN BE DEPLOYED")
        print("   • Model shows reasonable generalization")
        print("   • Positive average outperformance")
        
        if consistency < 70:
            print("\n⚠️  RECOMMENDATION: Train stock-specific models")
            print("   • Better results expected with individual training per stock")
            print("   • Use same architecture but train on each stock's data")
        else:
            print("\n🚀 PROCEED WITH CURRENT MODEL")
    else:
        print("\n⚠️  MODEL NEEDS IMPROVEMENT")
        
        if consistency < 50:
            print("   • Low generalization - model is TATASTEEL-specific")
            print("   → Train separate models for each stock")
        
        if avg_outperformance <= 0:
            print("   • Negative average outperformance")
            print("   → Model doesn't add value across stocks")
        
        if avg_win_rate <= 50:
            print("   • Low average win rate")
            print("   → Adjust threshold or improve model")
        
        print("\n⏸️  DO NOT DEPLOY - FIX ISSUES FIRST")
    
    # Save results
    results_df.to_csv('cross_stock_validation_results.csv', index=False)
    print(f"\n💾 Results saved to 'cross_stock_validation_results.csv'")
    
    # Save validation report
    validation_report = {
        'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'Pre-trained NHITS (TATASTEEL)',
        'stocks_tested': total_stocks,
        'avg_strategy_return': float(avg_strategy_return),
        'avg_bh_return': float(avg_bh_return),
        'avg_outperformance': float(avg_outperformance),
        'avg_win_rate': float(avg_win_rate),
        'avg_sharpe': float(avg_sharpe),
        'generalization_pct': float(consistency),
        'deploy_ready': bool(deploy_ready),  # Convert to Python bool
        'detailed_results': results_df.to_dict('records')
    }
    
    with open('cross_stock_validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"💾 Validation report saved to 'cross_stock_validation_report.json'")

print("\n" + "="*90)
print("✅ CROSS-STOCK VALIDATION COMPLETE")
print("="*90)