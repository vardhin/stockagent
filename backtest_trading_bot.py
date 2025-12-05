import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("💰 TRADING BOT BACKTESTING ENGINE")
print("="*80)

# 1. LOAD CONFIGURATION AND DATA
print("\n📥 Loading saved configuration and data...")

with open('TATASTEEL.NS_production_config.json', 'r') as f:
    config = json.load(f)

STOCK_SYMBOL = config['stock_symbol']
THRESHOLD = config['threshold']
INITIAL_CAPITAL = 100000  # ₹1,00,000

# Load backtest signals
signals_df = pd.read_csv(f'{STOCK_SYMBOL}_backtest_signals.csv')
signals_df['ds'] = pd.to_datetime(signals_df['ds'])

print(f"✅ Loaded {len(signals_df)} days of signals")
print(f"   Stock: {STOCK_SYMBOL}")
print(f"   Threshold: {THRESHOLD}%")
print(f"   Starting Capital: ₹{INITIAL_CAPITAL:,.0f}")

# 2. DEFINE TRADING STRATEGIES

class TradingStrategy:
    def __init__(self, name, capital):
        self.name = name
        self.initial_capital = capital
        self.capital = capital
        self.shares = 0
        self.trades = []
        self.portfolio_value = []
        self.dates = []
        
    def execute_trade(self, date, action, price, shares, reason=""):
        """Record a trade"""
        cost = shares * price
        self.trades.append({
            'date': date,
            'action': action,
            'price': price,
            'shares': shares,
            'cost': cost,
            'reason': reason,
            'portfolio_value': self.get_portfolio_value(price)
        })
    
    def get_portfolio_value(self, current_price):
        """Calculate current portfolio value"""
        return self.capital + (self.shares * current_price)
    
    def record_portfolio(self, date, current_price):
        """Track daily portfolio value"""
        self.dates.append(date)
        self.portfolio_value.append(self.get_portfolio_value(current_price))


# STRATEGY 1: Simple Signal Following
def strategy_simple_signal(df, capital):
    """Buy on BUY signal, sell on SELL signal"""
    strategy = TradingStrategy("Simple Signal", capital)
    position_open = False
    entry_price = 0
    
    for idx, row in df.iterrows():
        price = row['y']
        signal = row['signal']
        date = row['ds']
        
        if signal == 'BUY' and not position_open and strategy.capital > price:
            # Buy maximum shares
            shares_to_buy = int(strategy.capital // price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                strategy.shares += shares_to_buy
                strategy.capital -= cost
                entry_price = price
                position_open = True
                strategy.execute_trade(date, 'BUY', price, shares_to_buy, "Signal: BUY")
        
        elif signal == 'SELL' and position_open:
            # Sell all shares
            revenue = strategy.shares * price
            profit = revenue - (strategy.shares * entry_price)
            strategy.execute_trade(date, 'SELL', price, strategy.shares, 
                                 f"Signal: SELL | P/L: ₹{profit:,.0f}")
            strategy.capital += revenue
            strategy.shares = 0
            position_open = False
        
        strategy.record_portfolio(date, price)
    
    # Close position at end if open
    if position_open:
        final_price = df.iloc[-1]['y']
        revenue = strategy.shares * final_price
        strategy.capital += revenue
        strategy.execute_trade(df.iloc[-1]['ds'], 'SELL', final_price, 
                             strategy.shares, "End of period")
        strategy.shares = 0
    
    return strategy


# STRATEGY 2: Stop-Loss & Take-Profit
def strategy_stop_loss(df, capital, stop_loss_pct=3.0, take_profit_pct=5.0):
    """Buy on signal, exit at -3% or +5%"""
    strategy = TradingStrategy(f"Stop-Loss ({stop_loss_pct}%/{take_profit_pct}%)", capital)
    position_open = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    for idx, row in df.iterrows():
        price = row['y']
        signal = row['signal']
        date = row['ds']
        
        if signal == 'BUY' and not position_open and strategy.capital > price:
            shares_to_buy = int(strategy.capital // price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                strategy.shares += shares_to_buy
                strategy.capital -= cost
                entry_price = price
                stop_loss = entry_price * (1 - stop_loss_pct/100)
                take_profit = entry_price * (1 + take_profit_pct/100)
                position_open = True
                strategy.execute_trade(date, 'BUY', price, shares_to_buy, 
                                     f"Signal BUY | SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}")
        
        elif position_open:
            # Check stop-loss
            if price <= stop_loss:
                revenue = strategy.shares * price
                loss = revenue - (strategy.shares * entry_price)
                strategy.execute_trade(date, 'SELL', price, strategy.shares, 
                                     f"Stop-Loss Hit | Loss: ₹{loss:,.0f}")
                strategy.capital += revenue
                strategy.shares = 0
                position_open = False
            
            # Check take-profit
            elif price >= take_profit:
                revenue = strategy.shares * price
                profit = revenue - (strategy.shares * entry_price)
                strategy.execute_trade(date, 'SELL', price, strategy.shares, 
                                     f"Take-Profit Hit | Profit: ₹{profit:,.0f}")
                strategy.capital += revenue
                strategy.shares = 0
                position_open = False
            
            # Check SELL signal
            elif signal == 'SELL':
                revenue = strategy.shares * price
                pnl = revenue - (strategy.shares * entry_price)
                strategy.execute_trade(date, 'SELL', price, strategy.shares, 
                                     f"SELL Signal | P/L: ₹{pnl:,.0f}")
                strategy.capital += revenue
                strategy.shares = 0
                position_open = False
        
        strategy.record_portfolio(date, price)
    
    # Close position at end
    if position_open:
        final_price = df.iloc[-1]['y']
        revenue = strategy.shares * final_price
        strategy.capital += revenue
        strategy.execute_trade(df.iloc[-1]['ds'], 'SELL', final_price, 
                             strategy.shares, "End of period")
        strategy.shares = 0
    
    return strategy


# STRATEGY 3: Confidence-Based Position Sizing
def strategy_confidence_sizing(df, capital):
    """Adjust position size based on confidence level"""
    strategy = TradingStrategy("Confidence-Based Sizing", capital)
    position_open = False
    entry_price = 0
    
    for idx, row in df.iterrows():
        price = row['y']
        signal = row['signal']
        confidence = row['confidence']
        date = row['ds']
        
        if signal == 'BUY' and not position_open and strategy.capital > price:
            # Position size based on confidence (min 30%, max 70%)
            position_pct = min(0.7, max(0.3, confidence / 2))
            invest_amount = strategy.capital * position_pct
            shares_to_buy = int(invest_amount // price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                strategy.shares += shares_to_buy
                strategy.capital -= cost
                entry_price = price
                position_open = True
                strategy.execute_trade(date, 'BUY', price, shares_to_buy, 
                                     f"BUY (Confidence: {confidence:.2f}%, Size: {position_pct*100:.1f}%)")
        
        elif signal == 'SELL' and position_open:
            revenue = strategy.shares * price
            pnl = revenue - (strategy.shares * entry_price)
            strategy.execute_trade(date, 'SELL', price, strategy.shares, 
                                 f"SELL | P/L: ₹{pnl:,.0f}")
            strategy.capital += revenue
            strategy.shares = 0
            position_open = False
        
        strategy.record_portfolio(date, price)
    
    if position_open:
        final_price = df.iloc[-1]['y']
        revenue = strategy.shares * final_price
        strategy.capital += revenue
        strategy.execute_trade(df.iloc[-1]['ds'], 'SELL', final_price, 
                             strategy.shares, "End of period")
        strategy.shares = 0
    
    return strategy


# STRATEGY 4: Hold for N Days
def strategy_hold_duration(df, capital, hold_days=3):
    """Buy on signal and hold for N days"""
    strategy = TradingStrategy(f"Hold {hold_days} Days", capital)
    position_open = False
    entry_date = None
    entry_price = 0
    
    for idx, row in df.iterrows():
        price = row['y']
        signal = row['signal']
        date = row['ds']
        
        if signal == 'BUY' and not position_open and strategy.capital > price:
            shares_to_buy = int(strategy.capital // price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                strategy.shares += shares_to_buy
                strategy.capital -= cost
                entry_price = price
                entry_date = date
                position_open = True
                strategy.execute_trade(date, 'BUY', price, shares_to_buy, 
                                     f"BUY (Hold {hold_days} days)")
        
        elif position_open:
            days_held = (date - entry_date).days
            
            # Exit after N days
            if days_held >= hold_days:
                revenue = strategy.shares * price
                pnl = revenue - (strategy.shares * entry_price)
                strategy.execute_trade(date, 'SELL', price, strategy.shares, 
                                     f"Held {days_held} days | P/L: ₹{pnl:,.0f}")
                strategy.capital += revenue
                strategy.shares = 0
                position_open = False
            
            # Or SELL signal
            elif signal == 'SELL':
                revenue = strategy.shares * price
                pnl = revenue - (strategy.shares * entry_price)
                strategy.execute_trade(date, 'SELL', price, strategy.shares, 
                                     f"SELL Signal (held {days_held}d) | P/L: ₹{pnl:,.0f}")
                strategy.capital += revenue
                strategy.shares = 0
                position_open = False
        
        strategy.record_portfolio(date, price)
    
    if position_open:
        final_price = df.iloc[-1]['y']
        revenue = strategy.shares * final_price
        strategy.capital += revenue
        strategy.execute_trade(df.iloc[-1]['ds'], 'SELL', final_price, 
                             strategy.shares, "End of period")
        strategy.shares = 0
    
    return strategy


# STRATEGY 5: Buy & Hold Benchmark
def strategy_buy_and_hold(df, capital):
    """Buy at start and hold till end (benchmark)"""
    strategy = TradingStrategy("Buy & Hold (Benchmark)", capital)
    
    first_price = df.iloc[0]['y']
    shares = int(capital // first_price)
    cost = shares * first_price
    
    strategy.shares = shares
    strategy.capital -= cost
    strategy.execute_trade(df.iloc[0]['ds'], 'BUY', first_price, shares, "Initial Buy")
    
    for idx, row in df.iterrows():
        strategy.record_portfolio(row['ds'], row['y'])
    
    final_price = df.iloc[-1]['y']
    revenue = strategy.shares * final_price
    strategy.capital += revenue
    strategy.execute_trade(df.iloc[-1]['ds'], 'SELL', final_price, 
                         strategy.shares, "Final Sell")
    strategy.shares = 0
    
    return strategy


# 3. RUN ALL STRATEGIES
print("\n" + "="*80)
print("🏃 RUNNING BACKTESTS...")
print("="*80)

strategies_results = []

print("\n1️⃣  Strategy 1: Simple Signal Following")
s1 = strategy_simple_signal(signals_df, INITIAL_CAPITAL)
strategies_results.append(s1)
print(f"   ✅ Completed: {len(s1.trades)} trades")

print("\n2️⃣  Strategy 2: Stop-Loss (3%) & Take-Profit (5%)")
s2 = strategy_stop_loss(signals_df, INITIAL_CAPITAL, 3.0, 5.0)
strategies_results.append(s2)
print(f"   ✅ Completed: {len(s2.trades)} trades")

print("\n3️⃣  Strategy 3: Confidence-Based Position Sizing")
s3 = strategy_confidence_sizing(signals_df, INITIAL_CAPITAL)
strategies_results.append(s3)
print(f"   ✅ Completed: {len(s3.trades)} trades")

print("\n4️⃣  Strategy 4: Hold for 3 Days")
s4 = strategy_hold_duration(signals_df, INITIAL_CAPITAL, 3)
strategies_results.append(s4)
print(f"   ✅ Completed: {len(s4.trades)} trades")

print("\n5️⃣  Strategy 5: Buy & Hold Benchmark")
s5 = strategy_buy_and_hold(signals_df, INITIAL_CAPITAL)
strategies_results.append(s5)
print(f"   ✅ Completed: {len(s5.trades)} trades")


# 4. CALCULATE PERFORMANCE METRICS
def calculate_metrics(strategy):
    """Calculate performance metrics"""
    final_value = strategy.portfolio_value[-1]
    total_return = ((final_value - strategy.initial_capital) / strategy.initial_capital) * 100
    
    # Count winning and losing trades
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    
    for i in range(0, len(strategy.trades)-1, 2):  # Pairs of BUY/SELL
        if i+1 < len(strategy.trades):
            buy_trade = strategy.trades[i]
            sell_trade = strategy.trades[i+1]
            pnl = (sell_trade['price'] - buy_trade['price']) * buy_trade['shares']
            
            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
            else:
                losing_trades += 1
                total_loss += abs(pnl)
    
    total_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate max drawdown
    portfolio_values = np.array(strategy.portfolio_value)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max * 100
    max_drawdown = abs(drawdown.min())
    
    # Sharpe ratio (simplified)
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    return {
        'final_value': final_value,
        'total_return_pct': total_return,
        'total_return_inr': final_value - strategy.initial_capital,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe
    }


print("\n" + "="*80)
print("📊 PERFORMANCE COMPARISON")
print("="*80)

results_summary = []

for strategy in strategies_results:
    metrics = calculate_metrics(strategy)
    results_summary.append({
        'Strategy': strategy.name,
        **metrics
    })
    
    print(f"\n{strategy.name}")
    print("-" * 80)
    print(f"Final Portfolio Value: ₹{metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:+.2f}% (₹{metrics['total_return_inr']:+,.0f})")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}% ({metrics['winning_trades']}W / {metrics['losing_trades']}L)")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Find best strategy
results_df = pd.DataFrame(results_summary)
best_strategy_idx = results_df['total_return_pct'].idxmax()
best_strategy = results_df.iloc[best_strategy_idx]

print("\n" + "="*80)
print("🏆 BEST PERFORMING STRATEGY")
print("="*80)
print(f"\n{best_strategy['Strategy']}")
print(f"   Return: {best_strategy['total_return_pct']:+.2f}%")
print(f"   Final Value: ₹{best_strategy['final_value']:,.2f}")
print(f"   Win Rate: {best_strategy['win_rate']:.1f}%")
print(f"   Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")

# Save results
results_df.to_csv(f'{STOCK_SYMBOL}_strategy_comparison.csv', index=False)
print(f"\n✅ Results saved to '{STOCK_SYMBOL}_strategy_comparison.csv'")

# 5. VISUALIZATION
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Portfolio value over time
for strategy in strategies_results:
    ax1.plot(strategy.dates, strategy.portfolio_value, label=strategy.name, linewidth=2)
ax1.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Value (₹)')
ax1.set_title('Portfolio Value Over Time', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Returns comparison
returns = results_df['total_return_pct'].values
strategies_names = results_df['Strategy'].values
colors = ['green' if x > 0 else 'red' for x in returns]
bars = ax2.barh(strategies_names, returns, color=colors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, returns):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, 
            f'{val:+.1f}%', ha='left' if val > 0 else 'right', 
            va='center', fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Return (%)')
ax2.set_title('Total Returns Comparison', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Win rate and trade count
x = np.arange(len(strategies_names))
width = 0.35
ax3_twin = ax3.twinx()

bars1 = ax3.bar(x - width/2, results_df['win_rate'], width, 
               label='Win Rate (%)', color='green', alpha=0.7)
bars2 = ax3_twin.bar(x + width/2, results_df['total_trades'], width, 
                    label='Total Trades', color='blue', alpha=0.7)

ax3.set_xlabel('Strategy')
ax3.set_ylabel('Win Rate (%)', color='green')
ax3_twin.set_ylabel('Total Trades', color='blue')
ax3.set_title('Win Rate & Trade Frequency', fontweight='bold', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(strategies_names, rotation=45, ha='right')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Plot 4: Risk-adjusted metrics
ax4.scatter(results_df['max_drawdown'], results_df['total_return_pct'], 
           s=results_df['sharpe_ratio']*100, alpha=0.6, 
           c=results_df['win_rate'], cmap='RdYlGn', edgecolors='black', linewidth=2)

for idx, row in results_df.iterrows():
    ax4.annotate(row['Strategy'], 
                (row['max_drawdown'], row['total_return_pct']),
                fontsize=8, ha='center')

ax4.set_xlabel('Max Drawdown (%)')
ax4.set_ylabel('Total Return (%)')
ax4.set_title('Risk-Return Profile (size=Sharpe, color=WinRate)', fontweight='bold', fontsize=14)
ax4.grid(True, alpha=0.3)
cbar = plt.colorbar(ax4.collections[0], ax=ax4)
cbar.set_label('Win Rate (%)')

plt.tight_layout()
plt.savefig(f'{STOCK_SYMBOL}_backtest_results.png', dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to '{STOCK_SYMBOL}_backtest_results.png'")

plt.show()

print("\n" + "="*80)
print("✅ BACKTESTING COMPLETE!")
print("="*80)