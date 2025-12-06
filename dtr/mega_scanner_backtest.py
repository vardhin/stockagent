import yfinance as yf
import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import warnings
from colorama import Fore, Style, init
from datetime import datetime, timedelta

init(autoreset=True)
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
UNIVERSE = {
    "FINANCE": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS"],
    "IT":      ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"],
    "AUTO":    ["M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
    "METALS":  ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS", "ADANIENT.NS"],
    "ENERGY":  ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
    "FMCG":    ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS"],
    "PHARMA":  ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "INFRA":   ["LT.NS", "ULTRACEMCO.NS", "ADANIPORTS.NS", "GRASIM.NS"],
    "CONSUMER":["TITAN.NS", "ASIANPAINT.NS", "HAVELLS.NS", "TRENT.NS"]
}

# Backtest Config
BACKTEST_DAYS = 30
INITIAL_CAPITAL = 10000
MAX_POSITIONS = 3
COMMISSION = 0.0003  # 0.03%

# Position Sizing Strategy - Choose one:
# 1. "EQUAL" - Equal weight (default)
# 2. "KELLY" - Kelly Criterion (aggressive)
# 3. "RISK_BASED" - Based on ATR risk
# 4. "CONFIDENCE" - Based on model confidence
# 5. "VOLATILITY" - Inverse volatility weighting
POSITION_SIZING = "RISK_BASED"

# Risk Parameters
RISK_PER_TRADE = 0.02  # 2% risk per trade for RISK_BASED
KELLY_FRACTION = 0.25  # Use 25% of Kelly for safety

# Directories
os.makedirs("backtest_cache", exist_ok=True)
os.makedirs("backtest_models", exist_ok=True)

# ==========================================
# UTILITY: Check if Trading Day
# ==========================================
def is_trading_day(date):
    """Check if date is a weekday (Monday=0, Sunday=6)"""
    return date.weekday() < 5

# ==========================================
# POSITION SIZING STRATEGIES
# ==========================================
def calculate_position_size(strategy, capital, signal, model_history=None):
    """
    Calculate position size based on chosen strategy
    
    Args:
        strategy: Position sizing method
        capital: Available capital
        signal: Dict with Price, SL, TP, Score, ATR
        model_history: Dict with win_rate, avg_win, avg_loss for Kelly
    
    Returns:
        position_value: Amount to invest in this trade
    """
    if strategy == "EQUAL":
        # Equal weight across max positions
        return capital / MAX_POSITIONS
    
    elif strategy == "RISK_BASED":
        # Size based on risk per trade (most professional)
        # Risk = Capital * Risk% / (Entry - StopLoss)
        entry = signal['Price']
        sl = signal['SL']
        risk_per_share = abs(entry - sl)
        
        if risk_per_share == 0:
            return capital / MAX_POSITIONS
        
        # Calculate shares based on risk tolerance
        risk_amount = capital * RISK_PER_TRADE
        shares = int(risk_amount / risk_per_share)
        position_value = min(shares * entry, capital / MAX_POSITIONS * 1.5)  # Cap at 1.5x equal
        
        return position_value
    
    elif strategy == "CONFIDENCE":
        # Size based on model confidence (higher score = bigger position)
        # Score ranges from 0.55 to ~0.90
        score = signal['Score']
        confidence_factor = (score - 0.55) / 0.35  # Normalize to 0-1
        confidence_factor = max(0.5, min(1.5, confidence_factor * 2))  # Scale 0.5x to 1.5x
        
        base_size = capital / MAX_POSITIONS
        return base_size * confidence_factor
    
    elif strategy == "VOLATILITY":
        # Inverse volatility - allocate less to volatile stocks
        atr = signal['ATR']
        price = signal['Price']
        volatility = atr / price  # ATR as % of price
        
        # Lower volatility = higher allocation
        vol_factor = 1 / (1 + volatility * 10)  # Normalize
        vol_factor = max(0.5, min(1.5, vol_factor * 2))
        
        base_size = capital / MAX_POSITIONS
        return base_size * vol_factor
    
    elif strategy == "KELLY":
        # Kelly Criterion - requires historical win rate
        if model_history is None or model_history['total_trades'] < 10:
            # Not enough history, use equal weight
            return capital / MAX_POSITIONS
        
        win_rate = model_history['win_rate']
        avg_win_pct = model_history['avg_win_pct']
        avg_loss_pct = model_history['avg_loss_pct']
        
        if avg_loss_pct == 0:
            return capital / MAX_POSITIONS
        
        # Kelly Formula: f = (p*b - q) / b
        # where p=win%, q=loss%, b=avg_win/avg_loss
        b = avg_win_pct / avg_loss_pct
        kelly_pct = (win_rate * b - (1 - win_rate)) / b
        
        # Apply Kelly fraction for safety
        kelly_pct = max(0, kelly_pct) * KELLY_FRACTION
        kelly_pct = min(kelly_pct, 1.0 / MAX_POSITIONS * 1.5)  # Cap at 1.5x equal
        
        return capital * kelly_pct
    
    else:
        return capital / MAX_POSITIONS

# ==========================================
# 1. DATA FETCHING
# ==========================================
def get_data_for_backtest(ticker, end_date):
    """Fetch data up to end_date"""
    try:
        data = yf.download(ticker, end=end_date, period="max", interval="1d", progress=False)
        
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            return pd.DataFrame()

        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'ds'}, inplace=True)
        data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)
        return data
        
    except Exception:
        return pd.DataFrame()

def get_test_data(ticker, start_date, end_date):
    """Fetch 30m data for testing period"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval="30m", progress=False)
        
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            return pd.DataFrame()

        data.reset_index(inplace=True)
        data.rename(columns={'Datetime': 'ds'}, inplace=True)
        data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)
        return data
        
    except Exception:
        return pd.DataFrame()

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def get_features(df, interval):
    if df.empty or len(df) < 50: 
        return pd.DataFrame()
    df = df.copy()
    
    try:
        if interval == "1d":
            df['EMA_Fast'] = df['Close'].ewm(span=20).mean()
            df['EMA_Slow'] = df['Close'].ewm(span=200).mean()
        else:
            df['EMA_Fast'] = df['Close'].ewm(span=10).mean()
            df['EMA_Slow'] = df['Close'].ewm(span=50).mean()
            
        df['Trend_Dist'] = (df['Close'] - df['EMA_Slow']) / df['EMA_Slow']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        high_low = df['High'] - df['Low']
        df['ATR'] = high_low.rolling(14).mean()
        
        for lag in [1, 2, 3]:
            df[f'Ret_{lag}'] = df['Close'].pct_change(lag)
            
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# ==========================================
# 3. MODEL TRAINING
# ==========================================
def train_models(stock, train_end_date):
    """Train models on data before train_end_date"""
    clean_name = stock.replace(".NS", "")
    
    print(f"   🧠 Training {clean_name}...", end=" ")
    try:
        df = get_data_for_backtest(stock, train_end_date)
        df = get_features(df, "1d")
        
        if df.empty or len(df) < 100:
            print("❌ Insufficient data")
            return False
            
        df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.01).astype(int)
        df.dropna(inplace=True)
        
        if len(df) < 50:
            return False
            
        cols = [c for c in df.columns if c not in ['ds', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        dtrain = lgb.Dataset(df[cols], label=df['Target'])
        params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 20, 'verbose': -1, 'seed': 42}
        model = lgb.train(params, dtrain, num_boost_round=100)
        model.save_model(f"backtest_models/{clean_name}_daily.txt")
        
        print("✅")
        return True, cols
        
    except Exception as e:
        print(f"❌ {str(e)[:40]}")
        return False

# ==========================================
# 4. BACKTESTING ENGINE
# ==========================================
def run_backtest():
    print("="*80)
    print(f"📈 NIFTY ALPHA BACKTEST - {BACKTEST_DAYS} DAY SIMULATION")
    print(f"🎯 Position Sizing: {POSITION_SIZING}")
    print("="*80)
    
    today = datetime.now()
    test_end = today
    test_start = today - timedelta(days=BACKTEST_DAYS)
    train_end = test_start - timedelta(days=1)
    
    print(f"📅 Training Period: Up to {train_end.strftime('%Y-%m-%d')}")
    print(f"📅 Testing Period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,.2f}")
    print(f"📊 Max Positions: {MAX_POSITIONS}")
    if POSITION_SIZING == "RISK_BASED":
        print(f"⚠️ Risk Per Trade: {RISK_PER_TRADE*100:.1f}%")
    print()
    
    # Step 1: Train Models
    print("="*80)
    print("PHASE 1: Training Models")
    print("="*80)
    
    all_stocks = []
    trained_stocks = []
    feature_cols = None
    
    for sector, stocks in UNIVERSE.items():
        for s in stocks:
            all_stocks.append((s, sector))
            result = train_models(s, train_end)
            if result and result != False:
                trained_stocks.append((s, sector))
                if feature_cols is None and len(result) == 2:
                    feature_cols = result[1]
    
    print(f"\n✅ Trained {len(trained_stocks)}/{len(all_stocks)} models successfully\n")
    
    if not trained_stocks or feature_cols is None:
        print("❌ No models trained. Exiting.")
        return
    
    # Step 2: Trading Simulation
    print("="*80)
    print("PHASE 2: Generating Signals & Trading")
    print("="*80)
    
    capital = INITIAL_CAPITAL
    positions = {}
    trades = []
    model_history = {'total_trades': 0, 'wins': 0, 'win_rate': 0.5, 'avg_win_pct': 0.05, 'avg_loss_pct': 0.03}
    
    current_date = test_start
    trading_days = 0
    weekend_days = 0
    
    while current_date <= test_end:
        if not is_trading_day(current_date):
            print(f"\n📅 {current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A')})")
            print(f"   💤 Market Closed - Weekend")
            weekend_days += 1
            current_date += timedelta(days=1)
            continue
        
        trading_days += 1
        print(f"\n📅 Date: {current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A')}) - Trading Day #{trading_days}")
        
        # Get signals
        daily_signals = []
        
        for stock, sector in trained_stocks:
            clean_name = stock.replace(".NS", "")
            
            try:
                model = lgb.Booster(model_file=f"backtest_models/{clean_name}_daily.txt")
                df = get_data_for_backtest(stock, current_date)
                df = get_features(df, "1d")
                
                if df.empty or len(df) < 10:
                    continue
                
                latest = df.iloc[[-1]]
                prob = model.predict(latest[feature_cols])[0]
                
                if prob > 0.55:
                    price = latest['Close'].values[0]
                    atr = latest['ATR'].values[0]
                    
                    daily_signals.append({
                        'Stock': stock,
                        'Name': clean_name,
                        'Sector': sector,
                        'Score': prob,
                        'Price': price,
                        'ATR': atr,
                        'SL': price - (atr * 1.5),
                        'TP': price + (atr * 3)
                    })
                    
            except Exception:
                continue
        
        daily_signals.sort(key=lambda x: x['Score'], reverse=True)
        print(f"   📊 Found {len(daily_signals)} signals")
        
        # Check exits
        for stock in list(positions.keys()):
            pos = positions[stock]
            df_30m = get_test_data(stock, current_date, current_date + timedelta(days=1))
            
            if df_30m.empty:
                continue
            
            for idx, row in df_30m.iterrows():
                low_price = row['Low']
                high_price = row['High']
                
                if low_price <= pos['sl']:
                    exit_price = pos['sl']
                    pnl = (exit_price - pos['entry']) * pos['qty']
                    pnl_pct = ((exit_price / pos['entry']) - 1) * 100
                    capital += (pos['entry'] * pos['qty']) + pnl - (exit_price * pos['qty'] * COMMISSION)
                    
                    trades.append({
                        'Stock': stock,
                        'Entry': pos['entry'],
                        'Exit': exit_price,
                        'PnL': pnl,
                        'PnL%': pnl_pct,
                        'Type': 'SL Hit',
                        'Date': row['ds'],
                        'Position_Size': pos['position_value']
                    })
                    
                    # Update model history
                    model_history['total_trades'] += 1
                    if pnl > 0:
                        model_history['wins'] += 1
                    
                    print(f"   ❌ {stock}: SL Hit at ₹{exit_price:.2f} | PnL: ₹{pnl:.2f} ({pnl_pct:.2f}%) | Size: ₹{pos['position_value']:.0f}")
                    del positions[stock]
                    break
                
                elif high_price >= pos['tp']:
                    exit_price = pos['tp']
                    pnl = (exit_price - pos['entry']) * pos['qty']
                    pnl_pct = ((exit_price / pos['entry']) - 1) * 100
                    capital += (pos['entry'] * pos['qty']) + pnl - (exit_price * pos['qty'] * COMMISSION)
                    
                    trades.append({
                        'Stock': stock,
                        'Entry': pos['entry'],
                        'Exit': exit_price,
                        'PnL': pnl,
                        'PnL%': pnl_pct,
                        'Type': 'TP Hit',
                        'Date': row['ds'],
                        'Position_Size': pos['position_value']
                    })
                    
                    model_history['total_trades'] += 1
                    model_history['wins'] += 1
                    
                    print(f"   ✅ {stock}: TP Hit at ₹{exit_price:.2f} | PnL: ₹{pnl:.2f} ({pnl_pct:.2f}%) | Size: ₹{pos['position_value']:.0f}")
                    del positions[stock]
                    break
        
        # Update model statistics
        if model_history['total_trades'] > 0:
            model_history['win_rate'] = model_history['wins'] / model_history['total_trades']
            winning = [t for t in trades if t['PnL'] > 0]
            losing = [t for t in trades if t['PnL'] <= 0]
            if winning:
                model_history['avg_win_pct'] = sum(t['PnL%'] for t in winning) / len(winning) / 100
            if losing:
                model_history['avg_loss_pct'] = abs(sum(t['PnL%'] for t in losing) / len(losing)) / 100
        
        # Enter new positions
        available_slots = MAX_POSITIONS - len(positions)
        
        if available_slots > 0 and daily_signals:
            for signal in daily_signals[:available_slots]:
                stock = signal['Stock']
                
                if stock in positions:
                    continue
                
                # Calculate intelligent position size
                position_value = calculate_position_size(POSITION_SIZING, capital, signal, model_history)
                qty = int(position_value / signal['Price'])
                
                if qty == 0:
                    continue
                
                cost = signal['Price'] * qty * (1 + COMMISSION)
                
                if cost <= capital:
                    positions[stock] = {
                        'entry': signal['Price'],
                        'qty': qty,
                        'sl': signal['SL'],
                        'tp': signal['TP'],
                        'position_value': position_value
                    }
                    capital -= cost
                    
                    print(f"   🟢 ENTRY: {signal['Name']} @ ₹{signal['Price']:.2f} | Qty: {qty} | Size: ₹{position_value:.0f} | Score: {signal['Score']:.2f}")
        
        print(f"   💰 Capital: ₹{capital:,.2f} | Active Positions: {len(positions)}")
        current_date += timedelta(days=1)
    
    # Close remaining positions
    print("\n" + "="*80)
    print("PHASE 3: Closing Remaining Positions")
    print("="*80)
    
    for stock, pos in positions.items():
        df_final = get_data_for_backtest(stock, test_end)
        if df_final.empty:
            continue
            
        exit_price = df_final['Close'].iloc[-1]
        pnl = (exit_price - pos['entry']) * pos['qty']
        pnl_pct = ((exit_price / pos['entry']) - 1) * 100
        capital += (exit_price * pos['qty']) * (1 - COMMISSION)
        
        trades.append({
            'Stock': stock,
            'Entry': pos['entry'],
            'Exit': exit_price,
            'PnL': pnl,
            'PnL%': pnl_pct,
            'Type': 'Market Close',
            'Date': test_end,
            'Position_Size': pos['position_value']
        })
        
        print(f"   🔵 {stock}: Market Close at ₹{exit_price:.2f} | PnL: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
    
    # Final Report
    print("\n" + "="*80)
    print("📊 BACKTEST RESULTS")
    print("="*80)
    
    total_pnl = capital - INITIAL_CAPITAL
    total_return = (capital / INITIAL_CAPITAL - 1) * 100
    
    winning_trades = [t for t in trades if t['PnL'] > 0]
    losing_trades = [t for t in trades if t['PnL'] <= 0]
    
    print(f"📅 Period: {BACKTEST_DAYS} calendar days ({trading_days} trading days, {weekend_days} weekend days)")
    print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,.2f}")
    print(f"💰 Final Capital:   ₹{capital:,.2f}")
    print(f"{'📈' if total_pnl > 0 else '📉'} Total P&L:      ₹{total_pnl:,.2f} ({total_return:+.2f}%)")
    print(f"\n📊 Trade Statistics:")
    print(f"   Total Trades:    {len(trades)}")
    print(f"   Winning Trades:  {len(winning_trades)} ({len(winning_trades)/len(trades)*100 if trades else 0:.1f}%)")
    print(f"   Losing Trades:   {len(losing_trades)}")
    
    if winning_trades:
        avg_win = sum(t['PnL'] for t in winning_trades) / len(winning_trades)
        print(f"   Avg Win:         ₹{avg_win:,.2f}")
    
    if losing_trades:
        avg_loss = sum(t['PnL'] for t in losing_trades) / len(losing_trades)
        print(f"   Avg Loss:        ₹{avg_loss:,.2f}")
    
    if winning_trades and losing_trades:
        profit_factor = abs(sum(t['PnL'] for t in winning_trades) / sum(t['PnL'] for t in losing_trades))
        print(f"   Profit Factor:   {profit_factor:.2f}")
    
    print("\n📋 Trade Log:")
    print(f"{'Date':<12} | {'Stock':<15} | {'Size':<10} | {'Entry':<10} | {'Exit':<10} | {'P&L':<12} | {'P&L%':<8} | {'Type':<12}")
    print("-" * 110)
    for t in trades:
        color = Fore.GREEN if t['PnL'] > 0 else Fore.RED
        date_str = t['Date'].strftime('%Y-%m-%d') if isinstance(t['Date'], datetime) else str(t['Date'])[:10]
        print(f"{date_str:<12} | {t['Stock']:<15} | ₹{t['Position_Size']:<9.0f} | ₹{t['Entry']:<9.2f} | ₹{t['Exit']:<9.2f} | {color}₹{t['PnL']:<11.2f}{Style.RESET_ALL} | {color}{t['PnL%']:>6.2f}%{Style.RESET_ALL} | {t['Type']:<12}")
    
    print("="*110)

if __name__ == "__main__":
    run_backtest()