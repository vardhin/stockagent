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
INITIAL_CAPITAL = 10000
MAX_POSITIONS = 3  # Max concurrent positions
POSITION_SIZE = INITIAL_CAPITAL / MAX_POSITIONS
COMMISSION = 0.0003  # 0.03%

# Directories
os.makedirs("backtest_cache", exist_ok=True)
os.makedirs("backtest_models", exist_ok=True)

# ==========================================
# 1. DATA FETCHING
# ==========================================
def get_data_for_backtest(ticker, end_date):
    """Fetch data up to end_date (2 weeks ago from now)"""
    clean_ticker = ticker.replace(".NS", "")
    
    try:
        # Get enough historical data
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
    """Fetch 30m data for testing period (last 2 weeks)"""
    try:
        # Get 30m data for last 2 weeks
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
        # Trend
        if interval == "1d":
            df['EMA_Fast'] = df['Close'].ewm(span=20).mean()
            df['EMA_Slow'] = df['Close'].ewm(span=200).mean()
        else:
            df['EMA_Fast'] = df['Close'].ewm(span=10).mean()
            df['EMA_Slow'] = df['Close'].ewm(span=50).mean()
            
        df['Trend_Dist'] = (df['Close'] - df['EMA_Slow']) / df['EMA_Slow']
        
        # RSI & ATR
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        high_low = df['High'] - df['Low']
        df['ATR'] = high_low.rolling(14).mean()
        
        # Lags
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
        # Train Daily Model
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
    print(f"📈 NIFTY ALPHA BACKTEST - 2 WEEK SIMULATION")
    print("="*80)
    
    # Define dates
    today = datetime.now()
    test_end = today
    test_start = today - timedelta(days=14)
    train_end = test_start - timedelta(days=1)
    
    print(f"📅 Training Period: Up to {train_end.strftime('%Y-%m-%d')}")
    print(f"📅 Testing Period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,.2f}")
    print(f"📊 Max Positions: {MAX_POSITIONS}")
    print(f"💵 Position Size: ₹{POSITION_SIZE:,.2f}\n")
    
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
    
    # Step 2: Get Signals Every Day
    print("="*80)
    print("PHASE 2: Generating Signals & Trading")
    print("="*80)
    
    capital = INITIAL_CAPITAL
    positions = {}  # {stock: {'entry': price, 'qty': qty, 'sl': sl, 'tp': tp}}
    trades = []
    
    # Generate signals for each trading day
    test_dates = pd.date_range(start=test_start, end=test_end, freq='D')
    
    for current_date in test_dates:
        print(f"\n📅 Date: {current_date.strftime('%Y-%m-%d')}")
        
        # Get signals for this day
        daily_signals = []
        
        for stock, sector in trained_stocks:
            clean_name = stock.replace(".NS", "")
            
            try:
                # Load model
                model = lgb.Booster(model_file=f"backtest_models/{clean_name}_daily.txt")
                
                # Get latest data up to current_date
                df = get_data_for_backtest(stock, current_date)
                df = get_features(df, "1d")
                
                if df.empty or len(df) < 10:
                    continue
                
                # Get latest features
                latest = df.iloc[[-1]]
                
                # Predict
                prob = model.predict(latest[feature_cols])[0]
                
                if prob > 0.55:  # Signal threshold
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
        
        # Sort by score and take top signals
        daily_signals.sort(key=lambda x: x['Score'], reverse=True)
        
        print(f"   📊 Found {len(daily_signals)} signals")
        
        # Check exits first (using 30m data)
        for stock in list(positions.keys()):
            pos = positions[stock]
            
            # Get 30m data for this day
            df_30m = get_test_data(stock, current_date, current_date + timedelta(days=1))
            
            if df_30m.empty:
                continue
            
            # Check each candle for SL/TP
            for idx, row in df_30m.iterrows():
                low_price = row['Low']
                high_price = row['High']
                
                # Check Stop Loss
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
                        'Date': row['ds']
                    })
                    
                    print(f"   ❌ {stock}: SL Hit at ₹{exit_price:.2f} | PnL: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
                    del positions[stock]
                    break
                
                # Check Target
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
                        'Date': row['ds']
                    })
                    
                    print(f"   ✅ {stock}: TP Hit at ₹{exit_price:.2f} | PnL: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
                    del positions[stock]
                    break
        
        # Enter new positions if we have space
        available_slots = MAX_POSITIONS - len(positions)
        
        if available_slots > 0 and daily_signals:
            for signal in daily_signals[:available_slots]:
                stock = signal['Stock']
                
                # Skip if already in position
                if stock in positions:
                    continue
                
                # Calculate position size
                qty = int(POSITION_SIZE / signal['Price'])
                if qty == 0:
                    continue
                
                cost = signal['Price'] * qty * (1 + COMMISSION)
                
                if cost <= capital:
                    positions[stock] = {
                        'entry': signal['Price'],
                        'qty': qty,
                        'sl': signal['SL'],
                        'tp': signal['TP']
                    }
                    capital -= cost
                    
                    print(f"   🟢 ENTRY: {signal['Name']} @ ₹{signal['Price']:.2f} | Qty: {qty} | Score: {signal['Score']:.2f}")
        
        print(f"   💰 Capital: ₹{capital:,.2f} | Active Positions: {len(positions)}")
    
    # Close remaining positions at market
    print("\n" + "="*80)
    print("PHASE 3: Closing Remaining Positions")
    print("="*80)
    
    for stock, pos in positions.items():
        # Get final price
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
            'Date': test_end
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
    
    print("\n📋 Trade Log:")
    print(f"{'Stock':<15} | {'Entry':<10} | {'Exit':<10} | {'P&L':<12} | {'P&L%':<8} | {'Type':<12}")
    print("-" * 80)
    for t in trades:
        color = Fore.GREEN if t['PnL'] > 0 else Fore.RED
        print(f"{t['Stock']:<15} | ₹{t['Entry']:<9.2f} | ₹{t['Exit']:<9.2f} | {color}₹{t['PnL']:<11.2f}{Style.RESET_ALL} | {color}{t['PnL%']:>6.2f}%{Style.RESET_ALL} | {t['Type']:<12}")
    
    print("="*80)

if __name__ == "__main__":
    run_backtest()