import yfinance as yf
import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ V6 CONFIGURATION (Hourly Swing)
# ==========================================
STOCK = "TATASTEEL.NS"
MODEL_PATH = "tata_hourly_swing.txt"
TIMEFRAME = "1h"
LOOKBACK_DAYS = 700  # 2 Years of data! (Huge advantage)
CAPITAL = 100000    
LEVERAGE = 3         # Lower leverage for Swing (Safety)
COMMISSION = 0.0003 

# STRATEGY SETTINGS
PROB_THRESH = 0.60   # Confidence Threshold
SL_PCT = 0.015       # 1.5% Stop Loss (Wider for Hourly)
TP_PCT = 0.040       # 4.0% Take Profit (Aiming for BIG moves)

# ==========================================
# 1. FEATURE ENGINEERING (Hourly Context)
# ==========================================
def get_features(df):
    df = df.copy()
    
    # --- A. Trend Indicators ---
    # EMA 50 (Short Trend) & EMA 200 (Major Trend)
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()
    df['Trend_Dist'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
    
    # --- B. Momentum ---
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    
    # --- C. Volatility ---
    # ATR (14)
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift()), 
                                     abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # --- D. Lag Features (Memory) ---
    # What happened 1 hour, 4 hours, and 1 day ago?
    for lag in [1, 4, 7]: # 7 hours is roughly 1 trading day
        df[f'Ret_{lag}'] = df['Close'].pct_change(lag)
        df[f'Vol_{lag}'] = df['Volume'].pct_change(lag)

    df.dropna(inplace=True)
    return df

# ==========================================
# 2. DATA UTILS
# ==========================================
def fetch_data():
    print(f"📥 Fetching last {LOOKBACK_DAYS} days of HOURLY data...")
    # Yahoo Finance 1h data limit is 730 days
    data = yf.download(STOCK, period=f"{LOOKBACK_DAYS}d", interval=TIMEFRAME, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    
    date_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
    data.rename(columns={date_col: 'ds'}, inplace=True)
    data['ds'] = data['ds'].dt.tz_localize(None)
    return data

# ==========================================
# 3. BACKTEST (The Recovery Plan)
# ==========================================
def backtest_v6():
    df = fetch_data()
    df = get_features(df)
    
    # Test on LAST 60 DAYS (Recent Market)
    # Train on previous ~600 DAYS
    split_date = df['ds'].max() - timedelta(days=60)
    
    train_df = df[df['ds'] < split_date]
    test_df = df[df['ds'] >= split_date].copy()
    
    print(f"🧪 Training on {len(train_df)} hours (approx {len(train_df)/7:.0f} days)...")
    print(f"🧪 Testing on {len(test_df)} hours (Last 60 days)...")
    
    # TARGET: Predict return over NEXT 4 HOURS
    # 1 (Buy) if price goes up > 0.5% in 4 hours
    # 0 (Sell/Hold) otherwise
    future_ret = train_df['Close'].shift(-4) / train_df['Close'] - 1
    train_df['Target'] = (future_ret > 0.005).astype(int)
    train_df.dropna(inplace=True)
    
    cols = [c for c in train_df.columns if c not in ['ds', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    # Train Strong Model (More Trees because we have more data)
    dtrain = lgb.Dataset(train_df[cols], label=train_df['Target'])
    params = {
        'objective': 'binary', 
        'metric': 'auc', 
        'learning_rate': 0.02, # Slower learning for stability
        'num_leaves': 31, 
        'verbose': -1, 
        'seed': 42
    }
    model = lgb.train(params, dtrain, num_boost_round=1000)
    
    # Predict
    test_df['Prob'] = model.predict(test_df[cols])
    
    balance = CAPITAL
    trades = []
    
    print("🔄 Running Hourly Swing Simulation...")
    
    i = 0
    while i < len(test_df) - 10:
        row = test_df.iloc[i]
        
        # ENTRY LOGIC:
        # 1. AI High Confidence
        # 2. RSI is not Overbought (>70) - Avoid buying tops
        ai_buy = row['Prob'] > PROB_THRESH
        rsi_ok = row['RSI'] < 70
        
        if ai_buy and rsi_ok:
            entry_price = row['Close']
            qty = int((balance * LEVERAGE) / entry_price)
            
            # Risk Management
            sl = entry_price * (1 - SL_PCT)
            tp = entry_price * (1 + TP_PCT)
            
            exit_price = entry_price
            holding_hours = 0
            
            # Check next 24 Hours (Swing)
            for j in range(1, 25): 
                if i + j >= len(test_df): break
                curr = test_df.iloc[i+j]
                
                # Check Price Limits
                if curr['Low'] <= sl: 
                    exit_price = sl
                    break
                if curr['High'] >= tp: 
                    exit_price = tp
                    break
                
                # Time Stop: If we held for 24 hours and nothing happened, exit
                if j == 24:
                    exit_price = curr['Close']
            
            # PnL
            gross = (exit_price - entry_price) * qty
            comm = (entry_price * qty * COMMISSION) + (exit_price * qty * COMMISSION)
            net = gross - comm
            
            balance += net
            trades.append(net)
            
            # Skip the holding period
            i += j
        else:
            i += 1
            
    # REPORT
    wins = [t for t in trades if t > 0]
    
    print("\n" + "="*40)
    print(f"📊 V6 RESULTS (Hourly Swing)")
    print("="*40)
    print(f"💰 Final Balance:   ₹{int(balance)}")
    print(f"📈 Net Profit:      ₹{int(balance - CAPITAL)} ({(balance-CAPITAL)/CAPITAL*100:.2f}%)")
    print(f"Total Trades:       {len(trades)}")
    print(f"Win Rate:           {len(wins)/len(trades)*100:.1f}%" if trades else "0%")
    print("="*40)

if __name__ == "__main__":
    backtest_v6()