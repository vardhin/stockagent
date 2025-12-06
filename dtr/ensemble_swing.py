import yfinance as yf
import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import warnings
import time

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
SECTOR_STOCKS = [
    "TATASTEEL.NS", 
    "JSWSTEEL.NS", 
    "HINDALCO.NS", 
    "VEDL.NS", 
    "SAIL.NS"
]

# Capital Allocation
TOTAL_CAPITAL = 500000      # ₹5 Lakh Total
PER_TRADE_CAP = 100000      # ₹1 Lakh per stock max
COMMISSION = 0.0003

# Directories
os.makedirs("data_cache", exist_ok=True)
os.makedirs("model_vault", exist_ok=True)

# ==========================================
# 1. SMART CACHING SYSTEM
# ==========================================
def get_data_cached(ticker, period, interval, force_update=False):
    """
    Checks local cache first. If missing or force_update=True, downloads from Yahoo.
    """
    clean_ticker = ticker.replace(".NS", "")
    file_path = f"data_cache/{clean_ticker}_{interval}.csv"
    
    # 1. Try Load from Disk
    if os.path.exists(file_path) and not force_update:
        # print(f"   📂 Loading {ticker} {interval} from cache...")
        try:
            df = pd.read_csv(file_path)
            df['ds'] = pd.to_datetime(df['ds'])
            return df
        except Exception:
            print("   ⚠️ Cache corrupted. Redownloading...")

    # 2. Download (If cache miss)
    print(f"   ⬇️ Downloading {ticker} ({interval})...")
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        data.reset_index(inplace=True)
        date_col = 'Date' if 'Date' in data.columns else 'Datetime'
        data.rename(columns={date_col: 'ds'}, inplace=True)
        data['ds'] = data['ds'].dt.tz_localize(None)
        
        # 3. Save to Disk
        data.to_csv(file_path, index=False)
        return data
        
    except Exception as e:
        print(f"   ❌ Download failed for {ticker}: {e}")
        return pd.DataFrame()

# ==========================================
# 2. FEATURE ENGINEERING (Universal)
# ==========================================
def get_features(df, interval):
    df = df.copy()
    
    # Trend
    if interval == "1d":
        df['EMA_Fast'] = df['Close'].ewm(span=20).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=200).mean()
    else:
        df['EMA_Fast'] = df['Close'].ewm(span=10).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=50).mean()
        
    df['Trend_Dist'] = (df['Close'] - df['EMA_Slow']) / df['EMA_Slow']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    
    # ATR
    high_low = df['High'] - df['Low']
    df['ATR'] = high_low.rolling(14).mean()
    
    # Lags
    for lag in [1, 2, 3]:
        df[f'Ret_{lag}'] = df['Close'].pct_change(lag)
        
    df.dropna(inplace=True)
    return df

# ==========================================
# 3. MODEL TRAINING (Per Stock)
# ==========================================
def train_sector_models(force_update=False):
    print("="*60)
    print("🏭 TRAINING SECTOR MODELS")
    print("="*60)
    
    for stock in SECTOR_STOCKS:
        clean_name = stock.replace(".NS", "")
        print(f"\n🏗️ Processing {clean_name}...")
        
        # --- A. DAILY MODEL (10 Years) ---
        df_d = get_data_cached(stock, "10y", "1d", force_update)
        df_d = get_features(df_d, "1d")
        
        # Target: 1% gain in 3 days
        df_d['Target'] = (df_d['Close'].shift(-3) / df_d['Close'] - 1 > 0.01).astype(int)
        df_d.dropna(inplace=True)
        
        cols = [c for c in df_d.columns if c not in ['ds', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        # Train & Save Daily
        split = len(df_d) - 60
        dtrain = lgb.Dataset(df_d.iloc[:split][cols], label=df_d.iloc[:split]['Target'])
        params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 20, 'verbose': -1, 'seed': 42}
        model_d = lgb.train(params, dtrain, num_boost_round=200)
        model_d.save_model(f"model_vault/{clean_name}_daily.txt")
        
        # --- B. HOURLY MODEL (2 Years) ---
        df_h = get_data_cached(stock, "730d", "1h", force_update)
        df_h = get_features(df_h, "1h")
        
        # Target: 0.5% gain in 4 hours
        df_h['Target'] = (df_h['Close'].shift(-4) / df_h['Close'] - 1 > 0.005).astype(int)
        df_h.dropna(inplace=True)
        
        # Train & Save Hourly
        split = len(df_h) - (60*7)
        dtrain = lgb.Dataset(df_h.iloc[:split][cols], label=df_h.iloc[:split]['Target'])
        model_h = lgb.train(params, dtrain, num_boost_round=200)
        model_h.save_model(f"model_vault/{clean_name}_hourly.txt")
        
        # --- C. 30M MODEL (60 Days) ---
        df_m = get_data_cached(stock, "59d", "30m", force_update)
        df_m = get_features(df_m, "30m")
        
        # Target: 0.2% gain in 1 hour
        df_m['Target'] = (df_m['Close'].shift(-2) / df_m['Close'] - 1 > 0.002).astype(int)
        df_m.dropna(inplace=True)
        
        # Train & Save 30m
        split = int(len(df_m) * 0.8)
        dtrain = lgb.Dataset(df_m.iloc[:split][cols], label=df_m.iloc[:split]['Target'])
        model_m = lgb.train(params, dtrain, num_boost_round=200)
        model_m.save_model(f"model_vault/{clean_name}_30m.txt")
        
    print("\n✅ All Sector Models Trained & Saved.")
    return cols # Return feature columns for prediction

# ==========================================
# 4. PORTFOLIO SIMULATION
# ==========================================
def run_sector_simulation(feature_cols):
    print("\n" + "="*60)
    print("🚀 RUNNING PORTFOLIO SIMULATION (Last 2 Weeks)")
    print("="*60)
    
    total_pnl = 0
    total_trades = 0
    all_trades_log = []
    
    for stock in SECTOR_STOCKS:
        clean_name = stock.replace(".NS", "")
        
        # Load Data (Evaluation Phase)
        df_30m = get_data_cached(stock, "59d", "30m")
        df_30m = get_features(df_30m, "30m")
        
        # Load Models
        model_d = lgb.Booster(model_file=f"model_vault/{clean_name}_daily.txt")
        model_h = lgb.Booster(model_file=f"model_vault/{clean_name}_hourly.txt")
        model_m = lgb.Booster(model_file=f"model_vault/{clean_name}_30m.txt")
        
        # --- ALIGN SIGNALS ---
        # Fetch Context
        raw_d = get_data_cached(stock, "10y", "1d")
        raw_d = get_features(raw_d, "1d")
        raw_d['Prob_Daily'] = model_d.predict(raw_d[feature_cols])
        raw_d['Date_Only'] = raw_d['ds'].dt.date
        
        raw_h = get_data_cached(stock, "730d", "1h")
        raw_h = get_features(raw_h, "1h")
        raw_h['Prob_Hourly'] = model_h.predict(raw_h[feature_cols])
        raw_h['Join_Time'] = raw_h['ds'].dt.floor('h')
        
        # Predict 30m
        df_30m['Prob_30m'] = model_m.predict(df_30m[feature_cols])
        df_30m['Date_Only'] = df_30m['ds'].dt.date
        df_30m['Join_Time'] = df_30m['ds'].dt.floor('h')
        
        # Merge
        sim_df = pd.merge(df_30m, raw_d[['Date_Only', 'Prob_Daily']], on='Date_Only', how='left')
        sim_df = pd.merge(sim_df, raw_h[['Join_Time', 'Prob_Hourly']], on='Join_Time', how='left')
        sim_df.fillna(method='ffill', inplace=True)
        sim_df.dropna(inplace=True)
        
        # --- SIMULATE TRADES ---
        # Only test last 30% of data (approx 2 weeks)
        sim_df = sim_df.tail(int(len(sim_df)*0.3))
        
        balance = PER_TRADE_CAP
        stock_trades = 0
        stock_pnl = 0
        in_position = False
        
        for i in range(len(sim_df)-12):
            row = sim_df.iloc[i]
            
            # THE COUNCIL VOTE
            vote_daily = row['Prob_Daily'] > 0.50
            vote_hourly = row['Prob_Hourly'] > 0.50
            vote_30m = row['Prob_30m'] > 0.65
            
            if vote_daily and vote_hourly and vote_30m and not in_position:
                entry_price = row['Close']
                qty = int(balance / entry_price)
                
                atr = row['ATR']
                tp = entry_price + (atr * 3)
                sl = entry_price - (atr * 1.5)
                
                exit_price = entry_price # Fallback
                
                # Check outcome (next 6 hours)
                for j in range(1, 13):
                    curr = sim_df.iloc[i+j]
                    if curr['Low'] < sl:
                        exit_price = sl
                        break
                    if curr['High'] > tp:
                        exit_price = tp
                        break
                    if j == 12: exit_price = curr['Close']
                
                pnl = (exit_price - entry_price) * qty
                cost = (entry_price * qty * COMMISSION) + (exit_price * qty * COMMISSION)
                net_pnl = pnl - cost
                
                stock_pnl += net_pnl
                stock_trades += 1
                all_trades_log.append(f"{clean_name}: {net_pnl:.2f}")
                in_position = False # Reset immediately for simulation simplicity (or i+=j to skip)
        
        print(f"   📊 {clean_name}: {stock_trades} Trades | PnL: ₹{stock_pnl:.2f}")
        total_pnl += stock_pnl
        total_trades += stock_trades

    print("\n" + "="*60)
    print(f"🏆 SECTOR PORTFOLIO RESULT")
    print("="*60)
    print(f"💰 Starting Capital: ₹{TOTAL_CAPITAL}")
    print(f"💰 Net Profit:       ₹{total_pnl:.2f}")
    print(f"📈 ROI:              {(total_pnl/TOTAL_CAPITAL)*100:.2f}%")
    print(f"🔢 Total Trades:     {total_trades}")
    print("="*60)
    if total_trades < 5:
        print("ℹ️ Note: Low trade count indicates Sector-Wide Bearishness/Consolidation.")

if __name__ == "__main__":
    # Set force_update=True if you want to redownload everything fresh
    feature_cols = train_sector_models(force_update=False)
    run_sector_simulation(feature_cols)