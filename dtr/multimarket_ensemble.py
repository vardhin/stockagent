import yfinance as yf
import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import warnings
import time

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION: THE MARKET UNIVERSE
# ==========================================
SECTOR_MAP = {
    "METALS": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "SAIL.NS"],
    "BANKS":  ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "AUTO":   ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS"],
    "IT":     ["INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS"],
    "ENERGY": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS"]
}

# Capital Allocation
TOTAL_CAPITAL = 10000     # ₹10 Thousand Total
PER_STOCK_CAP = 5000       # Allocation per stock
COMMISSION = 0.0003

# Directories
os.makedirs("data_cache", exist_ok=True)
os.makedirs("model_vault", exist_ok=True)

# ==========================================
# 1. ROBUST DATA FETCHER (With Error Handling)
# ==========================================
def get_data_cached(ticker, period, interval, force_update=False):
    clean_ticker = ticker.replace(".NS", "")
    file_path = f"data_cache/{clean_ticker}_{interval}.csv"
    
    # 1. Try Load from Disk
    if os.path.exists(file_path) and not force_update:
        try:
            df = pd.read_csv(file_path)
            # Check if file is actually valid and not empty
            if len(df) > 10:
                df['ds'] = pd.to_datetime(df['ds'])
                return df
        except Exception:
            pass # If read fails, re-download

    # 2. Download from Yahoo
    print(f"   ⬇️ Downloading {ticker} ({interval})...")
    try:
        # Added threads=False to prevent API rate limits/errors
        data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            print(f"   ⚠️ Warning: No data found for {ticker} ({interval})")
            return pd.DataFrame()

        data.reset_index(inplace=True)
        date_col = 'Date' if 'Date' in data.columns else 'Datetime'
        data.rename(columns={date_col: 'ds'}, inplace=True)
        data['ds'] = data['ds'].dt.tz_localize(None)
        
        # Save only if valid
        if len(data) > 10:
            data.to_csv(file_path, index=False)
        return data
        
    except Exception as e:
        print(f"   ❌ Download Error {ticker}: {e}")
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
    except Exception as e:
        print(f"   ⚠️ Feature error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. MASSIVE MODEL TRAINING LOOP
# ==========================================
def train_market_models(force_update=False):
    print("="*60)
    print("🏭 TRAINING MARKET-WIDE MODELS")
    print("="*60)
    
    all_stocks = [s for sublist in SECTOR_MAP.values() for s in sublist]
    feature_cols = []
    
    for stock in all_stocks:
        clean_name = stock.replace(".NS", "")
        print(f"🏗️ Processing {clean_name}...")
        
        try:
            # --- DAILY MODEL ---
            df_d = get_data_cached(stock, "10y", "1d", force_update)
            df_d = get_features(df_d, "1d")
            
            # CRITICAL CHECK: Skip if data is missing/empty
            if df_d.empty or len(df_d) < 100:
                print(f"   ⏩ Skipping {clean_name} (Insufficient Daily Data)")
                continue

            df_d['Target'] = (df_d['Close'].shift(-3) / df_d['Close'] - 1 > 0.01).astype(int)
            df_d.dropna(inplace=True)
            
            cols = [c for c in df_d.columns if c not in ['ds', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            feature_cols = cols 
            
            dtrain = lgb.Dataset(df_d[cols], label=df_d['Target'])
            params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 20, 'verbose': -1, 'seed': 42}
            model_d = lgb.train(params, dtrain, num_boost_round=150)
            model_d.save_model(f"model_vault/{clean_name}_daily.txt")
            
            # --- HOURLY MODEL ---
            df_h = get_data_cached(stock, "730d", "1h", force_update)
            df_h = get_features(df_h, "1h")
            
            if df_h.empty or len(df_h) < 100:
                print(f"   ⏩ Skipping {clean_name} (Insufficient Hourly Data)")
                continue
                
            df_h['Target'] = (df_h['Close'].shift(-4) / df_h['Close'] - 1 > 0.005).astype(int)
            df_h.dropna(inplace=True)
            dtrain = lgb.Dataset(df_h[cols], label=df_h['Target'])
            model_h = lgb.train(params, dtrain, num_boost_round=150)
            model_h.save_model(f"model_vault/{clean_name}_hourly.txt")
            
            # --- 30M MODEL ---
            df_m = get_data_cached(stock, "59d", "30m", force_update)
            df_m = get_features(df_m, "30m")
            
            if df_m.empty or len(df_m) < 100:
                print(f"   ⏩ Skipping {clean_name} (Insufficient 30m Data)")
                continue
                
            df_m['Target'] = (df_m['Close'].shift(-2) / df_m['Close'] - 1 > 0.002).astype(int)
            df_m.dropna(inplace=True)
            dtrain = lgb.Dataset(df_m[cols], label=df_m['Target'])
            model_m = lgb.train(params, dtrain, num_boost_round=150)
            model_m.save_model(f"model_vault/{clean_name}_30m.txt")
            
        except Exception as e:
            print(f"   ⚠️ Unexpected error training {stock}: {e}")
            continue

    print("\n✅ All Valid Models Trained.")
    return feature_cols

# ==========================================
# 4. MULTI-SECTOR SIMULATION
# ==========================================
def run_market_simulation(feature_cols):
    print("\n" + "="*60)
    print("🚀 RUNNING MULTI-SECTOR SIMULATION (Last 2 Weeks)")
    print("="*60)
    
    total_pnl = 0
    total_trades = 0
    sector_performance = {}
    
    for sector, stocks in SECTOR_MAP.items():
        sector_pnl = 0
        sector_trades = 0
        print(f"\n📂 SECTOR: {sector}")
        
        for stock in stocks:
            clean_name = stock.replace(".NS", "")
            
            # Check if models exist (Skip if training failed)
            if not os.path.exists(f"model_vault/{clean_name}_30m.txt"):
                # print(f"   ⏩ Skipping {clean_name} (No Model Found)")
                continue

            try:
                # Load Models
                model_d = lgb.Booster(model_file=f"model_vault/{clean_name}_daily.txt")
                model_h = lgb.Booster(model_file=f"model_vault/{clean_name}_hourly.txt")
                model_m = lgb.Booster(model_file=f"model_vault/{clean_name}_30m.txt")
                
                # Fetch Data
                df_30m = get_data_cached(stock, "59d", "30m")
                df_30m = get_features(df_30m, "30m")
                
                raw_d = get_data_cached(stock, "10y", "1d")
                raw_d = get_features(raw_d, "1d")
                
                raw_h = get_data_cached(stock, "730d", "1h")
                raw_h = get_features(raw_h, "1h")
                
                # Predict
                raw_d['Prob_Daily'] = model_d.predict(raw_d[feature_cols])
                raw_d['Date_Only'] = raw_d['ds'].dt.date
                
                raw_h['Prob_Hourly'] = model_h.predict(raw_h[feature_cols])
                raw_h['Join_Time'] = raw_h['ds'].dt.floor('h')
                
                df_30m['Prob_30m'] = model_m.predict(df_30m[feature_cols])
                df_30m['Date_Only'] = df_30m['ds'].dt.date
                df_30m['Join_Time'] = df_30m['ds'].dt.floor('h')
                
                # Merge & Sync
                sim_df = pd.merge(df_30m, raw_d[['Date_Only', 'Prob_Daily']], on='Date_Only', how='left')
                sim_df = pd.merge(sim_df, raw_h[['Join_Time', 'Prob_Hourly']], on='Join_Time', how='left')
                sim_df.fillna(method='ffill', inplace=True)
                sim_df.dropna(inplace=True)
                
                # Test last 30%
                sim_df = sim_df.tail(int(len(sim_df)*0.3))
                
                stock_pnl = 0
                stock_trade_count = 0
                
                # TRADING LOOP
                i = 0
                while i < len(sim_df)-12:
                    row = sim_df.iloc[i]
                    
                    vote_daily = row['Prob_Daily'] > 0.50
                    vote_hourly = row['Prob_Hourly'] > 0.50
                    vote_30m = row['Prob_30m'] > 0.65
                    
                    if vote_daily and vote_hourly and vote_30m:
                        entry_price = row['Close']
                        qty = int(PER_STOCK_CAP / entry_price)
                        
                        atr = row['ATR']
                        tp = entry_price + (atr * 3)
                        sl = entry_price - (atr * 1.5)
                        
                        exit_price = entry_price
                        held = 0
                        
                        for j in range(1, 13):
                            curr = sim_df.iloc[i+j]
                            held = j
                            if curr['Low'] < sl:
                                exit_price = sl
                                break
                            if curr['High'] > tp:
                                exit_price = tp
                                break
                            if j == 12: exit_price = curr['Close']
                        
                        pnl = (exit_price - entry_price) * qty
                        cost = (entry_price * qty * COMMISSION * 2) 
                        net_pnl = pnl - cost
                        
                        stock_pnl += net_pnl
                        stock_trade_count += 1
                        i += held 
                    else:
                        i += 1
                
                print(f"   📊 {clean_name}: {stock_trade_count} Trades | PnL: ₹{stock_pnl:.2f}")
                sector_pnl += stock_pnl
                sector_trades += stock_trade_count
                
            except Exception as e:
                print(f"   ⚠️ Error processing {stock} simulation: {e}")
        
        sector_performance[sector] = {'PnL': sector_pnl, 'Trades': sector_trades}
        total_pnl += sector_pnl
        total_trades += sector_trades

    # FINAL REPORT
    print("\n" + "="*60)
    print(f"🏆 MARKET PERFORMANCE MATRIX")
    print("="*60)
    print(f"{'SECTOR':<15} | {'TRADES':<10} | {'PnL (₹)':<15}")
    print("-" * 45)
    for sec, data in sector_performance.items():
        print(f"{sec:<15} | {data['Trades']:<10} | ₹{data['PnL']:<15.2f}")
    print("-" * 45)
    print(f"{'TOTAL':<15} | {total_trades:<10} | ₹{total_pnl:<15.2f}")
    print("="*60)

if __name__ == "__main__":
    # If training failed previously, use force_update=False to try resume/cache
    feature_cols = train_market_models(force_update=False)
    if feature_cols:
        run_market_simulation(feature_cols)
    else:
        print("❌ Could not determine features. Training likely failed for all stocks.")