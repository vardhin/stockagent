import yfinance as yf
import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import warnings
from colorama import Fore, Style, init

init(autoreset=True)
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ THE NIFTY 100 UNIVERSE (Liquid Only)
# ==========================================
# Grouped for easier management
UNIVERSE = {
    "FINANCE": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS"],
    "IT":      ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"],
    "AUTO":    ["M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],  # Fixed to TATAMOTORS
    "METALS":  ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS", "ADANIENT.NS"],
    "ENERGY":  ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
    "FMCG":    ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS"],
    "PHARMA":  ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "INFRA":   ["LT.NS", "ULTRACEMCO.NS", "ADANIPORTS.NS", "GRASIM.NS"],
    "CONSUMER":["TITAN.NS", "ASIANPAINT.NS", "HAVELLS.NS", "TRENT.NS"]
}

# Directories
os.makedirs("data_cache", exist_ok=True)
os.makedirs("model_vault", exist_ok=True)

# ==========================================
# 1. SMART CACHING (Crucial for 50+ stocks)
# ==========================================
def get_data_cached(ticker, period, interval):
    clean_ticker = ticker.replace(".NS", "")
    file_path = f"data_cache/{clean_ticker}_{interval}.csv"
    
    # Try Load
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if len(df) > 10:
                df['ds'] = pd.to_datetime(df['ds'])
                # More lenient cache check - allow 7 days for daily data
                last_date = df['ds'].max()
                cache_days = 7 if interval == "1d" else 2
                if (pd.Timestamp.now() - last_date).days < cache_days:
                    return df
        except Exception: 
            pass

    # Download with error handling
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        
        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        # Check if empty
        if data.empty:
            # Silent fail - no need to spam console
            return pd.DataFrame()

        data.reset_index(inplace=True)
        date_col = 'Date' if 'Date' in data.columns else 'Datetime'
        data.rename(columns={date_col: 'ds'}, inplace=True)
        data['ds'] = data['ds'].dt.tz_localize(None)
        
        if len(data) > 10: 
            data.to_csv(file_path, index=False)
        return data
        
    except Exception:
        # Silent fail
        return pd.DataFrame()

# ==========================================
# 2. FEATURE ENGINEERING (Standard)
# ==========================================
def get_features(df, interval):
    if df.empty or len(df) < 50: return pd.DataFrame()
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
# 3. TRAIN OR LOAD MODELS (Auto-Management)
# ==========================================
def ensure_model_exists(stock):
    clean_name = stock.replace(".NS", "")
    path_m = f"model_vault/{clean_name}_30m.txt"
    
    # If model exists, skip training
    if os.path.exists(path_m): return True
    
    print(f"   🧠 Training {clean_name}...", end=" ")
    try:
        # Train Daily
        df = get_data_cached(stock, "10y", "1d")
        df = get_features(df, "1d")
        if df.empty or len(df) < 100:
            print("❌ No data")
            return False
            
        df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.01).astype(int)
        df.dropna(inplace=True)
        
        if len(df) < 50:
            print("❌ Insufficient data")
            return False
            
        cols = [c for c in df.columns if c not in ['ds', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        dtrain = lgb.Dataset(df[cols], label=df['Target'])
        params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 20, 'verbose': -1, 'seed': 42}
        model = lgb.train(params, dtrain, num_boost_round=100)
        model.save_model(f"model_vault/{clean_name}_daily.txt")
        
        # Train Hourly
        df = get_data_cached(stock, "730d", "1h")
        df = get_features(df, "1h")
        if df.empty or len(df) < 100:
            print("❌ No hourly data")
            return False
            
        df['Target'] = (df['Close'].shift(-4) / df['Close'] - 1 > 0.005).astype(int)
        df.dropna(inplace=True)
        
        if len(df) < 50:
            return False
            
        dtrain = lgb.Dataset(df[cols], label=df['Target'])
        model = lgb.train(params, dtrain, num_boost_round=100)
        model.save_model(f"model_vault/{clean_name}_hourly.txt")
        
        # Train 30m
        df = get_data_cached(stock, "59d", "30m")
        df = get_features(df, "30m")
        if df.empty or len(df) < 100:
            print("❌ No 30m data")
            return False
            
        df['Target'] = (df['Close'].shift(-2) / df['Close'] - 1 > 0.002).astype(int)
        df.dropna(inplace=True)
        
        if len(df) < 50:
            return False
            
        dtrain = lgb.Dataset(df[cols], label=df['Target'])
        model = lgb.train(params, dtrain, num_boost_round=100)
        model.save_model(f"model_vault/{clean_name}_30m.txt")
        
        print("✅")
        return True
        
    except Exception as e:
        print(f"❌ {str(e)[:40]}")
        return False

# ==========================================
# 4. THE 100-STOCK SCANNER
# ==========================================
def run_scanner():
    print("="*70)
    print(f"🚀 NIFTY ALPHA SCANNER (Scanning {sum(len(v) for v in UNIVERSE.values())} Stocks)")
    print("="*70)
    
    all_opportunities = []
    all_scores = []  # NEW: Track all scores for debugging
    
    # Flatten universe
    all_stocks = []
    for sector, stocks in UNIVERSE.items():
        for s in stocks: all_stocks.append((s, sector))
        
    print(f"⏳ Processing market data...\n")
    
    for stock, sector in all_stocks:
        clean_name = stock.replace(".NS", "")
        
        # 1. Ensure Models Exist (Auto-Train)
        if not ensure_model_exists(stock): continue
        
        try:
            # 2. Load Models
            md = lgb.Booster(model_file=f"model_vault/{clean_name}_daily.txt")
            mh = lgb.Booster(model_file=f"model_vault/{clean_name}_hourly.txt")
            mm = lgb.Booster(model_file=f"model_vault/{clean_name}_30m.txt")
            
            # 3. Fetch Live Data
            df_d = get_features(get_data_cached(stock, "10y", "1d"), "1d")
            df_h = get_features(get_data_cached(stock, "730d", "1h"), "1h")
            df_m = get_features(get_data_cached(stock, "59d", "30m"), "30m")
            
            if df_d.empty or df_h.empty or df_m.empty: continue
            
            # 4. Get Latest Candle
            feat_d = df_d.iloc[[-1]]
            feat_h = df_h.iloc[[-1]]
            feat_m = df_m.iloc[[-1]]
            
            # Feature Cols (Dynamic)
            drop = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'ds', 'Date', 'Datetime']
            cols = [c for c in feat_d.columns if c not in drop]
            
            # 5. Predict
            pd_prob = md.predict(feat_d[cols])[0]
            ph_prob = mh.predict(feat_h[cols])[0]
            pm_prob = mm.predict(feat_m[cols])[0]
            
            # 6. Scoring Logic
            # Weighted Score: Daily (30%) + Hourly (30%) + 30m (40%)
            final_score = (pd_prob * 0.3) + (ph_prob * 0.3) + (pm_prob * 0.4)
            
            # Track ALL scores for debugging
            all_scores.append({
                'Stock': clean_name,
                'Daily': pd_prob,
                'Hourly': ph_prob,
                '30m': pm_prob,
                'Final': final_score
            })
            
            # RELAXED FILTER (changed from 0.50, 0.50, 0.65 to 0.45, 0.45, 0.55)
            if pd_prob > 0.45 and ph_prob > 0.45 and pm_prob > 0.55:
                price = feat_m['Close'].values[0]
                atr = feat_m['ATR'].values[0]
                
                all_opportunities.append({
                    'Stock': clean_name,
                    'Sector': sector,
                    'Price': price,
                    'Score': final_score * 100,
                    'SL': price - (atr * 1.5),
                    'TP': price + (atr * 3),
                    'Daily': pd_prob,
                    'Hourly': ph_prob,
                    '30m': pm_prob
                })
                
        except Exception:
            pass
            
    # ==========================================
    # 5. THE LEADERBOARD (Ranking)
    # ==========================================
    # Sort by Highest AI Score
    all_opportunities.sort(key=lambda x: x['Score'], reverse=True)
    all_scores.sort(key=lambda x: x['Final'], reverse=True)
    
    print("\n" + "="*90)
    print(f"🏆 TOP 10 HIGH-PROBABILITY TRADES (Sorted by Confidence)")
    print("="*90)
    print(f"{'STOCK':<12} | {'SECTOR':<8} | {'PRICE':<8} | {'SCORE':<6} | {'D':<5} {'H':<5} {'30m':<5} | {'SL':<8} | {'TP':<8}")
    print("-" * 90)
    
    if not all_opportunities:
        print("💤 No setups found matching criteria.")
        print("\n📊 Top 5 Stocks by Score (for reference):")
        for s in all_scores[:5]:
            print(f"   {s['Stock']:<12} | Score: {s['Final']*100:>5.1f}% | D:{s['Daily']:.2f} H:{s['Hourly']:.2f} 30m:{s['30m']:.2f}")
    else:
        for op in all_opportunities[:10]: # SHOW TOP 10
            print(f"{Fore.GREEN}{op['Stock']:<12}{Style.RESET_ALL} | {op['Sector']:<8} | ₹{op['Price']:<7.2f} | {op['Score']:<5.1f}% | {op['Daily']:.2f} {op['Hourly']:.2f} {op['30m']:.2f} | {Fore.RED}₹{op['SL']:<7.2f}{Style.RESET_ALL} | {Fore.GREEN}₹{op['TP']:<7.2f}{Style.RESET_ALL}")
            
    print("="*90)
    print(f"ℹ️ Scanned {len(all_stocks)} stocks. Trained {len(all_scores)} models. Found {len(all_opportunities)} setups.")

if __name__ == "__main__":
    run_scanner()